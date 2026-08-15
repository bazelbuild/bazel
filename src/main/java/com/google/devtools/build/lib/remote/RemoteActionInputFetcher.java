// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Stages output files that are stored remotely to the local filesystem.
 *
 * <p>This is used to ensure that the inputs to a local action are present, even when they are
 * provided by a remote action when building without the bytes, or by an external repository when
 * building with a remote repository cache enabled.
 */
public class RemoteActionInputFetcher extends AbstractActionInputPrefetcher {

  private final String buildRequestId;
  private final String commandId;
  private final CombinedCache combinedCache;
  private final ConcurrentArtifactPathTrie rewoundActionOutputs = new ConcurrentArtifactPathTrie();

  RemoteActionInputFetcher(
      Reporter reporter,
      String buildRequestId,
      String commandId,
      CombinedCache combinedCache,
      Path execRoot,
      TempPathGenerator tempPathGenerator,
      RemoteOutputChecker remoteOutputChecker,
      @Nullable ActionOutputDirectoryHelper outputDirectoryHelper,
      OutputPermissions outputPermissions) {
    super(
        reporter,
        execRoot,
        tempPathGenerator,
        remoteOutputChecker,
        outputDirectoryHelper,
        outputPermissions);
    this.buildRequestId = Preconditions.checkNotNull(buildRequestId);
    this.commandId = Preconditions.checkNotNull(commandId);
    this.combinedCache = Preconditions.checkNotNull(combinedCache);
  }

  @Override
  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {
    input.atomicallyWriteRelativeTo(execRoot);
  }

  @Override
  protected boolean canDownloadFile(Path path, FileArtifactValue metadata) {
    // Only files and directories have remote-only content that can be downloaded.
    if (metadata.getType() != FileStateType.REGULAR_FILE
        && metadata.getType() != FileStateType.DIRECTORY) {
      return false;
    }
    // When action rewinding is enabled, an action that had remote metadata at some point during the
    // build may have been re-executed locally to regenerate lost inputs, but may then be rewound
    // again and thus have its (now local) outputs deleted. In this case, we need to download the
    // outputs again, even if they are now considered local.
    return metadata.isRemote() || (forceRefetch(path) && !path.exists(Symlinks.NOFOLLOW));
  }

  @Override
  protected boolean forceRefetch(Path path) {
    // Caches for download operations and output directory creation need to be disregarded for the
    // outputs of rewound actions as they may have been deleted after they were first created.
    // Compare as fragments since execRoot may be located on a file system overlaying the host file
    // system where downloads are written to.
    PathFragment execRootFragment = execRoot.asFragment();
    PathFragment pathFragment = path.asFragment();
    return pathFragment.startsWith(execRootFragment)
        && rewoundActionOutputs.contains(pathFragment.relativeTo(execRootFragment));
  }

  @Override
  protected ListenableFuture<Void> doDownloadFile(
      @Nullable ActionExecutionMetadata action,
      Reporter reporter,
      ActionInput input,
      Path tempPath,
      FileArtifactValue metadata,
      Priority priority,
      Reason reason)
      throws IOException {
    RequestMetadata requestMetadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId,
            commandId,
            switch (reason) {
              case INPUTS -> "input";
              case OUTPUTS -> "output";
            },
            action != null ? action.getMnemonic() : null,
            action != null && action.getOwner().getLabel() != null
                ? action.getOwner().getLabel().getCanonicalForm()
                : null,
            action != null ? action.getOwner().getConfigurationChecksum() : null);
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(requestMetadata);

    Digest digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());

    // Treat other download error as CacheNotFoundException so that Bazel can
    // correctly rewind the action/build.
    // Intentionally, do not transform IOExceptions directly thrown by downloadFile rather than in
    // the returned future, as those are likely to be caused by local FS issues.
    return Futures.catchingAsync(
        combinedCache.downloadFile(
            context,
            input.getExecPathString(),
            input.getExecPath(),
            tempPath.forHostFileSystem(),
            digest,
            new CombinedCache.DownloadProgressReporter(
                progress -> {
                  if (action != null) {
                    progress.postTo(reporter, action);
                  }
                },
                input.getExecPathString(),
                digest.getSizeBytes())),
        IOException.class,
        e ->
            immediateFailedFuture(
                switch (e) {
                  case CacheNotFoundException cacheNotFoundException -> cacheNotFoundException;
                  default -> {
                    var cacheNotFoundException =
                        new CacheNotFoundException(digest, input.getExecPath());
                    cacheNotFoundException.addSuppressed(e);
                    yield cacheNotFoundException;
                  }
                }),
        directExecutor());
  }

  public void handleRewoundActionOutputs(Collection<Artifact> outputs) {
    // SkyframeActionExecutor#prepareForRewinding does *not* call this method because the
    // RemoteActionFileSystem corresponds to an ActionFileSystemType with inMemoryFileSystem() ==
    // true. While it is true that resetting outputDirectoryHelper isn't necessary to undo the
    // caching of output directory creation during action preparation, we still need to reset here
    // since outputDirectoryHelper is also used by AbstractActionInputPrefetcher.
    outputDirectoryHelper.invalidateTreeArtifactDirectoryCreation(outputs);
    var exactPaths = ImmutableList.<PathFragment>builder();
    var treePaths = ImmutableList.<PathFragment>builder();
    for (Artifact output : trackedRewoundOutputs(outputs)) {
      rewoundActionOutputs.add(output);
      (output.isTreeArtifact() ? treePaths : exactPaths).add(output.getExecPath());
    }
    // The outputs are about to be regenerated, so whatever the prefetcher remembers about their
    // previous incarnation is stale. forceRefetch covers this while the trie entries are present,
    // but they are dropped again by finishRewoundActionOutputs, after which the caches are
    // consulted normally. Nothing can be prefetching these outputs concurrently: consumers hold a
    // read lock on the rewound action in RemoteRewoundActionSynchronizer.
    invalidateDownloads(exactPaths.build());
    invalidateDownloadsUnder(treePaths.build());
  }

  /**
   * Stops forcing refetches of a rewound action's outputs, which have been regenerated by the time
   * this is called.
   */
  public void finishRewoundActionOutputs(Collection<Artifact> outputs) {
    for (Artifact output : trackedRewoundOutputs(outputs)) {
      rewoundActionOutputs.remove(output);
    }
  }

  /**
   * Action templates have {@link Artifact.TreeFileArtifact}s as outputs, which aren't supported by
   * the trie. Only the tree artifacts themselves need to be tracked.
   */
  private static Iterable<Artifact> trackedRewoundOutputs(Collection<Artifact> outputs) {
    return Iterables.transform(
        outputs,
        output -> output instanceof Artifact.TreeFileArtifact ? output.getParent() : output);
  }
}
