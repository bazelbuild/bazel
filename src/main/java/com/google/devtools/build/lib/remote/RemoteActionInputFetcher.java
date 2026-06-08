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
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DelegatingPairInputMetadataProvider;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
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
  public ListenableFuture<Void> prefetchFiles(
      @Nullable ActionExecutionMetadata action,
      @Nullable Spawn spawn,
      Supplier<Iterable<? extends ActionInput>> expandedInputs,
      InputMetadataProvider metadataProvider,
      Priority priority,
      Reason reason) {
    if (reason != Reason.INPUTS
        || !(execRoot.getFileSystem()
            instanceof InjectedRepoFileMetadataSupplier repoFileMetadataSupplier)) {
      return super.prefetchFiles(
          action, spawn, expandedInputs, metadataProvider, priority, reason);
    }
    // Materializing a repo's source files for a local action must match the materialization that
    // happens when another repo rule accesses the same repo (see
    // RemoteExternalOverlayFileSystem#doMaterialize): both should recreate the repo's symlinks per
    // hop and download their targets. The generic prefetcher otherwise diverges because the
    // InputMetadataProvider derived by Skyframe resolves repo source symlinks to their targets,
    // causing the prefetcher to collapse them instead of recreating them. To make both paths
    // consistent:
    //   1. Override the metadata of repo source files with the overlay's own (symlink-preserving)
    //      metadata, so that the prefetcher recreates symlinks via plantUnresolvedSymlink.
    //   2. Expand each repo symlink to its target chain, so that the targets - which need not be
    //      action inputs themselves - are materialized alongside the symlinks pointing at them.
    List<ActionInput> expandedInputsWithSymlinkTargets;
    try {
      expandedInputsWithSymlinkTargets =
          expandRepoSymlinkChains(expandedInputs.get(), repoFileMetadataSupplier);
    } catch (IOException e) {
      return immediateFailedFuture(e);
    }
    InputMetadataProvider repoAwareMetadataProvider =
        new DelegatingPairInputMetadataProvider(
            new RepoFileMetadataProvider(repoFileMetadataSupplier), metadataProvider);
    return super.prefetchFiles(
        action,
        spawn,
        () -> expandedInputsWithSymlinkTargets,
        repoAwareMetadataProvider,
        priority,
        reason);
  }

  /**
   * Returns the given inputs, with each input that is a symlink in an injected (not yet
   * materialized) external repo followed hop by hop, adding every intermediate symlink and the final
   * regular file as additional inputs. This ensures that the targets of repo symlinks, which need
   * not be action inputs themselves, are materialized alongside the symlinks.
   */
  private static List<ActionInput> expandRepoSymlinkChains(
      Iterable<? extends ActionInput> inputs,
      InjectedRepoFileMetadataSupplier repoFileMetadataSupplier)
      throws IOException {
    List<ActionInput> result = new ArrayList<>();
    Set<PathFragment> addedTargets = new HashSet<>();
    for (ActionInput input : inputs) {
      result.add(input);
      PathFragment path = repoFilePath(input);
      if (path == null) {
        continue;
      }
      while (true) {
        FileArtifactValue metadata = repoFileMetadataSupplier.getInjectedRepoFileMetadata(path);
        if (metadata == null || metadata.getType() != FileStateType.SYMLINK) {
          break;
        }
        PathFragment target = resolveSymlinkTarget(path, metadata.getUnresolvedSymlinkTarget());
        if (!addedTargets.add(target)) {
          break;
        }
        result.add(ActionInputHelper.fromPath(target));
        path = target;
      }
    }
    return result;
  }

  /**
   * Returns the absolute path of the given input as seen by the overlay file system, or null if it
   * does not have one.
   */
  @Nullable
  private static PathFragment repoFilePath(ActionInput input) {
    if (input instanceof Artifact artifact) {
      return artifact.getPath().asFragment();
    }
    return input.getExecPath().isAbsolute() ? input.getExecPath() : null;
  }

  private static PathFragment resolveSymlinkTarget(PathFragment linkPath, String target) {
    PathFragment targetFragment = PathFragment.create(target);
    return targetFragment.isAbsolute()
        ? targetFragment
        : linkPath.getParentDirectory().getRelative(targetFragment);
  }

  /**
   * An {@link InputMetadataProvider} that reports the symlink-preserving metadata of injected
   * external repo source files and defers to a delegate (via {@link
   * DelegatingPairInputMetadataProvider}) for everything else.
   */
  private static final class RepoFileMetadataProvider implements InputMetadataProvider {
    private final InjectedRepoFileMetadataSupplier repoFileMetadataSupplier;

    RepoFileMetadataProvider(InjectedRepoFileMetadataSupplier repoFileMetadataSupplier) {
      this.repoFileMetadataSupplier = repoFileMetadataSupplier;
    }

    @Nullable
    @Override
    public FileArtifactValue getInputMetadataChecked(ActionInput input) throws IOException {
      PathFragment path = repoFilePath(input);
      return path == null ? null : repoFileMetadataSupplier.getInjectedRepoFileMetadata(path);
    }

    @Nullable
    @Override
    public TreeArtifactValue getTreeMetadata(ActionInput input) {
      return null;
    }

    @Nullable
    @Override
    public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
      return null;
    }

    @Nullable
    @Override
    public FilesetOutputTree getFileset(ActionInput input) {
      return null;
    }

    @Override
    public Map<Artifact, FilesetOutputTree> getFilesets() {
      return ImmutableMap.of();
    }

    @Nullable
    @Override
    public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
      return null;
    }

    @Override
    public ImmutableList<RunfilesTree> getRunfilesTrees() {
      return ImmutableList.of();
    }

    @Nullable
    @Override
    public ActionInput getInput(PathFragment execPath) {
      return null;
    }
  }

  @Override
  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {
    input.atomicallyWriteRelativeTo(execRoot);
  }

  @Override
  protected boolean canDownloadFile(Path path, FileArtifactValue metadata) {
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
    return path.startsWith(execRoot) && rewoundActionOutputs.contains(path.relativeTo(execRoot));
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
            action);
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
    for (Artifact output : outputs) {
      // Action templates have TreeFileArtifacts as outputs, which isn't supported by the trie. We
      // only need to track the tree artifacts themselves.
      if (output instanceof Artifact.TreeFileArtifact) {
        rewoundActionOutputs.add(output.getParent());
      } else {
        rewoundActionOutputs.add(output);
      }
    }
  }
}
