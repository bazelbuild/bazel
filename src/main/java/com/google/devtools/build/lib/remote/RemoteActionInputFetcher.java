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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
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
    return metadata.isRemote();
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
    checkArgument(metadata.isRemote(), "Cannot download file that is not a remote file.");
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
}
