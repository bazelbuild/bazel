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

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.concurrent.GuardedBy;

/**
 * Stages output files that are stored remotely to the local filesystem.
 *
 * <p>This is necessary for remote caching/execution when {@code
 * --experimental_remote_download_outputs=minimal} is specified.
 */
class RemoteActionInputFetcher implements ActionInputPrefetcher {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Object lock = new Object();

  /** Set of successfully downloaded output files. */
  @GuardedBy("lock")
  private final Set<Path> downloadedPaths = new HashSet<>();

  @VisibleForTesting
  @GuardedBy("lock")
  final Map<Path, ListenableFuture<Void>> downloadsInProgress = new HashMap<>();

  private final String buildRequestId;
  private final String commandId;
  private final RemoteCache remoteCache;
  private final Path execRoot;

  RemoteActionInputFetcher(
      String buildRequestId, String commandId, RemoteCache remoteCache, Path execRoot) {
    this.buildRequestId = Preconditions.checkNotNull(buildRequestId);
    this.commandId = Preconditions.checkNotNull(commandId);
    this.remoteCache = Preconditions.checkNotNull(remoteCache);
    this.execRoot = Preconditions.checkNotNull(execRoot);
  }

  /**
   * Fetches remotely stored action outputs, that are inputs to this spawn, and stores them under
   * their path in the output base.
   *
   * <p>This method blocks until all downloads have finished.
   *
   * <p>This method is safe to be called concurrently from spawn runners before running any local
   * spawn.
   */
  @Override
  public void prefetchFiles(
      Iterable<? extends ActionInput> inputs, MetadataProvider metadataProvider)
      throws IOException, InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "stage remote inputs")) {
      Map<Path, ListenableFuture<Void>> downloadsToWaitFor = new HashMap<>();
      for (ActionInput input : inputs) {
        if (input instanceof VirtualActionInput) {
          if (!(input instanceof EmptyActionInput)) {
            VirtualActionInput virtualActionInput = (VirtualActionInput) input;
            Path outputPath = execRoot.getRelative(virtualActionInput.getExecPath());
            SandboxHelpers.atomicallyWriteVirtualInput(virtualActionInput, outputPath, ".remote");
          }
        } else {
          FileArtifactValue metadata = metadataProvider.getMetadata(input);
          if (metadata == null || !metadata.isRemote()) {
            continue;
          }

          Path path = execRoot.getRelative(input.getExecPath());
          synchronized (lock) {
            if (downloadedPaths.contains(path)) {
              continue;
            }
            ListenableFuture<Void> download = downloadFileAsync(path, metadata);
            downloadsToWaitFor.putIfAbsent(path, download);
          }
        }
      }

      try {
        RemoteCache.waitForBulkTransfer(
            downloadsToWaitFor.values(), /* cancelRemainingOnInterrupt=*/ true);
      } catch (BulkTransferException e) {
        if (e.onlyCausedByCacheNotFoundException()) {
          BulkTransferException bulkAnnotatedException = new BulkTransferException();
          for (Throwable t : e.getSuppressed()) {
            IOException annotatedException =
                new IOException(
                    String.format(
                        "Failed to fetch file with hash '%s' because it does not exist remotely."
                            + " --remote_download_outputs=minimal does not work if"
                            + " your remote cache evicts files during builds.",
                        ((CacheNotFoundException) t).getMissingDigest().getHash()));
            bulkAnnotatedException.add(annotatedException);
          }
          e = bulkAnnotatedException;
        }
        throw e;
      }
    }
  }

  ImmutableSet<Path> downloadedFiles() {
    synchronized (lock) {
      return ImmutableSet.copyOf(downloadedPaths);
    }
  }

  void downloadFile(Path path, FileArtifactValue metadata)
      throws IOException, InterruptedException {
    Utils.getFromFuture(downloadFileAsync(path, metadata));
  }

  private ListenableFuture<Void> downloadFileAsync(Path path, FileArtifactValue metadata)
      throws IOException {
    synchronized (lock) {
      if (downloadedPaths.contains(path)) {
        return Futures.immediateFuture(null);
      }

      ListenableFuture<Void> download = downloadsInProgress.get(path);
      if (download == null) {
        RequestMetadata requestMetadata =
            TracingMetadataUtils.buildMetadata(buildRequestId, commandId, metadata.getActionId());
        RemoteActionExecutionContext context = RemoteActionExecutionContext.create(requestMetadata);

        Digest digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
        download = remoteCache.downloadFile(context, path, digest);
        downloadsInProgress.put(path, download);
        Futures.addCallback(
            download,
            new FutureCallback<Void>() {
              @Override
              public void onSuccess(Void v) {
                synchronized (lock) {
                  downloadsInProgress.remove(path);
                  downloadedPaths.add(path);
                }

                try {
                  path.chmod(0755);
                } catch (IOException e) {
                  logger.atWarning().withCause(e).log("Failed to chmod 755 on %s", path);
                }
              }

              @Override
              public void onFailure(Throwable t) {
                synchronized (lock) {
                  downloadsInProgress.remove(path);
                }
                try {
                  path.delete();
                } catch (IOException e) {
                  logger.atWarning().withCause(e).log(
                      "Failed to delete output file after incomplete download: %s", path);
                }
              }
            },
            MoreExecutors.directExecutor());
      }
      return download;
    }
  }
}
