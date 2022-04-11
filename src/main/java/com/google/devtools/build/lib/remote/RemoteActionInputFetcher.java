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

import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
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
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.vfs.Path;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;

/**
 * Stages output files that are stored remotely to the local filesystem.
 *
 * <p>This is necessary for remote caching/execution when {@code
 * --experimental_remote_download_outputs=minimal} is specified.
 */
class RemoteActionInputFetcher implements ActionInputPrefetcher {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private final AsyncTaskCache.NoResult<Path> downloadCache = AsyncTaskCache.NoResult.create();

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
   * <p>This method is safe to be called concurrently from spawn runners before running any local
   * spawn.
   *
   * @return a future that is completed once all downloads have finished.
   */
  @Override
  public ListenableFuture<Void> prefetchFiles(
      Iterable<? extends ActionInput> inputs, MetadataProvider metadataProvider) {
    Flowable<TransferResult> transfers =
        Flowable.fromIterable(inputs)
            .flatMapSingle(input -> downloadOneInput(metadataProvider, input));
    Completable prefetch =
        mergeBulkTransfer(transfers)
            .onErrorResumeNext(
                e -> {
                  if (e instanceof BulkTransferException) {
                    if (((BulkTransferException) e).onlyCausedByCacheNotFoundException()) {
                      BulkTransferException bulkAnnotatedException = new BulkTransferException();
                      for (Throwable t : e.getSuppressed()) {
                        IOException annotatedException =
                            new IOException(
                                String.format(
                                    "Failed to fetch file with hash '%s' because it does not"
                                        + " exist remotely. --remote_download_outputs=minimal"
                                        + " does not work if your remote cache evicts files"
                                        + " during builds.",
                                    ((CacheNotFoundException) t).getMissingDigest().getHash()));
                        bulkAnnotatedException.add(annotatedException);
                      }
                      e = bulkAnnotatedException;
                    }
                  }
                  return Completable.error(e);
                });
    Completable prefetchWithProfiler =
        Completable.using(
            () -> Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "stage remote inputs"),
            profiler -> prefetch,
            SilentCloseable::close);
    return toListenableFuture(prefetchWithProfiler);
  }

  private Single<TransferResult> downloadOneInput(
      MetadataProvider metadataProvider, ActionInput input) {
    return toTransferResult(
        Completable.defer(
            () -> {
              if (input instanceof VirtualActionInput) {
                if (!(input instanceof EmptyActionInput)) {
                  VirtualActionInput virtualActionInput = (VirtualActionInput) input;
                  Path outputPath = execRoot.getRelative(virtualActionInput.getExecPath());
                  SandboxHelpers.atomicallyWriteVirtualInput(
                      virtualActionInput, outputPath, ".remote");
                }
              } else {
                FileArtifactValue metadata = metadataProvider.getMetadata(input);
                if (metadata != null && metadata.isRemote()) {
                  Path path = execRoot.getRelative(input.getExecPath());
                  return downloadFileAsync(path, metadata);
                }
              }

              return Completable.complete();
            }));
  }

  ImmutableSet<Path> downloadedFiles() {
    return downloadCache.getFinishedTasks();
  }

  ImmutableSet<Path> downloadsInProgress() {
    return downloadCache.getInProgressTasks();
  }

  @VisibleForTesting
  AsyncTaskCache.NoResult<Path> getDownloadCache() {
    return downloadCache;
  }

  void downloadFile(Path path, FileArtifactValue metadata)
      throws IOException, InterruptedException {
    Utils.getFromFuture(toListenableFuture(downloadFileAsync(path, metadata)));
  }

  private Completable downloadFileAsync(Path path, FileArtifactValue metadata) {
    Completable download =
        RxFutures.toCompletable(
                () -> {
                  RequestMetadata requestMetadata =
                      TracingMetadataUtils.buildMetadata(
                          buildRequestId, commandId, metadata.getActionId(), null);
                  RemoteActionExecutionContext context =
                      RemoteActionExecutionContext.create(requestMetadata);

                  Digest digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());

                  return remoteCache.downloadFile(context, path, digest);
                },
                MoreExecutors.directExecutor())
            .doOnComplete(() -> finalizeDownload(path))
            .doOnError(error -> deletePartialDownload(path))
            .doOnDispose(() -> deletePartialDownload(path));

    return downloadCache.executeIfNot(path, download);
  }

  private void finalizeDownload(Path path) {
    try {
      // The permission of output file is changed to 0555 after action execution. We manually change
      // the permission here
      // for the downloaded file to keep this behaviour consistent.
      path.chmod(0555);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to chmod 555 on %s", path);
    }
  }

  private void deletePartialDownload(Path path) {
    try {
      path.delete();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to delete output file after incomplete download: %s", path);
    }
  }
}
