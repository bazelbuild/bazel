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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.remote.util.TempPathGenerator;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.functions.Function;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Abstract implementation of {@link ActionInputPrefetcher} which implements the orchestration of
 * prefeching multiple inputs so subclasses can focus on prefetching / downloading single input.
 */
public abstract class AbstractActionInputPrefetcher implements ActionInputPrefetcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AsyncTaskCache.NoResult<Path> downloadCache = AsyncTaskCache.NoResult.create();
  private final TempPathGenerator tempPathGenerator;

  protected final Path execRoot;

  protected AbstractActionInputPrefetcher(Path execRoot, TempPathGenerator tempPathGenerator) {
    this.execRoot = execRoot;
    this.tempPathGenerator = tempPathGenerator;
  }

  public void startBuild(EventHandler eventHandler) throws AbruptExitException {
    Path tempDir = tempPathGenerator.getTempDir();
    if (tempDir.exists()) {
      eventHandler.handle(Event.warn("Found stale downloads from previous build, deleting..."));
      try {
        tempDir.deleteTree();
      } catch (IOException e) {
        throw new AbruptExitException(
            DetailedExitCode.of(
                ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
                FailureDetail.newBuilder()
                    .setMessage(
                        String.format("Failed to delete stale downloads: %s", e.getMessage()))
                    .setRemoteExecution(
                        RemoteExecution.newBuilder()
                            .setCode(Code.DOWNLOADED_INPUTS_DELETION_FAILURE))
                    .build()));
      }
    }
  }

  public void finalizeBuild() {
    Path tempDir = tempPathGenerator.getTempDir();
    try {
      tempDir.deleteTree();
    } catch (IOException ignored) {
      // Intentionally left empty.
    }
  }

  protected abstract boolean shouldDownloadInput(
      ActionInput input, @Nullable FileArtifactValue metadata);

  /**
   * Downloads the {@code input} to the given path via the metadata.
   *
   * @param path the destination which the input should be written to.
   */
  protected abstract ListenableFuture<Void> downloadInput(
      Path path, ActionInput input, FileArtifactValue metadata) throws IOException;

  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {}

  /** Transforms the error encountered during the prefetch . */
  protected Completable onErrorResumeNext(Throwable error) {
    return Completable.error(error);
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
            .flatMapSingle(input -> toTransferResult(prefetchInput(metadataProvider, input)));
    Completable prefetch = mergeBulkTransfer(transfers).onErrorResumeNext(this::onErrorResumeNext);
    Completable prefetchWithProfiler =
        Completable.using(
            () -> Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "stage remote inputs"),
            profiler -> prefetch,
            SilentCloseable::close);
    return toListenableFuture(prefetchWithProfiler);
  }

  private Completable prefetchInput(MetadataProvider metadataProvider, ActionInput input)
      throws IOException {
    if (input instanceof VirtualActionInput) {
      prefetchVirtualActionInput((VirtualActionInput) input);
      return Completable.complete();
    }

    FileArtifactValue metadata = metadataProvider.getMetadata(input);
    if (!shouldDownloadInput(input, metadata)) {
      return Completable.complete();
    }

    Path path = execRoot.getRelative(input.getExecPath());
    return downloadFileIfNot(path, (p) -> downloadInput(p, input, metadata));
  }

  /**
   * Downloads file into the {@code path} with given downloader.
   *
   * <p>The file will be written into a temporary file and moved to the final destination after the
   * download finished.
   */
  protected Completable downloadFileIfNot(
      Path path, Function<Path, ListenableFuture<Void>> downloader) {
    AtomicBoolean completed = new AtomicBoolean(false);
    Completable download =
        Completable.using(
            tempPathGenerator::generateTempPath,
            tempPath ->
                toCompletable(() -> downloader.apply(tempPath), directExecutor())
                    .doOnComplete(
                        () -> {
                          finalizeDownload(tempPath, path);
                          completed.set(true);
                        }),
            tempPath -> {
              if (!completed.get()) {
                deletePartialDownload(tempPath);
              }
            },
            // Set eager=false here because we want cleanup the download *after* upstream is
            // disposed.
            /* eager= */ false);
    return downloadCache.executeIfNot(path, download);
  }

  private void finalizeDownload(Path tmpPath, Path path) throws IOException {
    // The permission of output file is changed to 0555 after action execution. We manually change
    // the permission here for the downloaded file to keep this behaviour consistent.
    tmpPath.chmod(0555);
    FileSystemUtils.moveFile(tmpPath, path);
  }

  private void deletePartialDownload(Path path) {
    try {
      path.delete();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to delete output file after incomplete download: %s", path);
    }
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
}
