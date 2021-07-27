// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.common.ProgressStatusListener.NO_ACTION;
import static com.google.devtools.build.lib.remote.util.Utils.bytesCountToDisplayString;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.RemoteUploadEvent;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnProgressEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.LazyFileOutputStream;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.ProgressStatusListener;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import io.reactivex.rxjava3.core.Completable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * A cache for storing artifacts (input and output) as well as the output of running an action.
 *
 * <p>The cache is reference counted. Initially, the reference count is 1. Use {@link #retain()} to
 * increase and {@link #release()} to decrease the reference count respectively. Once the reference
 * count is reached to 0, the underlying resources will be released (after network I/Os finished).
 *
 * <p>Use {@link #awaitTermination()} to wait for the underlying network I/Os to finish. Use {@link
 * #shutdownNow()} to cancel all active network I/Os and reject new requests.
 */
@ThreadSafety.ThreadSafe
public class RemoteCache extends AbstractReferenceCounted {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** See {@link SpawnExecutionContext#lockOutputFiles()}. */
  @FunctionalInterface
  interface OutputFilesLocker {
    void lock() throws InterruptedException;
  }

  private static final ListenableFuture<Void> COMPLETED_SUCCESS = immediateFuture(null);
  private static final ListenableFuture<byte[]> EMPTY_BYTES = immediateFuture(new byte[0]);

  private final Reporter reporter;
  private final CountDownLatch closedLatch = new CountDownLatch(1);
  private final AtomicBoolean closed = new AtomicBoolean(false);
  private final AsyncTaskCache.NoResult<Digest> casUploadCache = AsyncTaskCache.NoResult.create();

  protected final RemoteCacheClient cacheProtocol;
  protected final RemoteOptions options;
  protected final DigestUtil digestUtil;

  public RemoteCache(
      Reporter reporter,
      RemoteCacheClient cacheProtocol,
      RemoteOptions options,
      DigestUtil digestUtil) {
    this.reporter = reporter;
    this.cacheProtocol = cacheProtocol;
    this.options = options;
    this.digestUtil = digestUtil;
  }

  public ActionResult downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr)
      throws IOException, InterruptedException {
    return getFromFuture(cacheProtocol.downloadActionResult(context, actionKey, inlineOutErr));
  }

  public final ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests) {
    checkState(!closed.get(), "closed");

    Set<Digest> digestsInProgress = casUploadCache.getInProgressTasks();
    // Since this is a separate call, we may have digest that is in the digestsInProgress above and
    // then finishes resulting a duplicate digest in digestsUploaded. However, this race is fine for
    // our purpose as long as we check "in progress" before "uploaded".
    Set<Digest> digestsUploaded = casUploadCache.getFinishedTasks();

    Set<Digest> digestsRequested =
        StreamSupport.stream(digests.spliterator(), false).collect(Collectors.toSet());

    // Find digests that are neither in the progress of upload nor already uploaded.
    Set<Digest> digestsToUpload =
        digestsRequested.stream()
            .filter(
                digest -> !digestsInProgress.contains(digest) && !digestsUploaded.contains(digest))
            .collect(Collectors.toSet());

    ListenableFuture<ImmutableSet<Digest>> missingDigestsFuture;
    if (Iterables.isEmpty(digestsToUpload)) {
      missingDigestsFuture = immediateFuture(ImmutableSet.of());
    } else {
      missingDigestsFuture = cacheProtocol.findMissingDigests(context, digestsToUpload);
    }

    // Combine digests that are being uploaded and that are missing from cache.
    return Futures.transform(
        missingDigestsFuture,
        missingDigests -> {
          ImmutableSet.Builder<Digest> builder = ImmutableSet.builder();
          for (Digest digest : digestsInProgress) {
            if (digestsRequested.contains(digest)) {
              builder.add(digest);
            }
          }
          builder.addAll(missingDigests);
          return builder.build();
        },
        directExecutor());
  }

  public ListenableFuture<Void> uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult) {
    checkState(!closed.get(), "closed");

    String resourceId = "ac/" + actionKey.getDigest().getHash();
    String progress = "Uploading action result " + actionKey.getDigest().getHash();

    Completable completable =
        RxFutures.toCompletable(
                () -> cacheProtocol.uploadActionResult(context, actionKey, actionResult),
                directExecutor())
            .doOnSubscribe(
                disposable ->
                    reporter.post(
                        RemoteUploadEvent.create(resourceId, progress, /* finished= */ false)))
            .doFinally(
                () ->
                    reporter.post(
                        RemoteUploadEvent.create(resourceId, progress, /* finished= */ true)));

    return RxFutures.toListenableFuture(completable);
  }

  /**
   * Upload a local file to the remote cache.
   *
   * @param context the context for the action.
   * @param digest the digest of the file.
   * @param file the file to upload.
   */
  public final ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context, Digest digest, Path file) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    String resourceId = "cas/" + digest.getHash();
    String progress = "Uploading file " + file.getPathString();

    Completable upload =
        casUploadCache.executeIfNot(
            digest,
            RxFutures.toCompletable(
                    () -> cacheProtocol.uploadFile(context, digest, file), directExecutor())
                .doOnSubscribe(
                    disposable ->
                        reporter.post(
                            RemoteUploadEvent.create(resourceId, progress, /* finished= */ false)))
                .doFinally(
                    () ->
                        reporter.post(
                            RemoteUploadEvent.create(resourceId, progress, /* finished= */ true))));
    return RxFutures.toListenableFuture(upload);
  }

  /**
   * Upload sequence of bytes to the remote cache.
   *
   * @param context the context for the action.
   * @param digest the digest of the file.
   * @param data the BLOB to upload.
   */
  public final ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    String resourceId = "cas/" + digest.getHash();
    String progress = "Uploading blob " + digest.getHash();

    Completable upload =
        casUploadCache.executeIfNot(
            digest,
            RxFutures.toCompletable(
                    () -> cacheProtocol.uploadBlob(context, digest, data), directExecutor())
                .doOnSubscribe(
                    disposable ->
                        reporter.post(
                            RemoteUploadEvent.create(resourceId, progress, /* finished= */ false)))
                .doFinally(
                    () ->
                        reporter.post(
                            RemoteUploadEvent.create(resourceId, progress, /* finished= */ true))));
    return RxFutures.toListenableFuture(upload);
  }

  public static void waitForBulkTransfer(
      Iterable<? extends ListenableFuture<?>> transfers, boolean cancelRemainingOnInterrupt)
      throws BulkTransferException, InterruptedException {
    BulkTransferException bulkTransferException = null;
    InterruptedException interruptedException = null;
    boolean interrupted = Thread.currentThread().isInterrupted();
    for (ListenableFuture<?> transfer : transfers) {
      try {
        if (interruptedException == null) {
          // Wait for all transfers to finish.
          getFromFuture(transfer, cancelRemainingOnInterrupt);
        } else {
          transfer.cancel(true);
        }
      } catch (IOException e) {
        if (bulkTransferException == null) {
          bulkTransferException = new BulkTransferException();
        }
        bulkTransferException.add(e);
      } catch (InterruptedException e) {
        interrupted = Thread.interrupted() || interrupted;
        interruptedException = e;
        if (!cancelRemainingOnInterrupt) {
          // leave the rest of the transfers alone
          break;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    if (interruptedException != null) {
      if (bulkTransferException != null) {
        interruptedException.addSuppressed(bulkTransferException);
      }
      throw interruptedException;
    }
    if (bulkTransferException != null) {
      throw bulkTransferException;
    }
  }

  /**
   * Downloads a blob with content hash {@code digest} and stores its content in memory.
   *
   * @return a future that completes after the download completes (succeeds / fails). If successful,
   *     the content is stored in the future's {@code byte[]}.
   */
  public ListenableFuture<byte[]> downloadBlob(
      RemoteActionExecutionContext context, Digest digest) {
    if (digest.getSizeBytes() == 0) {
      return EMPTY_BYTES;
    }
    ByteArrayOutputStream bOut = new ByteArrayOutputStream((int) digest.getSizeBytes());
    SettableFuture<byte[]> outerF = SettableFuture.create();
    Futures.addCallback(
        cacheProtocol.downloadBlob(context, digest, bOut),
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void aVoid) {
            try {
              outerF.set(bOut.toByteArray());
            } catch (RuntimeException e) {
              logger.atWarning().withCause(e).log("Unexpected exception");
              outerF.setException(e);
            }
          }

          @Override
          public void onFailure(Throwable t) {
            outerF.setException(t);
          }
        },
        directExecutor());
    return outerF;
  }

  private ListenableFuture<Void> downloadBlob(
      RemoteActionExecutionContext context, Digest digest, OutputStream out) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    return cacheProtocol.downloadBlob(context, digest, out);
  }

  /** A reporter that reports download progresses. */
  public static class DownloadProgressReporter {
    private static final Pattern PATTERN = Pattern.compile("^bazel-out/[^/]+/[^/]+/");
    private final ProgressStatusListener listener;
    private final String id;
    private final String file;
    private final String totalSize;
    private final AtomicLong downloadedBytes = new AtomicLong(0);

    public DownloadProgressReporter(ProgressStatusListener listener, String file, long totalSize) {
      this.listener = listener;
      this.id = file;
      this.totalSize = bytesCountToDisplayString(totalSize);

      Matcher matcher = PATTERN.matcher(file);
      this.file = matcher.replaceFirst("");
    }

    void started() {
      reportProgress(false, false);
    }

    void downloadedBytes(int count) {
      downloadedBytes.addAndGet(count);
      reportProgress(true, false);
    }

    void finished() {
      reportProgress(true, true);
    }

    private void reportProgress(boolean includeBytes, boolean finished) {
      String progress;
      if (includeBytes) {
        progress =
            String.format(
                "Downloading %s, %s / %s",
                file, bytesCountToDisplayString(downloadedBytes.get()), totalSize);
      } else {
        progress = String.format("Downloading %s", file);
      }
      listener.onProgressStatus(SpawnProgressEvent.create(id, progress, finished));
    }
  }

  public ListenableFuture<Void> downloadFile(
      RemoteActionExecutionContext context,
      String outputPath,
      Path localPath,
      Digest digest,
      DownloadProgressReporter reporter)
      throws IOException {
    SettableFuture<Void> outerF = SettableFuture.create();
    ListenableFuture<Void> f = downloadFile(context, localPath, digest, reporter);
    Futures.addCallback(
        f,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void unused) {
            outerF.set(null);
          }

          @Override
          public void onFailure(Throwable throwable) {
            if (throwable instanceof OutputDigestMismatchException) {
              OutputDigestMismatchException e = ((OutputDigestMismatchException) throwable);
              e.setOutputPath(outputPath);
              e.setLocalPath(localPath);
            }
            outerF.setException(throwable);
          }
        },
        MoreExecutors.directExecutor());

    return outerF;
  }

  /** Downloads a file (that is not a directory). The content is fetched from the digest. */
  public ListenableFuture<Void> downloadFile(
      RemoteActionExecutionContext context, Path path, Digest digest) throws IOException {
    return downloadFile(context, path, digest, new DownloadProgressReporter(NO_ACTION, "", 0));
  }

  /** Downloads a file (that is not a directory). The content is fetched from the digest. */
  public ListenableFuture<Void> downloadFile(
      RemoteActionExecutionContext context,
      Path path,
      Digest digest,
      DownloadProgressReporter reporter)
      throws IOException {
    Preconditions.checkNotNull(path.getParentDirectory()).createDirectoryAndParents();
    if (digest.getSizeBytes() == 0) {
      // Handle empty file locally.
      FileSystemUtils.writeContent(path, new byte[0]);
      return COMPLETED_SUCCESS;
    }

    if (!options.remoteDownloadSymlinkTemplate.isEmpty()) {
      // Don't actually download files from the CAS. Instead, create a
      // symbolic link that points to a location where CAS objects may
      // be found. This could, for example, be a FUSE file system.
      path.createSymbolicLink(
          path.getRelative(
              options
                  .remoteDownloadSymlinkTemplate
                  .replace("{hash}", digest.getHash())
                  .replace("{size_bytes}", String.valueOf(digest.getSizeBytes()))));
      return COMPLETED_SUCCESS;
    }

    reporter.started();
    OutputStream out = new ReportingOutputStream(new LazyFileOutputStream(path), reporter);

    SettableFuture<Void> outerF = SettableFuture.create();
    ListenableFuture<Void> f = cacheProtocol.downloadBlob(context, digest, out);
    Futures.addCallback(
        f,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void result) {
            try {
              out.close();
              outerF.set(null);
              reporter.finished();
            } catch (IOException e) {
              outerF.setException(e);
            } catch (RuntimeException e) {
              logger.atWarning().withCause(e).log("Unexpected exception");
              outerF.setException(e);
            }
          }

          @Override
          public void onFailure(Throwable t) {
            try {
              out.close();
              reporter.finished();
            } catch (IOException e) {
              if (t != e) {
                t.addSuppressed(e);
              }
            } catch (RuntimeException e) {
              logger.atWarning().withCause(e).log("Unexpected exception");
              t.addSuppressed(e);
            } finally {
              outerF.setException(t);
            }
          }
        },
        directExecutor());
    return outerF;
  }

  /**
   * Download the stdout and stderr of an executed action.
   *
   * @param context the context for the action.
   * @param result the result of the action.
   * @param outErr the {@link OutErr} that the stdout and stderr will be downloaded to.
   */
  public final List<ListenableFuture<Void>> downloadOutErr(
      RemoteActionExecutionContext context, ActionResult result, OutErr outErr) {
    List<ListenableFuture<Void>> downloads = new ArrayList<>();
    if (!result.getStdoutRaw().isEmpty()) {
      try {
        result.getStdoutRaw().writeTo(outErr.getOutputStream());
        outErr.getOutputStream().flush();
      } catch (IOException e) {
        downloads.add(Futures.immediateFailedFuture(e));
      }
    } else if (result.hasStdoutDigest()) {
      downloads.add(downloadBlob(context, result.getStdoutDigest(), outErr.getOutputStream()));
    }
    if (!result.getStderrRaw().isEmpty()) {
      try {
        result.getStderrRaw().writeTo(outErr.getErrorStream());
        outErr.getErrorStream().flush();
      } catch (IOException e) {
        downloads.add(Futures.immediateFailedFuture(e));
      }
    } else if (result.hasStderrDigest()) {
      downloads.add(downloadBlob(context, result.getStderrDigest(), outErr.getErrorStream()));
    }
    return downloads;
  }

  @Override
  public RemoteCache retain() {
    super.retain();
    return this;
  }

  @Override
  public ReferenceCounted touch(Object hint) {
    return this;
  }

  /** Release resources associated with the cache. The cache may not be used after calling this. */
  @Override
  protected void deallocate() {
    checkState(!closed.get(), "closed");
    checkState(
        casUploadCache.getInProgressTasks().isEmpty(), "There are still in progress uploads.");

    closed.set(true);

    casUploadCache.shutdown();

    cacheProtocol.close();

    closedLatch.countDown();
  }

  public boolean isClosed() {
    return closed.get();
  }

  /** Cancels all active network I/Os and rejects new requests. */
  public void shutdownNow() {
    casUploadCache.shutdownNow();
    try {
      closedLatch.await();
    } catch (InterruptedException e) {
      throw new RuntimeException("Failed to close remote cache", e);
    }
  }

  /**
   * Waits the cache to terminate. Only returns if a) the internal reference count is reached to 0
   * and b) All network I/Os are finished.
   */
  public void awaitTermination() throws InterruptedException {
    try {
      casUploadCache.awaitTermination().blockingAwait();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      throwIfInstanceOf(cause, InterruptedException.class);
      throw e;
    }
  }

  public static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRemoteExecution(RemoteExecution.newBuilder().setCode(detailedCode))
        .build();
  }

  /**
   * An {@link OutputStream} that reports all the write operations with {@link
   * DownloadProgressReporter}.
   */
  private static class ReportingOutputStream extends OutputStream {

    private final OutputStream out;
    private final DownloadProgressReporter reporter;

    ReportingOutputStream(OutputStream out, DownloadProgressReporter reporter) {
      this.out = out;
      this.reporter = reporter;
    }

    @Override
    public void write(byte[] b) throws IOException {
      out.write(b);
      reporter.downloadedBytes(b.length);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      out.write(b, off, len);
      reporter.downloadedBytes(len);
    }

    @Override
    public void write(int b) throws IOException {
      out.write(b);
      reporter.downloadedBytes(1);
    }

    @Override
    public void flush() throws IOException {
      out.flush();
    }

    @Override
    public void close() throws IOException {
      out.close();
    }
  }
}
