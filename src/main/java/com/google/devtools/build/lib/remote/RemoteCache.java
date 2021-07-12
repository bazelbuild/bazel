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
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.OutputSymlink;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.exec.SpawnProgressEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteCache.ActionResultMetadata.DirectoryMetadata;
import com.google.devtools.build.lib.remote.RemoteCache.ActionResultMetadata.FileMetadata;
import com.google.devtools.build.lib.remote.RemoteCache.ActionResultMetadata.SymlinkMetadata;
import com.google.devtools.build.lib.remote.common.LazyFileOutputStream;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.ProgressStatusListener;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionFileArtifactValue;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import io.reactivex.rxjava3.core.Completable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;

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

  private final AtomicBoolean closed = new AtomicBoolean(false);
  private final AsyncTaskCache.NoResult<Digest> uploadCache = AsyncTaskCache.NoResult.create();

  protected final RemoteCacheClient cacheProtocol;
  protected final RemoteOptions options;
  protected final DigestUtil digestUtil;

  private Path captureCorruptedOutputsDir;

  public RemoteCache(
      RemoteCacheClient cacheProtocol, RemoteOptions options, DigestUtil digestUtil) {
    this.cacheProtocol = cacheProtocol;
    this.options = options;
    this.digestUtil = digestUtil;
  }

  public void setCaptureCorruptedOutputsDir(Path captureCorruptedOutputsDir) {
    this.captureCorruptedOutputsDir = captureCorruptedOutputsDir;
  }

  public ActionResult downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr)
      throws IOException, InterruptedException {
    return getFromFuture(cacheProtocol.downloadActionResult(context, actionKey, inlineOutErr));
  }

  public final ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests) {
    checkState(!closed.get(), "closed");

    Set<Digest> digestsInProgress = uploadCache.getInProgressTasks();
    Set<Digest> digestsUploaded = uploadCache.getFinishedTasks();
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

    Completable upload = uploadCache.executeIfNot(
            actionKey.getDigest(),
            RxFutures.toCompletable(
                () -> cacheProtocol.uploadActionResult(context, actionKey, actionResult),
                directExecutor()));

    return RxFutures.toListenableFuture(upload);
  }

  /** Upload a local file to the remote cache. */
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context,
      Digest digest,
      Path file
  ) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    Completable upload =
        uploadCache.executeIfNot(
            digest,
            RxFutures.toCompletable(
                () -> cacheProtocol.uploadFile(context, digest, file), directExecutor()));
    return RxFutures.toListenableFuture(upload);
  }

  /** Upload sequence of bytes to the remote cache. */
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      ByteString data
  ) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    Completable upload =
        uploadCache.executeIfNot(
            digest,
            RxFutures.toCompletable(
                () -> cacheProtocol.uploadBlob(context, digest, data), directExecutor()));
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

  private static Path toTmpDownloadPath(Path actualPath) {
    return actualPath.getParentDirectory().getRelative(actualPath.getBaseName() + ".tmp");
  }

  static class DownloadProgressReporter {
    private static final Pattern PATTERN = Pattern.compile("^bazel-out/[^/]+/[^/]+/");
    private final ProgressStatusListener listener;
    private final String id;
    private final String file;
    private final String totalSize;
    private final AtomicLong downloadedBytes = new AtomicLong(0);

    DownloadProgressReporter(ProgressStatusListener listener, String file, long totalSize) {
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

  /**
   * Download the output files and directory trees of a remotely executed action to the local
   * machine, as well stdin / stdout to the given files.
   *
   * <p>In case of failure, this method deletes any output files it might have already created.
   *
   * @param outputFilesLocker ensures that we are the only ones writing to the output files when
   *     using the dynamic spawn strategy.
   * @throws IOException in case of a cache miss or if the remote cache is unavailable.
   * @throws ExecException in case clean up after a failed download failed.
   */
  public void download(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionResult result,
      FileOutErr origOutErr,
      OutputFilesLocker outputFilesLocker,
      ProgressStatusListener progressStatusListener)
      throws ExecException, IOException, InterruptedException {
    ActionResultMetadata metadata = parseActionResultMetadata(context, remotePathResolver, result);

    List<ListenableFuture<FileMetadata>> downloads =
        Stream.concat(
                metadata.files().stream(),
                metadata.directories().stream()
                    .flatMap((entry) -> entry.getValue().files().stream()))
            .map(
                (file) -> {
                  try {
                    ListenableFuture<Void> download =
                        downloadFile(
                            context,
                            remotePathResolver.localPathToOutputPath(file.path()),
                            toTmpDownloadPath(file.path()),
                            file.digest(),
                            new DownloadProgressReporter(
                                progressStatusListener,
                                remotePathResolver.localPathToOutputPath(file.path()),
                                file.digest().getSizeBytes()));
                    return Futures.transform(download, (d) -> file, directExecutor());
                  } catch (IOException e) {
                    return Futures.<FileMetadata>immediateFailedFuture(e);
                  }
                })
            .collect(Collectors.toList());

    // Subsequently we need to wait for *every* download to finish, even if we already know that
    // one failed. That's so that when exiting this method we can be sure that all downloads have
    // finished and don't race with the cleanup routine.

    FileOutErr tmpOutErr = null;
    if (origOutErr != null) {
      tmpOutErr = origOutErr.childOutErr();
    }
    downloads.addAll(downloadOutErr(context, result, tmpOutErr));

    try {
      waitForBulkTransfer(downloads, /* cancelRemainingOnInterrupt=*/ true);
    } catch (Exception e) {
      if (captureCorruptedOutputsDir != null) {
        if (e instanceof BulkTransferException) {
          for (Throwable suppressed : e.getSuppressed()) {
            if (suppressed instanceof OutputDigestMismatchException) {
              // Capture corrupted outputs
              try {
                String outputPath = ((OutputDigestMismatchException) suppressed).getOutputPath();
                Path localPath = ((OutputDigestMismatchException) suppressed).getLocalPath();
                Path dst = captureCorruptedOutputsDir.getRelative(outputPath);
                dst.createDirectoryAndParents();

                // Make sure dst is still under captureCorruptedOutputsDir, otherwise
                // IllegalArgumentException will be thrown.
                dst.relativeTo(captureCorruptedOutputsDir);

                FileSystemUtils.copyFile(localPath, dst);
              } catch (Exception ee) {
                ee.addSuppressed(ee);
              }
            }
          }
        }
      }

      try {
        // Delete any (partially) downloaded output files.
        for (OutputFile file : result.getOutputFilesList()) {
          toTmpDownloadPath(remotePathResolver.outputPathToLocalPath(file.getPath())).delete();
        }
        for (OutputDirectory directory : result.getOutputDirectoriesList()) {
          // Only delete the directories below the output directories because the output
          // directories will not be re-created
          remotePathResolver.outputPathToLocalPath(directory.getPath()).deleteTreesBelow();
        }
        if (tmpOutErr != null) {
          tmpOutErr.clearOut();
          tmpOutErr.clearErr();
        }
      } catch (IOException ioEx) {
        ioEx.addSuppressed(e);

        // If deleting of output files failed, we abort the build with a decent error message as
        // any subsequent local execution failure would likely be incomprehensible.
        ExecException execEx =
            new EnvironmentalExecException(
                ioEx,
                createFailureDetail(
                    "Failed to delete output files after incomplete download",
                    Code.INCOMPLETE_OUTPUT_DOWNLOAD_CLEANUP_FAILURE));
        execEx.addSuppressed(e);
        throw execEx;
      }
      throw e;
    }

    if (tmpOutErr != null) {
      FileOutErr.dump(tmpOutErr, origOutErr);
      tmpOutErr.clearOut();
      tmpOutErr.clearErr();
    }

    // Ensure that we are the only ones writing to the output files when using the dynamic spawn
    // strategy.
    outputFilesLocker.lock();

    moveOutputsToFinalLocation(downloads);

    List<SymlinkMetadata> symlinksInDirectories = new ArrayList<>();
    for (Entry<Path, DirectoryMetadata> entry : metadata.directories()) {
      entry.getKey().createDirectoryAndParents();
      symlinksInDirectories.addAll(entry.getValue().symlinks());
    }

    Iterable<SymlinkMetadata> symlinks =
        Iterables.concat(metadata.symlinks(), symlinksInDirectories);

    // Create the symbolic links after all downloads are finished, because dangling symlinks
    // might not be supported on all platforms
    createSymlinks(symlinks);
  }

  /**
   * Copies moves the downloaded outputs from their download location to their declared location.
   */
  private void moveOutputsToFinalLocation(List<ListenableFuture<FileMetadata>> downloads)
      throws IOException, InterruptedException {
    List<FileMetadata> finishedDownloads = new ArrayList<>(downloads.size());
    for (ListenableFuture<FileMetadata> finishedDownload : downloads) {
      FileMetadata outputFile = getFromFuture(finishedDownload);
      if (outputFile != null) {
        finishedDownloads.add(outputFile);
      }
    }
    /*
     * Sort the list lexicographically based on its temporary download path in order to avoid
     * filename clashes when moving the files:
     *
     * Consider an action that produces two outputs foo and foo.tmp. These outputs would initially
     * be downloaded to foo.tmp and foo.tmp.tmp. When renaming them to foo and foo.tmp we need to
     * ensure that rename(foo.tmp, foo) happens before rename(foo.tmp.tmp, foo.tmp). We ensure this
     * by doing the renames in lexicographical order of the download names.
     */
    Collections.sort(finishedDownloads, Comparator.comparing(f -> toTmpDownloadPath(f.path())));

    // Move the output files from their temporary name to the actual output file name.
    for (FileMetadata outputFile : finishedDownloads) {
      FileSystemUtils.moveFile(toTmpDownloadPath(outputFile.path()), outputFile.path());
      outputFile.path().setExecutable(outputFile.isExecutable());
    }
  }

  private void createSymlinks(Iterable<SymlinkMetadata> symlinks) throws IOException {
    for (SymlinkMetadata symlink : symlinks) {
      if (symlink.target().isAbsolute()) {
        // We do not support absolute symlinks as outputs.
        throw new IOException(
            String.format(
                "Action output %s is a symbolic link to an absolute path %s. "
                    + "Symlinks to absolute paths in action outputs are not supported.",
                symlink.path(), symlink.target()));
      }
      Preconditions.checkNotNull(
              symlink.path().getParentDirectory(),
              "Failed creating directory and parents for %s",
              symlink.path())
          .createDirectoryAndParents();
      symlink.path().createSymbolicLink(symlink.target());
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

  private ListenableFuture<Void> downloadBlob(RemoteActionExecutionContext context, Digest digest, OutputStream out) {
    if (digest.getSizeBytes() == 0) {
      return COMPLETED_SUCCESS;
    }

    return cacheProtocol.downloadBlob(context, digest, out);
  }

  public List<ListenableFuture<FileMetadata>> downloadOutErr(
      RemoteActionExecutionContext context, ActionResult result, OutErr outErr) {
    List<ListenableFuture<FileMetadata>> downloads = new ArrayList<>();
    if (!result.getStdoutRaw().isEmpty()) {
      try {
        result.getStdoutRaw().writeTo(outErr.getOutputStream());
        outErr.getOutputStream().flush();
      } catch (IOException e) {
        downloads.add(Futures.immediateFailedFuture(e));
      }
    } else if (result.hasStdoutDigest()) {
      downloads.add(
          Futures.transform(
              downloadBlob(
                  context, result.getStdoutDigest(), outErr.getOutputStream()),
              (d) -> null,
              directExecutor()));
    }
    if (!result.getStderrRaw().isEmpty()) {
      try {
        result.getStderrRaw().writeTo(outErr.getErrorStream());
        outErr.getErrorStream().flush();
      } catch (IOException e) {
        downloads.add(Futures.immediateFailedFuture(e));
      }
    } else if (result.hasStderrDigest()) {
      downloads.add(
          Futures.transform(
              downloadBlob(
                  context, result.getStderrDigest(), outErr.getErrorStream()),
              (d) -> null,
              directExecutor()));
    }
    return downloads;
  }

  /**
   * Avoids downloading the majority of action outputs but injects their metadata using {@link
   * MetadataInjector} instead.
   *
   * <p>This method only downloads output directory metadata, stdout and stderr as well as the
   * contents of {@code inMemoryOutputPath} if specified.
   *
   * @param context the context this action running with
   * @param result the action result metadata of a successfully executed action (exit code = 0).
   * @param outputs the action's declared output files
   * @param inMemoryOutputPath the path of an output file whose contents should be returned in
   *     memory by this method.
   * @param outErr stdout and stderr of this action
   * @param metadataInjector the action's metadata injector that allows this method to inject
   *     metadata about an action output instead of downloading the output
   * @param outputFilesLocker ensures that we are the only ones writing to the output files when
   *     using the dynamic spawn strategy.
   * @throws IOException in case of failure
   * @throws InterruptedException in case of receiving an interrupt
   */
  @Nullable
  public InMemoryOutput downloadMinimal(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionResult result,
      Collection<? extends ActionInput> outputs,
      @Nullable PathFragment inMemoryOutputPath,
      OutErr outErr,
      MetadataInjector metadataInjector,
      OutputFilesLocker outputFilesLocker)
      throws IOException, InterruptedException {
    checkState(
        result.getExitCode() == 0,
        "injecting remote metadata is only supported for successful actions (exit code 0).");

    ActionResultMetadata metadata;
    try (SilentCloseable c = Profiler.instance().profile("Remote.parseActionResultMetadata")) {
      metadata = parseActionResultMetadata(context, remotePathResolver, result);
    }

    if (!metadata.symlinks().isEmpty()) {
      throw new IOException(
          "Symlinks in action outputs are not yet supported by "
              + "--experimental_remote_download_outputs=minimal");
    }

    // Ensure that when using dynamic spawn strategy that we are the only ones writing to the
    // output files.
    outputFilesLocker.lock();

    ActionInput inMemoryOutput = null;
    Digest inMemoryOutputDigest = null;
    for (ActionInput output : outputs) {
      if (inMemoryOutputPath != null && output.getExecPath().equals(inMemoryOutputPath)) {
        Path localPath = remotePathResolver.outputPathToLocalPath(output);
        FileMetadata m = metadata.file(localPath);
        if (m == null) {
          // A declared output wasn't created. Ignore it here. SkyFrame will fail if not all
          // outputs were created.
          continue;
        }
        inMemoryOutputDigest = m.digest();
        inMemoryOutput = output;
      }
      if (output instanceof Artifact) {
        injectRemoteArtifact(
            context, remotePathResolver, (Artifact) output, metadata, metadataInjector);
      }
    }

    try (SilentCloseable c = Profiler.instance().profile("Remote.download")) {
      ListenableFuture<byte[]> inMemoryOutputDownload = null;
      if (inMemoryOutput != null) {
        inMemoryOutputDownload = downloadBlob(context, inMemoryOutputDigest);
      }
      waitForBulkTransfer(
          downloadOutErr(context, result, outErr), /* cancelRemainingOnInterrupt=*/ true);
      if (inMemoryOutputDownload != null) {
        waitForBulkTransfer(
            ImmutableList.of(inMemoryOutputDownload), /* cancelRemainingOnInterrupt=*/ true);
        byte[] data = getFromFuture(inMemoryOutputDownload);
        return new InMemoryOutput(inMemoryOutput, ByteString.copyFrom(data));
      }
    }
    return null;
  }

  private void injectRemoteArtifact(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      Artifact output,
      ActionResultMetadata metadata,
      MetadataInjector metadataInjector)
      throws IOException {
    Path path = remotePathResolver.outputPathToLocalPath(output);
    if (output.isTreeArtifact()) {
      DirectoryMetadata directory = metadata.directory(path);
      if (directory == null) {
        // A declared output wasn't created. It might have been an optional output and if not
        // SkyFrame will make sure to fail.
        return;
      }
      if (!directory.symlinks().isEmpty()) {
        throw new IOException(
            "Symlinks in action outputs are not yet supported by "
                + "--experimental_remote_download_outputs=minimal");
      }
      SpecialArtifact parent = (SpecialArtifact) output;
      TreeArtifactValue.Builder tree = TreeArtifactValue.newBuilder(parent);
      for (FileMetadata file : directory.files()) {
        TreeFileArtifact child =
            TreeFileArtifact.createTreeOutput(parent, file.path().relativeTo(parent.getPath()));
        RemoteActionFileArtifactValue value =
            new RemoteActionFileArtifactValue(
                DigestUtil.toBinaryDigest(file.digest()),
                file.digest().getSizeBytes(),
                /*locationIndex=*/ 1,
                context.getRequestMetadata().getActionId(),
                file.isExecutable());
        tree.putChild(child, value);
      }
      metadataInjector.injectTree(parent, tree.build());
    } else {
      FileMetadata outputMetadata = metadata.file(path);
      if (outputMetadata == null) {
        // A declared output wasn't created. It might have been an optional output and if not
        // SkyFrame will make sure to fail.
        return;
      }
      metadataInjector.injectFile(
          output,
          new RemoteActionFileArtifactValue(
              DigestUtil.toBinaryDigest(outputMetadata.digest()),
              outputMetadata.digest().getSizeBytes(),
              /*locationIndex=*/ 1,
              context.getRequestMetadata().getActionId(),
              outputMetadata.isExecutable()));
    }
  }

  private DirectoryMetadata parseDirectory(
      Path parent, Directory dir, Map<Digest, Directory> childDirectoriesMap) {
    ImmutableList.Builder<FileMetadata> filesBuilder = ImmutableList.builder();
    for (FileNode file : dir.getFilesList()) {
      filesBuilder.add(
          new FileMetadata(
              parent.getRelative(file.getName()), file.getDigest(), file.getIsExecutable()));
    }

    ImmutableList.Builder<SymlinkMetadata> symlinksBuilder = ImmutableList.builder();
    for (SymlinkNode symlink : dir.getSymlinksList()) {
      symlinksBuilder.add(
          new SymlinkMetadata(
              parent.getRelative(symlink.getName()), PathFragment.create(symlink.getTarget())));
    }

    for (DirectoryNode directoryNode : dir.getDirectoriesList()) {
      Path childPath = parent.getRelative(directoryNode.getName());
      Directory childDir =
          Preconditions.checkNotNull(childDirectoriesMap.get(directoryNode.getDigest()));
      DirectoryMetadata childMetadata = parseDirectory(childPath, childDir, childDirectoriesMap);
      filesBuilder.addAll(childMetadata.files());
      symlinksBuilder.addAll(childMetadata.symlinks());
    }

    return new DirectoryMetadata(filesBuilder.build(), symlinksBuilder.build());
  }

  private ActionResultMetadata parseActionResultMetadata(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionResult actionResult)
      throws IOException, InterruptedException {
    Preconditions.checkNotNull(actionResult, "actionResult");
    Map<Path, ListenableFuture<Tree>> dirMetadataDownloads =
        Maps.newHashMapWithExpectedSize(actionResult.getOutputDirectoriesCount());
    for (OutputDirectory dir : actionResult.getOutputDirectoriesList()) {
      dirMetadataDownloads.put(
          remotePathResolver.outputPathToLocalPath(dir.getPath()),
          Futures.transform(
              downloadBlob(context, dir.getTreeDigest()),
              (treeBytes) -> {
                try {
                  return Tree.parseFrom(treeBytes);
                } catch (InvalidProtocolBufferException e) {
                  throw new RuntimeException(e);
                }
              },
              directExecutor()));
    }

    waitForBulkTransfer(dirMetadataDownloads.values(), /* cancelRemainingOnInterrupt=*/ true);

    ImmutableMap.Builder<Path, DirectoryMetadata> directories = ImmutableMap.builder();
    for (Map.Entry<Path, ListenableFuture<Tree>> metadataDownload :
        dirMetadataDownloads.entrySet()) {
      Path path = metadataDownload.getKey();
      Tree directoryTree = getFromFuture(metadataDownload.getValue());
      Map<Digest, Directory> childrenMap = new HashMap<>();
      for (Directory childDir : directoryTree.getChildrenList()) {
        childrenMap.put(digestUtil.compute(childDir), childDir);
      }

      directories.put(path, parseDirectory(path, directoryTree.getRoot(), childrenMap));
    }

    ImmutableMap.Builder<Path, FileMetadata> files = ImmutableMap.builder();
    for (OutputFile outputFile : actionResult.getOutputFilesList()) {
      Path localPath = remotePathResolver.outputPathToLocalPath(outputFile.getPath());
      files.put(
          localPath,
          new FileMetadata(localPath, outputFile.getDigest(), outputFile.getIsExecutable()));
    }

    ImmutableMap.Builder<Path, SymlinkMetadata> symlinks = ImmutableMap.builder();
    Iterable<OutputSymlink> outputSymlinks =
        Iterables.concat(
            actionResult.getOutputFileSymlinksList(),
            actionResult.getOutputDirectorySymlinksList());
    for (OutputSymlink symlink : outputSymlinks) {
      Path localPath = remotePathResolver.outputPathToLocalPath(symlink.getPath());
      symlinks.put(
          localPath, new SymlinkMetadata(localPath, PathFragment.create(symlink.getTarget())));
    }

    return new ActionResultMetadata(files.build(), symlinks.build(), directories.build());
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
        uploadCache.getInProgressTasks().isEmpty(), "There are still in progress uploads.");

    closed.set(true);

    uploadCache.shutdown();

    cacheProtocol.close();
  }

  public boolean isClosed() {
    return closed.get();
  }

  /** Cancels all active network I/Os and rejects new requests. */
  public void shutdownNow() {
    uploadCache.shutdownNow();
  }

  /**
   * Waits the cache to terminate. Only returns if a) the internal reference count is reached to 0
   * and b) All network I/Os are finished.
   */
  public void awaitTermination() throws InterruptedException {
    try {
      uploadCache.awaitTermination().blockingAwait();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      throwIfInstanceOf(cause, InterruptedException.class);
      throw e;
    }
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
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

  /** In-memory representation of action result metadata. */
  static class ActionResultMetadata {

    static class SymlinkMetadata {
      private final Path path;
      private final PathFragment target;

      private SymlinkMetadata(Path path, PathFragment target) {
        this.path = path;
        this.target = target;
      }

      public Path path() {
        return path;
      }

      public PathFragment target() {
        return target;
      }
    }

    static class FileMetadata {
      private final Path path;
      private final Digest digest;
      private final boolean isExecutable;

      private FileMetadata(Path path, Digest digest, boolean isExecutable) {
        this.path = path;
        this.digest = digest;
        this.isExecutable = isExecutable;
      }

      public Path path() {
        return path;
      }

      public Digest digest() {
        return digest;
      }

      public boolean isExecutable() {
        return isExecutable;
      }
    }

    static class DirectoryMetadata {
      private final ImmutableList<FileMetadata> files;
      private final ImmutableList<SymlinkMetadata> symlinks;

      private DirectoryMetadata(
          ImmutableList<FileMetadata> files, ImmutableList<SymlinkMetadata> symlinks) {
        this.files = files;
        this.symlinks = symlinks;
      }

      public ImmutableList<FileMetadata> files() {
        return files;
      }

      public ImmutableList<SymlinkMetadata> symlinks() {
        return symlinks;
      }
    }

    private final ImmutableMap<Path, FileMetadata> files;
    private final ImmutableMap<Path, SymlinkMetadata> symlinks;
    private final ImmutableMap<Path, DirectoryMetadata> directories;

    private ActionResultMetadata(
        ImmutableMap<Path, FileMetadata> files,
        ImmutableMap<Path, SymlinkMetadata> symlinks,
        ImmutableMap<Path, DirectoryMetadata> directories) {
      this.files = files;
      this.symlinks = symlinks;
      this.directories = directories;
    }

    @Nullable
    public FileMetadata file(Path path) {
      return files.get(path);
    }

    @Nullable
    public DirectoryMetadata directory(Path path) {
      return directories.get(path);
    }

    public Collection<FileMetadata> files() {
      return files.values();
    }

    public ImmutableSet<Entry<Path, DirectoryMetadata>> directories() {
      return directories.entrySet();
    }

    public Collection<SymlinkMetadata> symlinks() {
      return symlinks.values();
    }
  }
}
