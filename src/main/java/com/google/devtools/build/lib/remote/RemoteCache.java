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

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.common.ProgressStatusListener.NO_ACTION;
import static com.google.devtools.build.lib.remote.util.Utils.bytesCountToDisplayString;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.exec.SpawnProgressEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.LazyFileOutputStream;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.ProgressStatusListener;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.CachedActionResult;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** A cache for storing artifacts (input and output) as well as the output of running an action. */
@ThreadSafety.ThreadSafe
public class RemoteCache implements AutoCloseable {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** See {@link SpawnExecutionContext#lockOutputFiles()}. */
  @FunctionalInterface
  interface OutputFilesLocker {
    void lock() throws InterruptedException;
  }

  private static final ListenableFuture<Void> COMPLETED_SUCCESS = immediateFuture(null);
  private static final ListenableFuture<byte[]> EMPTY_BYTES = immediateFuture(new byte[0]);

  protected final RemoteCacheClient cacheProtocol;
  protected final RemoteOptions options;
  protected final DigestUtil digestUtil;

  public RemoteCache(
      RemoteCacheClient cacheProtocol, RemoteOptions options, DigestUtil digestUtil) {
    this.cacheProtocol = cacheProtocol;
    this.options = options;
    this.digestUtil = digestUtil;
  }

  public CachedActionResult downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr)
      throws IOException, InterruptedException {
    return getFromFuture(cacheProtocol.downloadActionResult(context, actionKey, inlineOutErr));
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

    return cacheProtocol.uploadFile(context, digest, file);
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

    return cacheProtocol.uploadBlob(context, digest, data);
  }

  /**
   * Upload the result of a locally executed action to the remote cache.
   *
   * @throws IOException if there was an error uploading to the remote cache
   * @throws ExecException if uploading any of the action outputs is not supported
   */
  public ActionResult upload(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionKey actionKey,
      Action action,
      Command command,
      Collection<Path> outputs,
      FileOutErr outErr,
      int exitCode)
      throws ExecException, IOException, InterruptedException {
    ActionResult.Builder resultBuilder = ActionResult.newBuilder();
    uploadOutputs(
        context, remotePathResolver, actionKey, action, command, outputs, outErr, resultBuilder);
    resultBuilder.setExitCode(exitCode);
    ActionResult result = resultBuilder.build();
    if (exitCode == 0) {
      cacheProtocol.uploadActionResult(context, actionKey, result);
    }
    return result;
  }

  public ActionResult upload(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionKey actionKey,
      Action action,
      Command command,
      Collection<Path> outputs,
      FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    return upload(
        context,
        remotePathResolver,
        actionKey,
        action,
        command,
        outputs,
        outErr,
        /* exitCode= */ 0);
  }

  private void uploadOutputs(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      ActionKey actionKey,
      Action action,
      Command command,
      Collection<Path> files,
      FileOutErr outErr,
      ActionResult.Builder result)
      throws ExecException, IOException, InterruptedException {
    UploadManifest manifest =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            options.incompatibleRemoteSymlinks,
            options.allowSymlinkUpload);
    manifest.addFiles(files);
    manifest.setStdoutStderr(outErr);
    manifest.addAction(actionKey, action, command);

    Map<Digest, Path> digestToFile = manifest.getDigestToFile();
    Map<Digest, ByteString> digestToBlobs = manifest.getDigestToBlobs();
    Collection<Digest> digests = new ArrayList<>();
    digests.addAll(digestToFile.keySet());
    digests.addAll(digestToBlobs.keySet());

    ImmutableSet<Digest> digestsToUpload =
        getFromFuture(cacheProtocol.findMissingDigests(context, digests));
    ImmutableList.Builder<ListenableFuture<Void>> uploads = ImmutableList.builder();
    for (Digest digest : digestsToUpload) {
      Path file = digestToFile.get(digest);
      if (file != null) {
        uploads.add(uploadFile(context, digest, file));
      } else {
        ByteString blob = digestToBlobs.get(digest);
        if (blob == null) {
          String message = "FindMissingBlobs call returned an unknown digest: " + digest;
          throw new IOException(message);
        }
        uploads.add(uploadBlob(context, digest, blob));
      }
    }

    waitForBulkTransfer(uploads.build(), /* cancelRemainingOnInterrupt=*/ false);

    if (manifest.getStderrDigest() != null) {
      result.setStderrDigest(manifest.getStderrDigest());
    }
    if (manifest.getStdoutDigest() != null) {
      result.setStdoutDigest(manifest.getStdoutDigest());
    }
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

  /** UploadManifest adds output metadata to a {@link ActionResult}. */
  static class UploadManifest {
    private final DigestUtil digestUtil;
    private final RemotePathResolver remotePathResolver;
    private final ActionResult.Builder result;
    private final boolean allowSymlinks;
    private final boolean uploadSymlinks;
    private final Map<Digest, Path> digestToFile = new HashMap<>();
    private final Map<Digest, ByteString> digestToBlobs = new HashMap<>();
    private Digest stderrDigest;
    private Digest stdoutDigest;

    /**
     * Create an UploadManifest from an ActionResult builder and an exec root. The ActionResult
     * builder is populated through a call to {@link #addFile(Digest, Path)}.
     */
    public UploadManifest(
        DigestUtil digestUtil,
        RemotePathResolver remotePathResolver,
        ActionResult.Builder result,
        boolean uploadSymlinks,
        boolean allowSymlinks) {
      this.digestUtil = digestUtil;
      this.remotePathResolver = remotePathResolver;
      this.result = result;
      this.uploadSymlinks = uploadSymlinks;
      this.allowSymlinks = allowSymlinks;
    }

    public void setStdoutStderr(FileOutErr outErr) throws IOException {
      if (outErr.getErrorPath().exists()) {
        stderrDigest = digestUtil.compute(outErr.getErrorPath());
        digestToFile.put(stderrDigest, outErr.getErrorPath());
      }
      if (outErr.getOutputPath().exists()) {
        stdoutDigest = digestUtil.compute(outErr.getOutputPath());
        digestToFile.put(stdoutDigest, outErr.getOutputPath());
      }
    }

    /**
     * Add a collection of files or directories to the UploadManifest. Adding a directory has the
     * effect of 1) uploading a {@link Tree} protobuf message from which the whole structure of the
     * directory, including the descendants, can be reconstructed and 2) uploading all the
     * non-directory descendant files.
     */
    public void addFiles(Collection<Path> files) throws ExecException, IOException {
      for (Path file : files) {
        // TODO(ulfjack): Maybe pass in a SpawnResult here, add a list of output files to that, and
        // rely on the local spawn runner to stat the files, instead of statting here.
        FileStatus stat = file.statIfFound(Symlinks.NOFOLLOW);
        // TODO(#6547): handle the case where the parent directory of the output file is an
        // output symlink.
        if (stat == null) {
          // We ignore requested results that have not been generated by the action.
          continue;
        }
        if (stat.isDirectory()) {
          addDirectory(file);
        } else if (stat.isFile() && !stat.isSpecialFile()) {
          Digest digest = digestUtil.compute(file, stat.getSize());
          addFile(digest, file);
        } else if (stat.isSymbolicLink() && allowSymlinks) {
          PathFragment target = file.readSymbolicLink();
          // Need to resolve the symbolic link to know what to add, file or directory.
          FileStatus statFollow = file.statIfFound(Symlinks.FOLLOW);
          if (statFollow == null) {
            throw new IOException(
                String.format("Action output %s is a dangling symbolic link to %s ", file, target));
          }
          if (statFollow.isSpecialFile()) {
            illegalOutput(file);
          }
          Preconditions.checkState(
              statFollow.isFile() || statFollow.isDirectory(), "Unknown stat type for %s", file);
          if (uploadSymlinks && !target.isAbsolute()) {
            if (statFollow.isFile()) {
              addFileSymbolicLink(file, target);
            } else {
              addDirectorySymbolicLink(file, target);
            }
          } else {
            if (statFollow.isFile()) {
              addFile(digestUtil.compute(file), file);
            } else {
              addDirectory(file);
            }
          }
        } else {
          illegalOutput(file);
        }
      }
    }

    /**
     * Adds an action and command protos to upload. They need to be uploaded as part of the action
     * result.
     */
    public void addAction(RemoteCacheClient.ActionKey actionKey, Action action, Command command) {
      digestToBlobs.put(actionKey.getDigest(), action.toByteString());
      digestToBlobs.put(action.getCommandDigest(), command.toByteString());
    }

    /** Map of digests to file paths to upload. */
    public Map<Digest, Path> getDigestToFile() {
      return digestToFile;
    }

    /**
     * Map of digests to chunkers to upload. When the file is a regular, non-directory file it is
     * transmitted through {@link #getDigestToFile()}. When it is a directory, it is transmitted as
     * a {@link Tree} protobuf message through {@link #getDigestToBlobs()}.
     */
    public Map<Digest, ByteString> getDigestToBlobs() {
      return digestToBlobs;
    }

    @Nullable
    public Digest getStdoutDigest() {
      return stdoutDigest;
    }

    @Nullable
    public Digest getStderrDigest() {
      return stderrDigest;
    }

    private void addFileSymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputFileSymlinksBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setTarget(target.toString());
    }

    private void addDirectorySymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputDirectorySymlinksBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setTarget(target.toString());
    }

    private void addFile(Digest digest, Path file) throws IOException {
      result
          .addOutputFilesBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());

      digestToFile.put(digest, file);
    }

    private void addDirectory(Path dir) throws ExecException, IOException {
      Tree.Builder tree = Tree.newBuilder();
      Directory root = computeDirectory(dir, tree);
      tree.setRoot(root);

      ByteString data = tree.build().toByteString();
      Digest digest = digestUtil.compute(data.toByteArray());

      if (result != null) {
        result
            .addOutputDirectoriesBuilder()
            .setPath(remotePathResolver.localPathToOutputPath(dir))
            .setTreeDigest(digest);
      }

      digestToBlobs.put(digest, data);
    }

    private Directory computeDirectory(Path path, Tree.Builder tree)
        throws ExecException, IOException {
      Directory.Builder b = Directory.newBuilder();

      List<Dirent> sortedDirent = new ArrayList<>(path.readdir(Symlinks.NOFOLLOW));
      sortedDirent.sort(Comparator.comparing(Dirent::getName));

      for (Dirent dirent : sortedDirent) {
        String name = dirent.getName();
        Path child = path.getRelative(name);
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          Directory dir = computeDirectory(child, tree);
          b.addDirectoriesBuilder().setName(name).setDigest(digestUtil.compute(dir));
          tree.addChildren(dir);
        } else if (dirent.getType() == Dirent.Type.SYMLINK && allowSymlinks) {
          PathFragment target = child.readSymbolicLink();
          if (uploadSymlinks && !target.isAbsolute()) {
            // Whether it is dangling or not, we're passing it on.
            b.addSymlinksBuilder().setName(name).setTarget(target.toString());
            continue;
          }
          // Need to resolve the symbolic link now to know whether to upload a file or a directory.
          FileStatus statFollow = child.statIfFound(Symlinks.FOLLOW);
          if (statFollow == null) {
            throw new IOException(
                String.format(
                    "Action output %s is a dangling symbolic link to %s ", child, target));
          }
          if (statFollow.isFile() && !statFollow.isSpecialFile()) {
            Digest digest = digestUtil.compute(child);
            b.addFilesBuilder()
                .setName(name)
                .setDigest(digest)
                .setIsExecutable(child.isExecutable());
            digestToFile.put(digest, child);
          } else if (statFollow.isDirectory()) {
            Directory dir = computeDirectory(child, tree);
            b.addDirectoriesBuilder().setName(name).setDigest(digestUtil.compute(dir));
            tree.addChildren(dir);
          } else {
            illegalOutput(child);
          }
        } else if (dirent.getType() == Dirent.Type.FILE) {
          Digest digest = digestUtil.compute(child);
          b.addFilesBuilder().setName(name).setDigest(digest).setIsExecutable(child.isExecutable());
          digestToFile.put(digest, child);
        } else {
          illegalOutput(child);
        }
      }

      return b.build();
    }

    private void illegalOutput(Path what) throws ExecException {
      String kind = what.isSymbolicLink() ? "symbolic link" : "special file";
      String message =
          String.format(
              "Output %s is a %s. Only regular files and directories may be "
                  + "uploaded to a remote cache. "
                  + "Change the file type or use --remote_allow_symlink_upload.",
              remotePathResolver.localPathToOutputPath(what), kind);
      throw new UserExecException(createFailureDetail(message, Code.ILLEGAL_OUTPUT));
    }
  }

  /** Release resources associated with the cache. The cache may not be used after calling this. */
  @Override
  public void close() {
    cacheProtocol.close();
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
