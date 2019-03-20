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

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.OutputSymlink;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RemoteUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
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
import javax.annotation.Nullable;

/** A cache for storing artifacts (input and output) as well as the output of running an action. */
@ThreadSafety.ThreadSafe
public abstract class AbstractRemoteActionCache implements AutoCloseable {

  private static final ListenableFuture<Void> COMPLETED_SUCCESS = SettableFuture.create();
  private static final ListenableFuture<byte[]> EMPTY_BYTES = SettableFuture.create();

  static {
    ((SettableFuture<Void>) COMPLETED_SUCCESS).set(null);
    ((SettableFuture<byte[]>) EMPTY_BYTES).set(new byte[0]);
  }

  protected final RemoteOptions options;
  protected final DigestUtil digestUtil;

  public AbstractRemoteActionCache(RemoteOptions options, DigestUtil digestUtil) {
    this.options = options;
    this.digestUtil = digestUtil;
  }

  /**
   * Attempts to look up the given action in the remote cache and return its result, if present.
   * Returns {@code null} if there is no such entry. Note that a successful result from this method
   * does not guarantee the availability of the corresponding output files in the remote cache.
   *
   * @throws IOException if the remote cache is unavailable.
   */
  abstract @Nullable ActionResult getCachedActionResult(DigestUtil.ActionKey actionKey)
      throws IOException, InterruptedException;

  /**
   * Upload the result of a locally executed action to the remote cache.
   *
   * @throws IOException if there was an error uploading to the remote cache
   * @throws ExecException if uploading any of the action outputs is not supported
   */
  abstract void upload(
      DigestUtil.ActionKey actionKey,
      Action action,
      Command command,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr)
      throws ExecException, IOException, InterruptedException;

  /**
   * Downloads a blob with a content hash {@code digest} to {@code out}.
   *
   * @return a future that completes after the download completes (succeeds / fails).
   */
  protected abstract ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out);

  /**
   * Downloads a blob with content hash {@code digest} and stores its content in memory.
   *
   * @return a future that completes after the download completes (succeeds / fails). If successful,
   *     the content is stored in the future's {@code byte[]}.
   */
  public ListenableFuture<byte[]> downloadBlob(Digest digest) {
    if (digest.getSizeBytes() == 0) {
      return EMPTY_BYTES;
    }
    ByteArrayOutputStream bOut = new ByteArrayOutputStream((int) digest.getSizeBytes());
    SettableFuture<byte[]> outerF = SettableFuture.create();
    Futures.addCallback(
        downloadBlob(digest, bOut),
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void aVoid) {
            outerF.set(bOut.toByteArray());
          }

          @Override
          public void onFailure(Throwable t) {
            outerF.setException(t);
          }
        },
        MoreExecutors.directExecutor());
    return outerF;
  }

  /**
   * Download the output files and directory trees of a remotely executed action to the local
   * machine, as well stdin / stdout to the given files.
   *
   * <p>In case of failure, this method deletes any output files it might have already created.
   *
   * @throws IOException in case of a cache miss or if the remote cache is unavailable.
   * @throws ExecException in case clean up after a failed download failed.
   */
  public void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    List<FuturePathBooleanTuple> fileDownloads =
        Collections.synchronizedList(
            new ArrayList<>(result.getOutputFilesCount() + result.getOutputDirectoriesCount()));
    for (OutputFile file : result.getOutputFilesList()) {
      Path path = execRoot.getRelative(file.getPath());
      ListenableFuture<Void> download = downloadFile(path, file.getDigest());
      fileDownloads.add(new FuturePathBooleanTuple(download, path, file.getIsExecutable()));
    }

    List<ListenableFuture<Void>> dirDownloads = new ArrayList<>(result.getOutputDirectoriesCount());
    for (OutputDirectory dir : result.getOutputDirectoriesList()) {
      SettableFuture<Void> dirDownload = SettableFuture.create();
      ListenableFuture<byte[]> protoDownload = downloadBlob(dir.getTreeDigest());
      Futures.addCallback(
          protoDownload,
          new FutureCallback<byte[]>() {
            @Override
            public void onSuccess(byte[] b) {
              try {
                Tree tree = Tree.parseFrom(b);
                Map<Digest, Directory> childrenMap = new HashMap<>();
                for (Directory child : tree.getChildrenList()) {
                  childrenMap.put(digestUtil.compute(child), child);
                }
                Path path = execRoot.getRelative(dir.getPath());
                fileDownloads.addAll(downloadDirectory(path, tree.getRoot(), childrenMap));
                dirDownload.set(null);
              } catch (IOException e) {
                dirDownload.setException(e);
              }
            }

            @Override
            public void onFailure(Throwable t) {
              dirDownload.setException(t);
            }
          },
          MoreExecutors.directExecutor());
      dirDownloads.add(dirDownload);
    }

    // Subsequently we need to wait for *every* download to finish, even if we already know that
    // one failed. That's so that when exiting this method we can be sure that all downloads have
    // finished and don't race with the cleanup routine.
    // TODO(buchgr): Look into cancellation.

    IOException downloadException = null;
    try {
      fileDownloads.addAll(downloadOutErr(result, outErr));
    } catch (IOException e) {
      downloadException = e;
    }
    for (ListenableFuture<Void> dirDownload : dirDownloads) {
      // Block on all directory download futures, so that we can be sure that we have discovered
      // all file downloads and can subsequently safely iterate over the list of file downloads.
      try {
        getFromFuture(dirDownload);
      } catch (IOException e) {
        downloadException = downloadException == null ? e : downloadException;
      }
    }

    for (FuturePathBooleanTuple download : fileDownloads) {
      try {
        getFromFuture(download.getFuture());
        if (download.getPath() != null) {
          download.getPath().setExecutable(download.isExecutable());
        }
      } catch (IOException e) {
        downloadException = downloadException == null ? e : downloadException;
      }
    }

    if (downloadException != null) {
      try {
        // Delete any (partially) downloaded output files, since any subsequent local execution
        // of this action may expect none of the output files to exist.
        for (OutputFile file : result.getOutputFilesList()) {
          execRoot.getRelative(file.getPath()).delete();
        }
        for (OutputDirectory directory : result.getOutputDirectoriesList()) {
          // Only delete the directories below the output directories because the output
          // directories will not be re-created
          FileSystemUtils.deleteTreesBelow(execRoot.getRelative(directory.getPath()));
        }
        if (outErr != null) {
          outErr.getOutputPath().delete();
          outErr.getErrorPath().delete();
        }
      } catch (IOException e) {
        // If deleting of output files failed, we abort the build with a decent error message as
        // any subsequent local execution failure would likely be incomprehensible.

        // We don't propagate the downloadException, as this is a recoverable error and the cause
        // of the build failure is really that we couldn't delete output files.
        throw new EnvironmentalExecException(
            "Failed to delete output files after incomplete "
                + "download. Cannot continue with local execution.",
            e,
            true);
      }
      throw downloadException;
    }

    // We create the symbolic links after all regular downloads are finished, because dangling
    // links will not work on Windows.
    createSymbolicLinks(
        execRoot,
        Iterables.concat(
            result.getOutputFileSymlinksList(), result.getOutputDirectorySymlinksList()));
  }

  // Creates a local symbolic link. Only relative symlinks are supported.
  private void createSymbolicLink(Path path, String target) throws IOException {
    PathFragment targetPath = PathFragment.create(target);
    if (targetPath.isAbsolute()) {
      // Error, we do not support absolute symlinks as outputs.
      throw new IOException(
          String.format(
              "Action output %s is a symbolic link to an absolute path %s. "
                  + "Symlinks to absolute paths in action outputs are not supported.",
              path, target));
    }
    path.createSymbolicLink(targetPath);
  }

  // Creates symbolic links locally as created remotely by the action. Only relative symbolic
  // links are supported, because absolute symlinks break action hermeticity.
  private void createSymbolicLinks(Path execRoot, Iterable<OutputSymlink> symlinks)
      throws IOException {
    for (OutputSymlink symlink : symlinks) {
      Path path = execRoot.getRelative(symlink.getPath());
      Preconditions.checkNotNull(
              path.getParentDirectory(), "Failed creating directory and parents for %s", path)
          .createDirectoryAndParents();
      createSymbolicLink(path, symlink.getTarget());
    }
  }

  @VisibleForTesting
  protected <T> T getFromFuture(ListenableFuture<T> f) throws IOException, InterruptedException {
    return RemoteUtils.getFromFuture(f);
  }

  /** Tuple of {@code ListenableFuture, Path, boolean}. */
  private static class FuturePathBooleanTuple {
    private final ListenableFuture<?> future;
    private final Path path;
    private final boolean isExecutable;

    public FuturePathBooleanTuple(ListenableFuture<?> future, Path path, boolean isExecutable) {
      this.future = future;
      this.path = path;
      this.isExecutable = isExecutable;
    }

    public ListenableFuture<?> getFuture() {
      return future;
    }

    public Path getPath() {
      return path;
    }

    public boolean isExecutable() {
      return isExecutable;
    }
  }

  /**
   * Download a directory recursively. The directory is represented by a {@link Directory} protobuf
   * message, and the descendant directories are in {@code childrenMap}, accessible through their
   * digest.
   */
  private List<FuturePathBooleanTuple> downloadDirectory(
      Path path, Directory dir, Map<Digest, Directory> childrenMap) throws IOException {
    // Ensure that the directory is created here even though the directory might be empty
    path.createDirectoryAndParents();

    for (SymlinkNode symlink : dir.getSymlinksList()) {
      createSymbolicLink(path.getRelative(symlink.getName()), symlink.getTarget());
    }

    List<FuturePathBooleanTuple> downloads = new ArrayList<>(dir.getFilesCount());
    for (FileNode child : dir.getFilesList()) {
      Path childPath = path.getRelative(child.getName());
      downloads.add(
          new FuturePathBooleanTuple(
              downloadFile(childPath, child.getDigest()), childPath, child.getIsExecutable()));
    }

    for (DirectoryNode child : dir.getDirectoriesList()) {
      Path childPath = path.getRelative(child.getName());
      Digest childDigest = child.getDigest();
      Directory childDir = childrenMap.get(childDigest);
      if (childDir == null) {
        throw new IOException(
            "could not find subdirectory "
                + child.getName()
                + " of directory "
                + path
                + " for download: digest "
                + childDigest
                + "not found");
      }
      downloads.addAll(downloadDirectory(childPath, childDir, childrenMap));
    }

    return downloads;
  }

  /** Download a file (that is not a directory). The content is fetched from the digest. */
  public ListenableFuture<Void> downloadFile(Path path, Digest digest) throws IOException {
    Preconditions.checkNotNull(path.getParentDirectory()).createDirectoryAndParents();
    if (digest.getSizeBytes() == 0) {
      // Handle empty file locally.
      FileSystemUtils.writeContent(path, new byte[0]);
      return COMPLETED_SUCCESS;
    }

    OutputStream out = new LazyFileOutputStream(path);
    SettableFuture<Void> outerF = SettableFuture.create();
    ListenableFuture<Void> f = downloadBlob(digest, out);
    Futures.addCallback(
        f,
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void result) {
            try {
              out.close();
              outerF.set(null);
            } catch (IOException e) {
              outerF.setException(e);
            }
          }

          @Override
          public void onFailure(Throwable t) {
            try {
              out.close();
            } catch (IOException e) {
              // Intentionally left empty. The download already failed, so we can ignore
              // the error on close().
            } finally {
              outerF.setException(t);
            }
          }
        },
        MoreExecutors.directExecutor());
    return outerF;
  }

  private List<FuturePathBooleanTuple> downloadOutErr(ActionResult result, FileOutErr outErr)
      throws IOException {
    List<FuturePathBooleanTuple> downloads = new ArrayList<>();
    if (!result.getStdoutRaw().isEmpty()) {
      result.getStdoutRaw().writeTo(outErr.getOutputStream());
      outErr.getOutputStream().flush();
    } else if (result.hasStdoutDigest()) {
      downloads.add(
          new FuturePathBooleanTuple(
              downloadBlob(result.getStdoutDigest(), outErr.getOutputStream()), null, false));
    }
    if (!result.getStderrRaw().isEmpty()) {
      result.getStderrRaw().writeTo(outErr.getErrorStream());
      outErr.getErrorStream().flush();
    } else if (result.hasStderrDigest()) {
      downloads.add(
          new FuturePathBooleanTuple(
              downloadBlob(result.getStderrDigest(), outErr.getErrorStream()), null, false));
    }
    return downloads;
  }

  /** UploadManifest adds output metadata to a {@link ActionResult}. */
  static class UploadManifest {
    private final DigestUtil digestUtil;
    private final ActionResult.Builder result;
    private final Path execRoot;
    private final boolean allowSymlinks;
    private final boolean uploadSymlinks;
    private final Map<Digest, Path> digestToFile;
    private final Map<Digest, Chunker> digestToChunkers;

    /**
     * Create an UploadManifest from an ActionResult builder and an exec root. The ActionResult
     * builder is populated through a call to {@link #addFile(Digest, Path)}.
     */
    public UploadManifest(
        DigestUtil digestUtil,
        ActionResult.Builder result,
        Path execRoot,
        boolean uploadSymlinks,
        boolean allowSymlinks) {
      this.digestUtil = digestUtil;
      this.result = result;
      this.execRoot = execRoot;
      this.uploadSymlinks = uploadSymlinks;
      this.allowSymlinks = allowSymlinks;

      this.digestToFile = new HashMap<>();
      this.digestToChunkers = new HashMap<>();
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
    public void addAction(DigestUtil.ActionKey actionKey, Action action, Command command)
        throws IOException {
      byte[] actionBlob = action.toByteArray();
      digestToChunkers.put(
          actionKey.getDigest(),
          Chunker.builder(digestUtil)
              .setInput(actionKey.getDigest(), actionBlob)
              .setChunkSize(actionBlob.length)
              .build());
      byte[] commandBlob = command.toByteArray();
      digestToChunkers.put(
          action.getCommandDigest(),
          Chunker.builder(digestUtil)
              .setInput(action.getCommandDigest(), commandBlob)
              .setChunkSize(commandBlob.length)
              .build());
    }

    /** Map of digests to file paths to upload. */
    public Map<Digest, Path> getDigestToFile() {
      return digestToFile;
    }

    /**
     * Map of digests to chunkers to upload. When the file is a regular, non-directory file it is
     * transmitted through {@link #getDigestToFile()}. When it is a directory, it is transmitted as
     * a {@link Tree} protobuf message through {@link #getDigestToChunkers()}.
     */
    public Map<Digest, Chunker> getDigestToChunkers() {
      return digestToChunkers;
    }

    private void addFileSymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputFileSymlinksBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setTarget(target.toString());
    }

    private void addDirectorySymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputDirectorySymlinksBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setTarget(target.toString());
    }

    private void addFile(Digest digest, Path file) throws IOException {
      result
          .addOutputFilesBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());

      digestToFile.put(digest, file);
    }

    private void addDirectory(Path dir) throws ExecException, IOException {
      Tree.Builder tree = Tree.newBuilder();
      Directory root = computeDirectory(dir, tree);
      tree.setRoot(root);

      byte[] blob = tree.build().toByteArray();
      Digest digest = digestUtil.compute(blob);
      Chunker chunker =
          Chunker.builder(digestUtil).setInput(digest, blob).setChunkSize(blob.length).build();

      if (result != null) {
        result
            .addOutputDirectoriesBuilder()
            .setPath(dir.relativeTo(execRoot).getPathString())
            .setTreeDigest(digest);
      }

      digestToChunkers.put(chunker.digest(), chunker);
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

    private void illegalOutput(Path what) throws ExecException, IOException {
      String kind = what.isSymbolicLink() ? "symbolic link" : "special file";
      throw new UserExecException(
          String.format(
              "Output %s is a %s. Only regular files and directories may be "
                  + "uploaded to a remote cache. "
                  + "Change the file type or use --remote_allow_symlink_upload.",
              what.relativeTo(execRoot), kind));
    }
  }

  protected void verifyContents(String expectedHash, String actualHash) throws IOException {
    if (!expectedHash.equals(actualHash)) {
      String msg =
          String.format(
              "An output download failed, because the expected hash"
                  + "'%s' did not match the received hash '%s'.",
              expectedHash, actualHash);
      throw new IOException(msg);
    }
  }

  /** Release resources associated with the cache. The cache may not be used after calling this. */
  @Override
  public abstract void close();

  /**
   * Creates an {@link OutputStream} that isn't actually opened until the first data is written.
   * This is useful to only have as many open file descriptors as necessary at a time to avoid
   * running into system limits.
   */
  private static class LazyFileOutputStream extends OutputStream {

    private final Path path;
    private OutputStream out;

    public LazyFileOutputStream(Path path) {
      this.path = path;
    }

    @Override
    public void write(byte[] b) throws IOException {
      ensureOpen();
      out.write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      ensureOpen();
      out.write(b, off, len);
    }

    @Override
    public void write(int b) throws IOException {
      ensureOpen();
      out.write(b);
    }

    @Override
    public void flush() throws IOException {
      ensureOpen();
      out.flush();
    }

    @Override
    public void close() throws IOException {
      ensureOpen();
      out.close();
    }

    private void ensureOpen() throws IOException {
      if (out == null) {
        out = path.getOutputStream();
      }
    }
  }
}
