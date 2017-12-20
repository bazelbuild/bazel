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

import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.DirectoryNode;
import com.google.devtools.remoteexecution.v1test.FileNode;
import com.google.devtools.remoteexecution.v1test.OutputDirectory;
import com.google.devtools.remoteexecution.v1test.OutputFile;
import com.google.devtools.remoteexecution.v1test.Tree;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A cache for storing artifacts (input and output) as well as the output of running an action. */
@ThreadSafety.ThreadSafe
public abstract class AbstractRemoteActionCache implements AutoCloseable {
  protected final DigestUtil digestUtil;

  public AbstractRemoteActionCache(DigestUtil digestUtil) {
    this.digestUtil = digestUtil;
  }

  /**
   * Ensures that the tree structure of the inputs, the input files themselves, and the command are
   * available in the remote cache, such that the tree can be reassembled and executed on another
   * machine given the root digest.
   *
   * <p>The cache may check whether files or parts of the tree structure are already present, and do
   * not need to be uploaded again.
   *
   * <p>Note that this method is only required for remote execution, not for caching itself.
   * However, remote execution uses a cache to store input files, and that may be a separate
   * end-point from the executor itself, so the functionality lives here. A pure remote caching
   * implementation that does not support remote execution may choose not to implement this
   * function, and throw {@link UnsupportedOperationException} instead. If so, it should be clearly
   * documented that it cannot be used for remote execution.
   */
  public abstract void ensureInputsPresent(
      TreeNodeRepository repository, Path execRoot, TreeNode root, Command command)
      throws IOException, InterruptedException;

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
   * Upload the result of a locally executed action to the cache by uploading any necessary files,
   * stdin / stdout, as well as adding an entry for the given action key to the cache if
   * uploadAction is true.
   *
   * @throws IOException if the remote cache is unavailable.
   */
  abstract void upload(
      DigestUtil.ActionKey actionKey,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr,
      boolean uploadAction)
      throws IOException, InterruptedException;

  /**
   * Download a remote blob to a local destination.
   *
   * @param digest The digest of the remote blob.
   * @param dest The path to the local file.
   * @throws IOException if download failed.
   */
  protected abstract void downloadBlob(Digest digest, Path dest)
      throws IOException, InterruptedException;

  /**
   * Download a remote blob and store it in memory.
   *
   * @param digest The digest of the remote blob.
   * @return The remote blob.
   * @throws IOException if download failed.
   */
  protected abstract byte[] downloadBlob(Digest digest) throws IOException, InterruptedException;

  /**
   * Download the output files and directory trees of a remotely executed action to the local
   * machine, as well stdin / stdout to the given files.
   *
   * <p>In case of failure, this method deletes any output files it might have already created.
   *
   * @throws IOException in case of a cache miss or if the remote cache is unavailable.
   * @throws ExecException in case clean up after a failed download failed.
   */
  // TODO(olaola): will need to amend to include the TreeNodeRepository for updating.
  public void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    try {
      for (OutputFile file : result.getOutputFilesList()) {
        Path path = execRoot.getRelative(file.getPath());
        downloadFile(path, file.getDigest(), file.getIsExecutable(), file.getContent());
      }
      for (OutputDirectory dir : result.getOutputDirectoriesList()) {
        Digest treeDigest = dir.getTreeDigest();
        byte[] b = downloadBlob(treeDigest);
        Digest receivedTreeDigest = digestUtil.compute(b);
        if (!receivedTreeDigest.equals(treeDigest)) {
          throw new IOException(
              "Digest does not match " + receivedTreeDigest + " != " + treeDigest);
        }
        Tree tree = Tree.parseFrom(b);
        Map<Digest, Directory> childrenMap = new HashMap<>();
        for (Directory child : tree.getChildrenList()) {
          childrenMap.put(digestUtil.compute(child), child);
        }
        Path path = execRoot.getRelative(dir.getPath());
        downloadDirectory(path, tree.getRoot(), childrenMap);
      }
      // TODO(ulfjack): use same code as above also for stdout / stderr if applicable.
      downloadOutErr(result, outErr);
    } catch (IOException downloadException) {
      try {
        // Delete any (partially) downloaded output files, since any subsequent local execution
        // of this action may expect none of the output files to exist.
        for (OutputFile file : result.getOutputFilesList()) {
          execRoot.getRelative(file.getPath()).delete();
        }
        for (OutputDirectory directory : result.getOutputDirectoriesList()) {
          execRoot.getRelative(directory.getPath()).delete();
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
  }

  /**
   * Download a directory recursively. The directory is represented by a {@link Directory} protobuf
   * message, and the descendant directories are in {@code childrenMap}, accessible through their
   * digest.
   */
  private void downloadDirectory(Path path, Directory dir, Map<Digest, Directory> childrenMap)
      throws IOException, InterruptedException {
    // Ensure that the directory is created here even though the directory might be empty
    FileSystemUtils.createDirectoryAndParents(path);

    for (FileNode child : dir.getFilesList()) {
      Path childPath = path.getRelative(child.getName());
      downloadFile(childPath, child.getDigest(), child.getIsExecutable(), null);
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
      downloadDirectory(childPath, childDir, childrenMap);
    }
  }

  /**
   * Download a file (that is not a directory). If the {@code content} is not given, the content is
   * fetched from the digest.
   */
  protected void downloadFile(
      Path path, Digest digest, boolean isExecutable, @Nullable ByteString content)
      throws IOException, InterruptedException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    if (digest.getSizeBytes() == 0) {
      // Handle empty file locally.
      FileSystemUtils.writeContent(path, new byte[0]);
    } else {
      if (content != null && !content.isEmpty()) {
        try (OutputStream stream = path.getOutputStream()) {
          content.writeTo(stream);
        }
      } else {
        downloadBlob(digest, path);
        Digest receivedDigest = digestUtil.compute(path);
        if (!receivedDigest.equals(digest)) {
          throw new IOException("Digest does not match " + receivedDigest + " != " + digest);
        }
      }
    }
    path.setExecutable(isExecutable);
  }

  private void downloadOutErr(ActionResult result, FileOutErr outErr)
      throws IOException, InterruptedException {
    if (!result.getStdoutRaw().isEmpty()) {
      result.getStdoutRaw().writeTo(outErr.getOutputStream());
      outErr.getOutputStream().flush();
    } else if (result.hasStdoutDigest()) {
      byte[] stdoutBytes = downloadBlob(result.getStdoutDigest());
      outErr.getOutputStream().write(stdoutBytes);
      outErr.getOutputStream().flush();
    }
    if (!result.getStderrRaw().isEmpty()) {
      result.getStderrRaw().writeTo(outErr.getErrorStream());
      outErr.getErrorStream().flush();
    } else if (result.hasStderrDigest()) {
      byte[] stderrBytes = downloadBlob(result.getStderrDigest());
      outErr.getErrorStream().write(stderrBytes);
      outErr.getErrorStream().flush();
    }
  }

  /**
   * The UploadManifest is used to mutualize upload between the RemoteActionCache implementations.
   */
  public class UploadManifest {
    private final ActionResult.Builder result;
    private final Path execRoot;
    private final Map<Digest, Path> digestToFile;
    private final Map<Digest, Chunker> digestToChunkers;

    /**
     * Create an UploadManifest from an ActionResult builder and an exec root. The ActionResult
     * builder is populated through a call to {@link #addFile(Digest, Path)}.
     */
    public UploadManifest(ActionResult.Builder result, Path execRoot) {
      this.result = result;
      this.execRoot = execRoot;

      this.digestToFile = new HashMap<>();
      this.digestToChunkers = new HashMap<>();
    }

    /**
     * Add a collection of files (and directories) to the UploadManifest. Adding a directory has the
     * effect of 1) uploading a {@link Tree} protobuf message from which the whole structure of the
     * directory, including the descendants, can be reconstructed and 2) uploading all the
     * non-directory descendant files.
     */
    public void addFiles(Collection<Path> files) throws IOException, InterruptedException {
      for (Path file : files) {
        // TODO(ulfjack): Maybe pass in a SpawnResult here, add a list of output files to that, and
        // rely on the local spawn runner to stat the files, instead of statting here.
        if (!file.exists()) {
          // We ignore requested results that have not been generated by the action.
          continue;
        }
        if (file.isDirectory()) {
          addDirectory(file);
        } else {
          Digest digest = digestUtil.compute(file);
          addFile(digest, file);
        }
      }
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

    private void addFile(Digest digest, Path file) throws IOException {
      result
          .addOutputFilesBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());

      digestToFile.put(digest, file);
    }

    private void addDirectory(Path dir) throws IOException {
      Tree.Builder tree = Tree.newBuilder();
      Directory root = computeDirectory(dir, tree);
      tree.setRoot(root);

      byte[] blob = tree.build().toByteArray();
      Digest digest = digestUtil.compute(blob);
      Chunker chunker = new Chunker(blob, blob.length, digestUtil);

      if (result != null) {
        result
            .addOutputDirectoriesBuilder()
            .setPath(dir.relativeTo(execRoot).getPathString())
            .setTreeDigest(digest);
      }

      digestToChunkers.put(chunker.digest(), chunker);
    }

    private Directory computeDirectory(Path path, Tree.Builder tree) throws IOException {
      Directory.Builder b = Directory.newBuilder();

      List<Dirent> sortedDirent = new ArrayList<>(path.readdir(TreeNodeRepository.SYMLINK_POLICY));
      sortedDirent.sort(Comparator.comparing(Dirent::getName));

      for (Dirent dirent : sortedDirent) {
        String name = dirent.getName();
        Path child = path.getRelative(name);
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          Directory dir = computeDirectory(child, tree);
          b.addDirectoriesBuilder().setName(name).setDigest(digestUtil.compute(dir));
          tree.addChildren(dir);
        } else {
          Digest digest = digestUtil.compute(child);
          b.addFilesBuilder().setName(name).setDigest(digest).setIsExecutable(child.isExecutable());
          digestToFile.put(digest, child);
        }
      }

      return b.build();
    }
  }

  /** Release resources associated with the cache. The cache may not be used after calling this. */
  @Override
  public abstract void close();
}
