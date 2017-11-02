package com.google.devtools.build.lib.remote;

import com.google.common.base.Charsets;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.DirectoryNode;
import com.google.devtools.remoteexecution.v1test.FileNode;
import com.google.devtools.remoteexecution.v1test.OutputDirectory;
import com.google.devtools.remoteexecution.v1test.OutputFile;
import com.google.devtools.remoteexecution.v1test.Tree;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Base implementation of a {@link RemoteActionCache} shared by {@link SimpleBlobStoreActionCache}
 * and {@link GrpcRemoteCache}.
 */
@ThreadSafety.ThreadSafe
public abstract class RemoteActionCacheBase implements RemoteActionCache {
  protected abstract void downloadBlob(Digest digest, Path dest)
      throws IOException, InterruptedException;

  protected abstract byte[] downloadBlob(Digest digest) throws IOException, InterruptedException;

  /**
   * Download all results of a remotely executed action locally. TODO(olaola): will need to amend to
   * include the {@link com.google.devtools.build.lib.remote.TreeNodeRepository} for updating.
   */
  @Override
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
        Digest receivedTreeDigest = Digests.computeDigest(b);
        if (!receivedTreeDigest.equals(treeDigest)) {
          throw new IOException(
              "Digest does not match " + receivedTreeDigest + " != " + treeDigest);
        }
        Tree tree = Tree.parseFrom(b);
        Map<Digest, Directory> childrenMap = new HashMap<>();
        for (Directory child : tree.getChildrenList()) {
          childrenMap.put(Digests.computeDigest(child), child);
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
        throw new EnvironmentalExecException("Failed to delete output files after incomplete "
            + "download. Cannot continue with local execution.", e, true);
      }
      throw downloadException;
    }
  }

  /**
   * Download a directory recursively.  The directory is represented by a {@link Directory} protobuf message,
   * and the descendant directories are in {@code childrenMap}, accessible through their digest.
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
            "could not find subdirectory " + child.getName() +
                " of directory " + path + " for download: digest " +
                childDigest + "not found");
      }
      downloadDirectory(childPath, childDir, childrenMap);

      // Prevent reuse.
      childrenMap.remove(childDigest);
    }
  }

  /**
   * Download a file (that is not a directory).  If the {@code content} is not given, the content
   * is fetched from the digest.
   */
  protected void downloadFile(Path path, Digest digest, boolean isExecutable, @Nullable ByteString content)
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
        Digest receivedDigest = Digests.computeDigest(path);
        if (!receivedDigest.equals(digest)) {
          throw new IOException(
              "Digest does not match " + receivedDigest + " != " + digest);
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
    private ActionResult.Builder result;
    private Path execRoot;
    private Map<Digest, Path> digestToFile;
    private Map<Digest, Chunker> digestToChunkers;

    /**
     * Create an UploadManifest from an ActionResult builder and an exec root.
     * The ActionResult builder is populated through a call to {@link #addFile(Digest, Path)}.
     */
    public UploadManifest(ActionResult.Builder result, Path execRoot) {
      this.result = result;
      this.execRoot = execRoot;

      this.digestToFile = new HashMap<>();
      this.digestToChunkers = new HashMap<>();
    }

    /**
     * Add a collection of files (and directories) to the UploadManifest.  Adding a directory has the
     * effect of 1) uploading a {@link Tree} protobuf message from which the whole structure of the
     * directory, including the descendants, can be reconstructed and 2) uploading all the non-directory
     * descendant files.
     */
    public void addFiles(Collection<Path> files)
        throws IOException, InterruptedException {
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
          Digest digest = Digests.computeDigest(file);
          addFile(digest, file);
        }
      }
    }

    /**
     * Map of digests to file paths to upload.
     */
    public Map<Digest, Path> getDigestToFile() {
      return digestToFile;
    }

    /**
     * Map of digests to chunkers to upload.  When the file is a regular, non-directory file
     * it is transmitted through {@link #getDigestToFile()}.  When it is a directory, it is
     * transmitted as a {@link Tree} protobuf message through {@link #getDigestToChunkers()}.
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
      Digest digest = Digests.computeDigest(blob);
      Chunker chunker = new Chunker(blob, blob.length);

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

      List<Dirent> sortedDirent = new ArrayList<>(path.readdir(TreeNodeRepository.symlinkPolicy));
      sortedDirent.sort(Comparator.comparing(Dirent::getName));

      for (Dirent dirent: sortedDirent) {
        String name = dirent.getName();
        Path child = path.getRelative(name);
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          Directory dir = computeDirectory(child, tree);
          b.addDirectoriesBuilder()
              .setName(name)
              .setDigest(Digests.computeDigest(dir));
          tree.addChildren(dir);
        } else {
          Digest digest = Digests.computeDigest(child);
          b.addFilesBuilder()
              .setName(name)
              .setDigest(digest)
              .setIsExecutable(child.isExecutable());
          digestToFile.put(digest, child);
        }
      }

      return b.build();
    }
  }
}
