// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.DirectoryNode;
import com.google.devtools.remoteexecution.v1test.FileNode;
import com.google.devtools.remoteexecution.v1test.OutputFile;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;

/**
 * A RemoteActionCache implementation that uses a concurrent map as a distributed storage for files
 * and action output.
 *
 * <p>The thread safety is guaranteed by the underlying map.
 *
 * <p>Note that this class is used from src/tools/remote_worker.
 */
@ThreadSafe
public final class SimpleBlobStoreActionCache implements RemoteActionCache {
  private static final int MAX_BLOB_SIZE_FOR_INLINE = 10 * 1024;

  private final SimpleBlobStore blobStore;

  public SimpleBlobStoreActionCache(SimpleBlobStore blobStore) {
    this.blobStore = blobStore;
  }

  @Override
  public void ensureInputsPresent(
      TreeNodeRepository repository, Path execRoot, TreeNode root, Command command)
          throws IOException, InterruptedException {
    repository.computeMerkleDigests(root);
    uploadBlob(command.toByteArray());
    for (Directory directory : repository.treeToDirectories(root)) {
      uploadBlob(directory.toByteArray());
    }
    // TODO(ulfjack): Only upload files that aren't in the CAS yet?
    for (TreeNode leaf : repository.leaves(root)) {
      uploadFileContents(leaf.getActionInput(), execRoot, repository.getInputFileCache());
    }
  }

  public void downloadTree(Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    Directory directory = Directory.parseFrom(downloadBlob(rootDigest));
    for (FileNode file : directory.getFilesList()) {
      downloadFileContents(
          file.getDigest(), rootLocation.getRelative(file.getName()), file.getIsExecutable());
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  private Digest uploadFileContents(Path file) throws IOException, InterruptedException {
    Digest digest = Digests.computeDigest(file);
    try (InputStream in = file.getInputStream()) {
      return uploadStream(digest, in);
    }
  }

  private Digest uploadFileContents(
      ActionInput input, Path execRoot, MetadataProvider inputCache)
          throws IOException, InterruptedException {
    if (input instanceof VirtualActionInput) {
      byte[] blob = ((VirtualActionInput) input).getBytes().toByteArray();
      return uploadBlob(blob, Digests.computeDigest(blob));
    }
    try (InputStream in = execRoot.getRelative(input.getExecPathString()).getInputStream()) {
      return uploadStream(Digests.getDigestFromInputCache(input, inputCache), in);
    }
  }

  @Override
  public void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    try {
      for (OutputFile file : result.getOutputFilesList()) {
        if (!file.getContent().isEmpty()) {
          createFile(
              file.getContent().toByteArray(),
              execRoot.getRelative(file.getPath()),
              file.getIsExecutable());
        } else {
          downloadFileContents(
              file.getDigest(), execRoot.getRelative(file.getPath()), file.getIsExecutable());
        }
      }
      if (!result.getOutputDirectoriesList().isEmpty()) {
        throw new UnsupportedOperationException();
      }
      downloadOutErr(result, outErr);
    } catch (IOException downloadException) {
      try {
        // Delete any (partially) downloaded output files, since any subsequent local execution
        // of this action may expect none of the output files to exist.
        for (OutputFile file : result.getOutputFilesList()) {
          execRoot.getRelative(file.getPath()).delete();
        }
        outErr.getOutputPath().delete();
        outErr.getErrorPath().delete();
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

  private void downloadOutErr(ActionResult result, FileOutErr outErr)
          throws IOException, InterruptedException {
    if (!result.getStdoutRaw().isEmpty()) {
      result.getStdoutRaw().writeTo(outErr.getOutputStream());
      outErr.getOutputStream().flush();
    } else if (result.hasStdoutDigest()) {
      downloadFileContents(result.getStdoutDigest(), outErr.getOutputPath(), /*executable=*/false);
    }
    if (!result.getStderrRaw().isEmpty()) {
      result.getStderrRaw().writeTo(outErr.getErrorStream());
      outErr.getErrorStream().flush();
    } else if (result.hasStderrDigest()) {
      downloadFileContents(result.getStderrDigest(), outErr.getErrorPath(), /*executable=*/false);
    }
  }

  @Override
  public void upload(
      ActionKey actionKey,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr,
      boolean uploadAction)
      throws IOException, InterruptedException {
    ActionResult.Builder result = ActionResult.newBuilder();
    upload(result, execRoot, files);
    if (outErr.getErrorPath().exists()) {
      Digest stderr = uploadFileContents(outErr.getErrorPath());
      result.setStderrDigest(stderr);
    }
    if (outErr.getOutputPath().exists()) {
      Digest stdout = uploadFileContents(outErr.getOutputPath());
      result.setStdoutDigest(stdout);
    }
    if (uploadAction) {
      blobStore.putActionResult(actionKey.getDigest().getHash(), result.build().toByteArray());
    }
  }

  public void upload(ActionResult.Builder result, Path execRoot, Collection<Path> files)
      throws IOException, InterruptedException {
    for (Path file : files) {
      // TODO(ulfjack): Maybe pass in a SpawnResult here, add a list of output files to that, and
      // rely on the local spawn runner to stat the files, instead of statting here.
      if (!file.exists()) {
        continue;
      }
      if (file.isDirectory()) {
        // TODO(olaola): to implement this for a directory, will need to create or pass a
        // TreeNodeRepository to call uploadTree.
        throw new IOException("Storing a directory is not yet supported.");
      }
      // TODO(olaola): inline small file contents here.
      // First put the file content to cache.
      Digest digest = uploadFileContents(file);
      // Add to protobuf.
      result
          .addOutputFilesBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());
    }
  }

  public void uploadOutErr(ActionResult.Builder result, byte[] stdout, byte[] stderr)
      throws IOException, InterruptedException {
    if (stdout.length <= MAX_BLOB_SIZE_FOR_INLINE) {
      result.setStdoutRaw(ByteString.copyFrom(stdout));
    } else if (stdout.length > 0) {
      result.setStdoutDigest(uploadBlob(stdout));
    }
    if (stderr.length <= MAX_BLOB_SIZE_FOR_INLINE) {
      result.setStderrRaw(ByteString.copyFrom(stderr));
    } else if (stderr.length > 0) {
      result.setStderrDigest(uploadBlob(stderr));
    }
  }

  private void downloadFileContents(Digest digest, Path dest, boolean executable)
      throws IOException, InterruptedException {
    FileSystemUtils.createDirectoryAndParents(dest.getParentDirectory());
    try (OutputStream out = dest.getOutputStream()) {
      downloadBlob(digest, out);
    }
    dest.setExecutable(executable);
  }

  private void createFile(byte[] contents, Path dest, boolean executable) throws IOException {
    FileSystemUtils.createDirectoryAndParents(dest.getParentDirectory());
    try (OutputStream stream = dest.getOutputStream()) {
      stream.write(contents);
    }
    dest.setExecutable(executable);
  }

  public Digest uploadBlob(byte[] blob) throws IOException, InterruptedException {
    return uploadBlob(blob, Digests.computeDigest(blob));
  }

  private Digest uploadBlob(byte[] blob, Digest digest) throws IOException, InterruptedException {
    return uploadStream(digest, new ByteArrayInputStream(blob));
  }

  public Digest uploadStream(Digest digest, InputStream in)
      throws IOException, InterruptedException {
    blobStore.put(digest.getHash(), digest.getSizeBytes(), in);
    return digest;
  }

  private void downloadBlob(Digest digest, OutputStream out)
      throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return;
    }
    boolean success = blobStore.get(digest.getHash(), out);
    if (!success) {
      throw new CacheNotFoundException(digest);
    }
  }

  public byte[] downloadBlob(Digest digest)
      throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloadBlob(digest, out);
    return out.toByteArray();
  }

  public boolean containsKey(Digest digest) throws IOException, InterruptedException {
    return blobStore.containsKey(digest.getHash());
  }

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    try {
      byte[] data = downloadActionResult(actionKey.getDigest());
      return ActionResult.parseFrom(data);
    } catch (InvalidProtocolBufferException | CacheNotFoundException e) {
      return null;
    }
  }

  private byte[] downloadActionResult(Digest digest) throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    // This unconditionally downloads the whole blob into memory!
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    boolean success = blobStore.getActionResult(digest.getHash(), out);
    if (!success) {
      throw new CacheNotFoundException(digest);
    }
    return out.toByteArray();
  }

  public void setCachedActionResult(ActionKey actionKey, ActionResult result)
      throws IOException, InterruptedException {
    blobStore.putActionResult(actionKey.getDigest().getHash(), result.toByteArray());
  }

  @Override
  public void close() {
    blobStore.close();
  }
}
