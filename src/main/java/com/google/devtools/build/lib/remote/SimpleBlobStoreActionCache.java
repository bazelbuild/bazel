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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.DirectoryNode;
import com.google.devtools.remoteexecution.v1test.FileNode;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;

/**
 * A RemoteActionCache implementation that uses a concurrent map as a distributed storage for files
 * and action output.
 *
 * <p>The thread safety is guaranteed by the underlying map.
 *
 * <p>Note that this class is used from src/tools/remote.
 */
@ThreadSafe
public final class SimpleBlobStoreActionCache extends AbstractRemoteActionCache {
  private static final int MAX_BLOB_SIZE_FOR_INLINE = 10 * 1024;

  private final SimpleBlobStore blobStore;

  public SimpleBlobStoreActionCache(
      RemoteOptions options, SimpleBlobStore blobStore, DigestUtil digestUtil) {
    super(options, digestUtil);
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
    FileSystemUtils.createDirectoryAndParents(rootLocation);
    Directory directory = Directory.parseFrom(downloadBlob(rootDigest));
    for (FileNode file : directory.getFilesList()) {
      downloadFile(
          rootLocation.getRelative(file.getName()), file.getDigest(), file.getIsExecutable(), null);
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  private Digest uploadFileContents(Path file) throws IOException, InterruptedException {
    Digest digest = digestUtil.compute(file);
    try (InputStream in = file.getInputStream()) {
      return uploadStream(digest, in);
    }
  }

  private Digest uploadFileContents(
      ActionInput input, Path execRoot, MetadataProvider inputCache)
          throws IOException, InterruptedException {
    if (input instanceof VirtualActionInput) {
      byte[] blob = ((VirtualActionInput) input).getBytes().toByteArray();
      return uploadBlob(blob, digestUtil.compute(blob));
    }
    try (InputStream in = execRoot.getRelative(input.getExecPathString()).getInputStream()) {
      return uploadStream(DigestUtil.getFromInputCache(input, inputCache), in);
    }
  }

  @Override
  public void upload(
      ActionKey actionKey,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr,
      boolean uploadAction)
      throws ExecException, IOException, InterruptedException {
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
      throws ExecException, IOException, InterruptedException {
    UploadManifest manifest =
        new UploadManifest(digestUtil, result, execRoot, options.allowSymlinkUpload);
    manifest.addFiles(files);

    for (Map.Entry<Digest, Path> entry : manifest.getDigestToFile().entrySet()) {
      try (InputStream in = entry.getValue().getInputStream()) {
        uploadStream(entry.getKey(), in);
      }
    }

    for (Map.Entry<Digest, Chunker> entry : manifest.getDigestToChunkers().entrySet()) {
      uploadBlob(entry.getValue().next().getData().toByteArray(), entry.getKey());
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

  public Digest uploadBlob(byte[] blob) throws IOException, InterruptedException {
    return uploadBlob(blob, digestUtil.compute(blob));
  }

  private Digest uploadBlob(byte[] blob, Digest digest) throws IOException, InterruptedException {
    try (InputStream in = new ByteArrayInputStream(blob)) {
      return uploadStream(digest, in);
    }
  }

  public Digest uploadStream(Digest digest, InputStream in)
      throws IOException, InterruptedException {
    blobStore.put(digest.getHash(), digest.getSizeBytes(), in);
    return digest;
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
      throw new CacheNotFoundException(digest, digestUtil);
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

  @Override
  protected void downloadBlob(Digest digest, Path dest) throws IOException, InterruptedException {
    try (OutputStream out = dest.getOutputStream()) {
      boolean success = blobStore.get(digest.getHash(), out);
      if (!success) {
        throw new CacheNotFoundException(digest, digestUtil);
      }
    }
  }

  @Override
  public byte[] downloadBlob(Digest digest) throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      boolean success = blobStore.get(digest.getHash(), out);
      if (!success) {
        throw new CacheNotFoundException(digest, digestUtil);
      }
      return out.toByteArray();
    }
  }
}
