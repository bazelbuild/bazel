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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileMetadata;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileNode;
import com.google.devtools.build.lib.remote.RemoteProtocol.Output;
import com.google.devtools.build.lib.remote.RemoteProtocol.Output.ContentCase;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Semaphore;

/**
 * A RemoteActionCache implementation that uses a concurrent map as a distributed storage for files
 * and action output.
 *
 * <p>The thread safety is guaranteed by the underlying map.
 */
@ThreadSafe
public final class ConcurrentMapActionCache implements RemoteActionCache {
  private final ConcurrentMap<String, byte[]> cache;
  private static final int MAX_MEMORY_KBYTES = 512 * 1024;
  private final Semaphore uploadMemoryAvailable = new Semaphore(MAX_MEMORY_KBYTES, true);

  public ConcurrentMapActionCache(ConcurrentMap<String, byte[]> cache) {
    this.cache = cache;
  }

  @Override
  public void uploadTree(TreeNodeRepository repository, Path execRoot, TreeNode root)
      throws IOException, InterruptedException {
    repository.computeMerkleDigests(root);
    for (FileNode fileNode : repository.treeToFileNodes(root)) {
      uploadBlob(fileNode.toByteArray());
    }
    for (TreeNode leaf : repository.leaves(root)) {
      uploadFileContents(leaf.getActionInput(), execRoot, repository.getInputFileCache());
    }
  }

  @Override
  public void downloadTree(ContentDigest rootDigest, Path rootLocation)
      throws IOException, CacheNotFoundException {
    FileNode fileNode = FileNode.parseFrom(downloadBlob(rootDigest));
    if (fileNode.hasFileMetadata()) {
      FileMetadata meta = fileNode.getFileMetadata();
      downloadFileContents(meta.getDigest(), rootLocation, meta.getExecutable());
    }
    for (FileNode.Child child : fileNode.getChildList()) {
      downloadTree(child.getDigest(), rootLocation.getRelative(child.getPath()));
    }
  }

  @Override
  public ContentDigest uploadFileContents(Path file) throws IOException, InterruptedException {
    // This unconditionally reads the whole file into memory first!
    return uploadBlob(ByteString.readFrom(file.getInputStream()).toByteArray());
  }

  @Override
  public ContentDigest uploadFileContents(
      ActionInput input, Path execRoot, ActionInputFileCache inputCache)
      throws IOException, InterruptedException {
    // This unconditionally reads the whole file into memory first!
    return uploadBlob(
        ByteString.readFrom(execRoot.getRelative(input.getExecPathString()).getInputStream())
            .toByteArray(),
        ContentDigests.getDigestFromInputCache(input, inputCache));
  }

  @Override
  public void downloadAllResults(ActionResult result, Path execRoot)
      throws IOException, CacheNotFoundException {
    for (Output output : result.getOutputList()) {
      if (output.getContentCase() == ContentCase.FILE_METADATA) {
        FileMetadata m = output.getFileMetadata();
        downloadFileContents(
            m.getDigest(), execRoot.getRelative(output.getPath()), m.getExecutable());
      } else {
        downloadTree(output.getDigest(), execRoot.getRelative(output.getPath()));
      }
    }
  }

  @Override
  public void uploadAllResults(Path execRoot, Collection<Path> files, ActionResult.Builder result)
      throws IOException, InterruptedException {
    for (Path file : files) {
      if (file.isDirectory()) {
        // TODO(olaola): to implement this for a directory, will need to create or pass a
        // TreeNodeRepository to call uploadTree.
        throw new UnsupportedOperationException("Storing a directory is not yet supported.");
      }
      // First put the file content to cache.
      ContentDigest digest = uploadFileContents(file);
      // Add to protobuf.
      result
          .addOutputBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .getFileMetadataBuilder()
          .setDigest(digest)
          .setExecutable(file.isExecutable());
    }
  }

  @Override
  public void downloadFileContents(ContentDigest digest, Path dest, boolean executable)
      throws IOException, CacheNotFoundException {
    // This unconditionally downloads the whole file into memory first!
    byte[] contents = downloadBlob(digest);
    FileSystemUtils.createDirectoryAndParents(dest.getParentDirectory());
    try (OutputStream stream = dest.getOutputStream()) {
      stream.write(contents);
    }
    dest.setExecutable(executable);
  }

  @Override
  public ImmutableList<ContentDigest> uploadBlobs(Iterable<byte[]> blobs)
      throws InterruptedException {
    ArrayList<ContentDigest> digests = new ArrayList<>();
    for (byte[] blob : blobs) {
      digests.add(uploadBlob(blob));
    }
    return ImmutableList.copyOf(digests);
  }

  private void checkBlobSize(long blobSizeKBytes, String type) {
    Preconditions.checkArgument(
        blobSizeKBytes < MAX_MEMORY_KBYTES,
        type + ": maximum blob size exceeded: %sK > %sK.",
        blobSizeKBytes, MAX_MEMORY_KBYTES);
  }

  @Override
  public ContentDigest uploadBlob(byte[] blob) throws InterruptedException {
    return uploadBlob(blob, ContentDigests.computeDigest(blob));
  }

  private ContentDigest uploadBlob(byte[] blob, ContentDigest digest) throws InterruptedException {
    int blobSizeKBytes = blob.length / 1024;
    checkBlobSize(blobSizeKBytes, "Upload");
    uploadMemoryAvailable.acquire(blobSizeKBytes);
    try {
      cache.put(ContentDigests.toHexString(digest), blob);
    } finally {
      uploadMemoryAvailable.release(blobSizeKBytes);
    }
    return digest;
  }

  @Override
  public byte[] downloadBlob(ContentDigest digest) throws CacheNotFoundException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    // This unconditionally downloads the whole blob into memory!
    checkBlobSize(digest.getSizeBytes() / 1024, "Download");
    byte[] data = cache.get(ContentDigests.toHexString(digest));
    if (data == null) {
      throw new CacheNotFoundException(digest);
    }
    return data;
  }

  @Override
  public ImmutableList<byte[]> downloadBlobs(Iterable<ContentDigest> digests)
      throws CacheNotFoundException {
    ArrayList<byte[]> blobs = new ArrayList<>();
    for (ContentDigest c : digests) {
      blobs.add(downloadBlob(c));
    }
    return ImmutableList.copyOf(blobs);
  }

  public boolean containsKey(ContentDigest digest) {
    return cache.containsKey(ContentDigests.toHexString(digest));
  }

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey) {
    byte[] data = cache.get(ContentDigests.toHexString(actionKey.getDigest()));
    if (data == null) {
      return null;
    }
    try {
      return ActionResult.parseFrom(data);
    } catch (InvalidProtocolBufferException e) {
      return null;
    }
  }

  @Override
  public void setCachedActionResult(ActionKey actionKey, ActionResult result)
      throws InterruptedException {
    cache.put(ContentDigests.toHexString(actionKey.getDigest()), result.toByteArray());
  }
}
