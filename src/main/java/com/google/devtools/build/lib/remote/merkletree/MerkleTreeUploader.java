package com.google.devtools.build.lib.remote.merkletree;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** The basic cache operations needed to upload a {@link MerkleTree} and its associated blobs. */
public interface MerkleTreeUploader {
  /** Uploads an in-memory blob to the remote cache. */
  ListenableFuture<Void> upload(RemoteActionExecutionContext context, Digest digest, byte[] data);

  /** Uploads a local file to the remote cache. */
  ListenableFuture<Void> upload(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      Digest digest,
      Path path);

  /** Uploads a virtual action input to the remote cache. */
  ListenableFuture<Void> upload(
      RemoteActionExecutionContext context, Digest digest, VirtualActionInput virtualActionInput);

  /**
   * Ensures that all inputs as well as metadata protos in the given Merkle tree are present in the
   * remote cache by querying for and uploading missing blobs.
   *
   * @param force if true, all blobs in the tree are uploaded even if they have already been
   *     uploaded before.
   */
  void ensureInputsPresent(
      RemoteActionExecutionContext context,
      MerkleTree.Uploadable merkleTree,
      boolean force,
      RemotePathResolver remotePathResolver)
      throws IOException, InterruptedException;
}
