// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.merkletree;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;

/**
 * A representation of the inputs to a remotely executed action represented as a Merkle tree.
 *
 * <p>Every tree has a digest, which is the digest of the tree's root directory. The subtrees and
 * the blobs they contain may have been discarded or never computed in the first place, for example,
 * because they have already been uploaded to the remote cache or because the tree is being built
 * only to check for a remote cache hit.
 */
public sealed interface MerkleTree {
  /** The digest of the tree's root directory. */
  Digest digest();

  /** The total number of regular files and symlinks in this tree, including all subtrees. */
  long inputFiles();

  /**
   * The total number of content bytes in this tree, including all subtrees. This includes both file
   * contents and the protos describing directories.
   */
  long inputBytes();

  /** Returns the root of this tree, which may be the current instance. */
  RootOnly root();

  /**
   * A {@link MerkleTree} that doesn't retain any blobs, either because they have already been
   * uploaded or because only the root digest is needed (e.g., for a remote cache check).
   */
  sealed interface RootOnly extends MerkleTree {
    @Override
    default RootOnly root() {
      return this;
    }

    /**
     * A {@link MerkleTree} that retains no blobs since all of them have recently been uploaded to
     * the remote cache.
     */
    record BlobsUploaded(Digest digest, long inputFiles, long inputBytes) implements RootOnly {}

    /**
     * A {@link MerkleTree} that retains no blobs since they were discarded during the computation
     * (e.g., because they aren't needed for a remote cache check).
     */
    record BlobsDiscarded(Digest digest, long inputFiles, long inputBytes) implements RootOnly {}
  }

  /**
   * A {@link MerkleTree} that retains all blobs that still need to be uploaded.
   *
   * <p>The empty blob doesn't have to be uploaded and is thus never included in the blobs map.
   */
  final class Uploadable implements MerkleTree {
    private final RootOnly.BlobsUploaded root;
    private final ImmutableSortedMap<Digest, /* byte[] | ActionInput */ Object> blobs;

    Uploadable(RootOnly.BlobsUploaded root, SortedMap<Digest, Object> blobs) {
      this.root = root;
      // A sorted map requires less memory than a regular hash map as it only stores two flat sorted
      // arrays. Access performance is not critical since it's only used to find missing blobs,
      // which always require network access.
      this.blobs = ImmutableSortedMap.copyOfSorted(blobs);
    }

    @Override
    public Digest digest() {
      return root().digest();
    }

    @Override
    public long inputFiles() {
      return root().inputFiles();
    }

    @Override
    public long inputBytes() {
      return root().inputBytes();
    }

    public Collection<Digest> allDigests() {
      return blobs.keySet();
    }

    @VisibleForTesting
    public Map<Digest, Object> blobs() {
      return blobs;
    }

    @Override
    public RootOnly root() {
      return root;
    }

    /**
     * Returns a future that tracks the upload of the blob with the given digest, or {@link
     * Optional#empty()} if there is no blob with the given digest.
     */
    public Optional<ListenableFuture<Void>> upload(
        MerkleTreeUploader uploader,
        RemoteActionExecutionContext context,
        RemotePathResolver remotePathResolver,
        Digest digest) {
      return switch (blobs.get(digest)) {
        case byte[] data -> Optional.of(uploader.uploadBlob(context, digest, data));
        case VirtualActionInput virtualActionInput ->
            Optional.of(uploader.uploadVirtualActionInput(context, digest, virtualActionInput));
        case ActionInput actionInput -> {
          var spawnExecutionContext = context.getSpawnExecutionContext();
          var pathResolver =
              // This can only be null when uploading a tree created by
              // MerkleTreeComputer#buildForFiles, which only happens for remote repo execution and
              // tests. Only the latter actually reach this code path since remote repo execution
              // doesn't upload any inputs.
              spawnExecutionContext != null
                  ? spawnExecutionContext.getPathResolver()
                  : MerkleTreeComputer.PATH_ACTION_INPUT_RESOLVER;
          yield Optional.of(
              uploader.uploadFile(
                  context, remotePathResolver, digest, pathResolver.toPath(actionInput)));
        }
        case null -> Optional.empty();
        default -> throw new IllegalStateException("Unexpected blob type: " + blobs.get(digest));
      };
    }
  }
}
