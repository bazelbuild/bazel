package com.google.devtools.build.lib.remote.merkletree;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Optional;

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

  /** A {@link MerkleTree} that retains all blobs that still need to be uploaded. */
  final class Uploadable implements MerkleTree {
    private final RootOnly.BlobsUploaded root;
    private final ImmutableMap<Digest, /* byte[] | Path | VirtualActionInput */ Object> blobs;

    Uploadable(RootOnly.BlobsUploaded root, ImmutableMap<Digest, Object> blobs) {
      this.root = root;
      this.blobs = blobs;
    }

    public Digest digest() {
      return root().digest();
    }

    public long inputFiles() {
      return root().inputFiles();
    }

    public long inputBytes() {
      return root().inputBytes();
    }

    public ImmutableSet<Digest> allDigests() {
      return blobs.keySet();
    }

    @VisibleForTesting
    public ImmutableMap<Digest, Object> blobs() {
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
        case byte[] data -> Optional.of(uploader.upload(context, digest, data));
        case Path path -> Optional.of(uploader.upload(context, remotePathResolver, digest, path));
        case VirtualActionInput virtualActionInput ->
            Optional.of(uploader.upload(context, digest, virtualActionInput));
        case null -> Optional.empty();
        default -> throw new IllegalStateException("Unexpected blob type: " + blobs.get(digest));
      };
    }
  }
}
