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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.devtools.build.lib.remote.util.DigestUtil.DIGEST_COMPARATOR;
import static java.util.Comparator.comparing;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.primitives.UnsignedBytes;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

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
   *
   * <p>See {@link
   * com.google.devtools.build.lib.remote.RemoteExecutionServiceTest#buildRemoteAction_goldenTest}
   * for a test that verifies the memory footprint of this class. Since there can be thousands of
   * inflight remote executions that may have to retain their blobs until all inputs have been
   * uploaded, it's crucial to keep the memory footprint of this class as low as possible.
   */
  final class Uploadable implements MerkleTree {
    private static final Comparator<FileArtifactValue> FILE_ARTIFACT_VALUE_COMPARATOR =
        comparing(FileArtifactValue::getDigest, UnsignedBytes.lexicographicalComparator())
            .thenComparing(FileArtifactValue::getSize);
    static final Comparator<Object> DIGEST_AND_METADATA_COMPARATOR =
        (o1, o2) ->
            switch (o1) {
              case Digest digest1 ->
                  DIGEST_COMPARATOR.compare(
                      digest1,
                      switch (o2) {
                        case Digest digest2 -> digest2;
                        case FileArtifactValue metadata2 -> adaptToDigest(metadata2);
                        default -> throw new IllegalStateException("Unexpected blob type: " + o2);
                      });
              case FileArtifactValue metadata1 ->
                  switch (o2) {
                    case FileArtifactValue metadata2 ->
                        FILE_ARTIFACT_VALUE_COMPARATOR.compare(metadata1, metadata2);
                    case Digest digest2 ->
                        DIGEST_COMPARATOR.compare(adaptToDigest(metadata1), digest2);
                    default -> throw new IllegalStateException("Unexpected blob type: " + o2);
                  };
              default -> throw new IllegalStateException("Unexpected blob type: " + o1);
            };
    private final RootOnly.BlobsUploaded root;
    private final ImmutableSortedMap<
            /* Digest | FileArtifactValue */ Object, /* byte[] | ActionInput */ Object>
        blobs;

    Uploadable(
        RootOnly.BlobsUploaded root,
        SortedMap</* Digest | FileArtifactValue */ Object, /* byte[] | ActionInput */ Object>
            blobs) {
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
      return Collections2.transform(blobs.keySet(), MerkleTree.Uploadable::adaptToDigest);
    }

    @VisibleForTesting
    public Map<Digest, Object> blobs() {
      return blobs.entrySet().stream()
          .collect(toImmutableMap(e -> adaptToDigest(e.getKey()), Map.Entry::getValue));
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

    public int uniquelyRetainedBytes() {
      // See RemoteExecutionServiceTest#buildRemoteAction_goldenTest for a test that verifies the
      // real memory footprint of this class and serves as the base (and verification) for the
      // following estimation of unique retained bytes.
      int size =
          40 // ImmutableSortedMap object
              + 2 * 16 // RegularImmutableList objects
              + 32 // MerkleTree.RootOnly.BlobsUploaded object
              + 24 // RegularImmutableSortedSet object
              + 16 // MerkleTree.Uploadable object
              + 2 * objectArrayShallowSize(blobs.size()); // arrays for keys and values
      for (Object key : blobs.keySet()) {
        size +=
            switch (key) {
              case Digest digest ->
                  32 // Digest object
                      + 16 // ByteString$LiteralByteString wrapping the hash bytes
                      + byteArraySize(digest.getHashBytes().size());
              // FileArtifactValue is retained by Skyframe anyway.
              default -> 0;
            };
      }
      for (Object value : blobs.values()) {
        size +=
            switch (value) {
              case byte[] data -> byteArraySize(data.length);
              case MerkleTreeComputer.EmptyInputDirectory ignored ->
                  // outputDir is retained as a Spawn input.
                  16;
              case MerkleTreeComputer.ChildActionInput childActionInput ->
                  // parent is retained as a Spawn input.
                  16 + stringSize(childActionInput.relativePath);
              // Don't account for PathActionInput, which is only used in tests or remote repository
              // execution. All other ActionInputs are retained anyway (permanently by Skyframe in
              // the case of Artifacts or, in the case of VirtualActionInput, by the Spawn).
              default -> 0;
            };
      }
      return size;
    }

    private static int stringSize(String str) {
      return 24 + byteArraySize(str.length());
    }

    private static int objectArrayShallowSize(int length) {
      return arraySize(length, 4);
    }

    private static int byteArraySize(int length) {
      return arraySize(length, 1);
    }

    private static int arraySize(int length, int sizePerElement) {
      // 8 byte header with -XX:+UseCompactObjectHeaders + 4 byte length field
      int unpaddedSize = 12 + length * sizePerElement;
      // Pad to multiples of 8 bytes.
      return (unpaddedSize + 7) & ~7;
    }

    private static Digest adaptToDigest(Object key) {
      return switch (key) {
        case Digest digest -> digest;
        case FileArtifactValue metadata ->
            DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
        default -> throw new IllegalStateException("Unexpected blob type: " + key);
      };
    }

    static final class StatsCollector {
      private final LongAdder count = new LongAdder();
      private final AtomicInteger minRetainedBytes = new AtomicInteger(Integer.MAX_VALUE);
      private final AtomicInteger maxRetainedBytes = new AtomicInteger();
      private final LongAdder totalRetainedBytes = new LongAdder();
      private final AtomicLong currentConcurrentRetainedBytes = new AtomicLong();
      private final AtomicLong maxConcurrentRetainedBytes = new AtomicLong();

      public void track(MerkleTree.Uploadable tree) {
        int retainedBytes = tree.uniquelyRetainedBytes();
        count.increment();
        minRetainedBytes.getAndUpdate(current -> Math.min(current, retainedBytes));
        maxRetainedBytes.getAndUpdate(current -> Math.max(current, retainedBytes));
        totalRetainedBytes.add(retainedBytes);
        long newConcurrent = currentConcurrentRetainedBytes.addAndGet(retainedBytes);
        maxConcurrentRetainedBytes.getAndUpdate(current -> Math.max(current, newConcurrent));
      }

      public void untrack(MerkleTree.Uploadable tree) {
        int retainedBytes = tree.uniquelyRetainedBytes();
        currentConcurrentRetainedBytes.addAndGet(-retainedBytes);
      }

      public String getStats() {
        long count = this.count.sum();
        if (count == 0) {
          return "No Merkle trees tracked.";
        }
        long totalRetainedBytes = this.totalRetainedBytes.sum();
        return String.format(
            "Tracked %,d Merkle trees. Retained bytes: min=%,d, max=%,d, avg=%,.2f, max concurrent=%,d, current concurrent=%,d",
            count,
            minRetainedBytes.get(),
            maxRetainedBytes.get(),
            (double) totalRetainedBytes / count,
            maxConcurrentRetainedBytes.get(),
            currentConcurrentRetainedBytes.get());
      }
    }
  }
}
