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

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Comparator.comparing;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.primitives.UnsignedBytes;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.util.Collection;
import java.util.Comparator;
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
    private static final Comparator<Digest> DIGEST_COMPARATOR =
        comparing(Digest::getHash).thenComparing(Digest::getSizeBytes);
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
                        case FileArtifactValue metadata2 ->
                            DigestUtil.buildDigest(metadata2.getDigest(), metadata2.getSize());
                        default -> throw new IllegalStateException("Unexpected blob type: " + o2);
                      });
              case FileArtifactValue metadata1 ->
                  switch (o2) {
                    case FileArtifactValue metadata2 ->
                        FILE_ARTIFACT_VALUE_COMPARATOR.compare(metadata1, metadata2);
                    case Digest digest2 ->
                        DIGEST_COMPARATOR.compare(
                            DigestUtil.buildDigest(metadata1.getDigest(), metadata1.getSize()),
                            digest2);
                    default -> throw new IllegalStateException("Unexpected blob type: " + o2);
                  };
              default -> throw new IllegalStateException("Unexpected blob type: " + o1);
            };

    private final RootOnly.BlobsUploaded root;
    private final ImmutableSortedMap<Object, /* byte[] | ActionInput */ Object> blobs;
    private final Digest emptyDigest;

    Uploadable(
        RootOnly.BlobsUploaded root,
        SortedMap</* Digest | FileArtifactValue */ Object, /* byte[] | ActionInput */ Object> blobs,
        Digest emptyDigest) {
      this.root = root;
      // A sorted map requires less memory than a regular hash map as it only stores two flat sorted
      // arrays.
      this.blobs = ImmutableSortedMap.copyOfSorted(blobs);
      checkArgument(
          emptyDigest.getSizeBytes() == 0, "Empty digest must have size 0: %s", emptyDigest);
      this.emptyDigest = emptyDigest;
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

    public int retainedBytes() {
      // Example output of JOL's GraphLayout.parseInstance(...).toFootprint() for a
      // MerkleTree.Uploadable:
      //     COUNT       AVG       SUM   DESCRIPTION
      //        18       180      3240   [B
      //         2       112       224   [Ljava.lang.Object;
      //         9        40       360   build.bazel.remote.execution.v2.Digest
      //         1        40        40   com.google.common.collect.ImmutableSortedMap
      //         2        16        32   com.google.common.collect.RegularImmutableList
      //         1        24        24   com.google.common.collect.RegularImmutableSortedSet
      //         1        32        32
      // com.google.devtools.build.lib.remote.merkletree.MerkleTree$RootOnly$BlobsUploaded
      //         1        16        16
      // com.google.devtools.build.lib.remote.merkletree.MerkleTree$Uploadable
      //         9        24       216   java.lang.String
      //        44                4184   (total)
      int size =
          24 // MerkleTree.Uploadable object
              + 32 // MerkleTree.RootOnly.BlobsUploaded object
              + 40 // ImmutableSortedMap object
              + 24 // RegularImmutableSortedSet object
              + 2 * 16 // RegularImmutableList objects
              + 2 * arraySize(blobs.size(), 4); // Object[] arrays in the lists
      for (Object key : blobs.keySet()) {
        size +=
            switch (key) {
              case Digest digest ->
                  40 // Digest object
                      + 24 // String object for hash
                      + arraySize(digest.getHash().length(), 1); // byte[] for hash
              // FileArtifactValue is retained by Skyframe anyway.
              default -> 0;
            };
      }
      for (Object value : blobs.values()) {
        size +=
            switch (value) {
              case byte[] data -> arraySize(data.length, 1); // byte[] object
              case MerkleTreeComputer.EmptyInputDirectory ignored -> 16;
              case MerkleTreeComputer.ChildActionInput childActionInput ->
                  16 // ChildActionInput object
                      + 24 // String object for relative path
                      + arraySize(
                          childActionInput.relativePath.length(), 1); // byte[] for relative path
              case Object[] directory -> {
                // Compact directory representation from CompactDirectoryBuilder.
                int dirSize = arraySize(directory.length, 4); // Object[] array
                for (Object item : directory) {
                  dirSize +=
                      switch (item) {
                        case String str ->
                            24 // String object
                                + arraySize(str.length(), 1); // byte[] for string content
                        // Other types (Digest, FileArtifactValue, MerkleTree, Artifact,
                        // NodeProperties) are either counted elsewhere or retained globally.
                        case null, default -> 0;
                      };
                }
                yield dirSize;
              }
              // Don't account for PathActionInput, which is only used in tests or remote repository
              // execution. All other ActionInputs are retained anyway (permanently or, in the case
              // of VirtualActionInput, by the Spawn).
              default -> 0;
            };
      }
      return size;
    }

    private int arraySize(int length, int sizePerElement) {
      // 8 byte header with -XX:+UseCompactObjectHeaders + 4 byte length field
      int unpaddedSize = 12 + length * sizePerElement;
      // Pad to multiples of 8 bytes.
      return (unpaddedSize + 7) & ~7;
    }

    public Collection<Digest> allDigests() {
      return Collections2.transform(blobs.keySet(), MerkleTree.Uploadable::adaptToDigest);
    }

    @VisibleForTesting
    public Map<Digest, Object> blobs() {
      return blobs.entrySet().stream()
          .collect(
              ImmutableMap.toImmutableMap(
                  entry -> adaptToDigest(entry.getKey()),
                  entry -> {
                    var value = entry.getValue();
                    if (value instanceof Object[] directory) {
                      // Serialize compact directory representation to bytes for test compatibility.
                      var out = new java.io.ByteArrayOutputStream();
                      try {
                        DirectoryBuilder.writeTo(out, directory, emptyDigest);
                      } catch (java.io.IOException e) {
                        throw new IllegalStateException("Failed to serialize directory", e);
                      }
                      return out.toByteArray();
                    }
                    return value;
                  }));
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
      var blob = blobs.get(digest);
      return switch (blob) {
        case VirtualActionInput virtualActionInput ->
            Optional.of(
                uploader.uploadDeterministicWriterOutput(context, digest, virtualActionInput));
        case ActionInput actionInput -> {
          var spawnExecutionContext = context.getSpawnExecutionContext();
          var pathResolver =
              spawnExecutionContext != null
                  ? spawnExecutionContext.getPathResolver()
                  // Used by MerkleTreeComputer#buildForFiles.
                  : MerkleTreeComputer.actionInputWithPathResolver;
          yield Optional.of(
              uploader.uploadFile(
                  context, remotePathResolver, digest, pathResolver.toPath(actionInput)));
        }
        case null -> Optional.empty();
        default ->
            Optional.of(
                uploader.uploadDeterministicWriterOutput(
                    context, digest, out -> DirectoryBuilder.writeTo(out, blob, emptyDigest)));
      };
    }

    private static Digest adaptToDigest(Object key) {
      return switch (key) {
        case Digest digest -> digest;
        case FileArtifactValue metadata ->
            DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
        default -> throw new IllegalStateException("Unexpected blob type: " + key);
      };
    }
  }
}
