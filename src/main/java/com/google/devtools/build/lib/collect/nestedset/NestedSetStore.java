// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Supports association between fingerprints and NestedSet contents. A single NestedSetStore
 * instance should be globally available across a single process.
 *
 * <p>Maintains the fingerprint -> contents side of the bimap by decomposing nested Object[]'s.
 *
 * <p>For example, suppose the NestedSet A can be drawn as:
 *
 * <pre>
 *         A
 *       /  \
 *      B   C
 *     / \
 *    D  E
 * </pre>
 *
 * <p>Then, in memory, A = [[D, E], C]. To store the NestedSet, we would rely on the fingerprint
 * value FPb = fingerprint([D, E]) and write
 *
 * <pre>A -> fingerprint(FPb, C)</pre>
 *
 * <p>On retrieval, A will be reconstructed by first retrieving A using its fingerprint, and then
 * recursively retrieving B using its fingerprint.
 */
public class NestedSetStore {
  /** Stores fingerprint -> NestedSet associations. */
  public interface NestedSetStorageEndpoint {
    /**
     * Associates a fingerprint with the serialized representation of some NestedSet contents.
     * Returns a future that completes when the write completes.
     */
    ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) throws IOException;

    /**
     * Retrieves the serialized bytes for the NestedSet contents associated with this fingerprint.
     */
    byte[] get(ByteString fingerprint) throws IOException;
  }

  /** An in-memory {@link NestedSetStorageEndpoint} */
  private static class InMemoryNestedSetStorageEndpoint implements NestedSetStorageEndpoint {
    private final ConcurrentHashMap<ByteString, byte[]> fingerprintToContents =
        new ConcurrentHashMap<>();

    @Override
    public ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
      fingerprintToContents.put(fingerprint, serializedBytes);
      return Futures.immediateFuture(null);
    }

    @Override
    public byte[] get(ByteString fingerprint) {
      return fingerprintToContents.get(fingerprint);
    }
  }

  /** An in-memory cache for fingerprint <-> NestedSet associations. */
  private static class NestedSetCache {
    private final Cache<ByteString, Object[]> fingerprintToContents =
        CacheBuilder.newBuilder()
            .concurrencyLevel(SerializationConstants.DESERIALIZATION_POOL_SIZE)
            .weakValues()
            .build();

    /** Object/Object[] contents to fingerprint. Maintained for fast fingerprinting. */
    private final Cache<Object[], FingerprintComputationResult> contentsToFingerprint =
        CacheBuilder.newBuilder()
            .concurrencyLevel(SerializationConstants.DESERIALIZATION_POOL_SIZE)
            .weakKeys()
            .build();

    /**
     * Returns the NestedSet contents associated with the given fingerprint. Returns null if the
     * fingerprint is not known.
     */
    @Nullable
    public Object[] contentsForFingerprint(ByteString fingerprint) {
      return fingerprintToContents.getIfPresent(fingerprint);
    }

    /**
     * Retrieves the fingerprint associated with the given NestedSet contents, or null if the given
     * contents are not known.
     */
    @Nullable
    public FingerprintComputationResult fingerprintForContents(Object[] contents) {
      return contentsToFingerprint.getIfPresent(contents);
    }

    /** Associates the provided fingerprint and NestedSet contents. */
    public void put(FingerprintComputationResult fingerprintComputationResult, Object[] contents) {
      contentsToFingerprint.put(contents, fingerprintComputationResult);
      fingerprintToContents.put(fingerprintComputationResult.fingerprint(), contents);
    }
  }

  /** The result of a fingerprint computation, including the status of its storage. */
  @VisibleForTesting
  @AutoValue
  public abstract static class FingerprintComputationResult {
    static FingerprintComputationResult create(
        ByteString fingerprint, ListenableFuture<Void> writeStatus) {
      return new AutoValue_NestedSetStore_FingerprintComputationResult(fingerprint, writeStatus);
    }

    abstract ByteString fingerprint();

    @VisibleForTesting
    public abstract ListenableFuture<Void> writeStatus();
  }

  private final NestedSetCache nestedSetCache = new NestedSetCache();
  private final NestedSetStorageEndpoint nestedSetStorageEndpoint;

  /** Creates a NestedSetStore with the provided {@link NestedSetStorageEndpoint} as a backend. */
  public NestedSetStore(NestedSetStorageEndpoint nestedSetStorageEndpoint) {
    this.nestedSetStorageEndpoint = nestedSetStorageEndpoint;
  }

  /** Creates a NestedSetStore with an in-memory storage backend. */
  public static NestedSetStore inMemory() {
    return new NestedSetStore(new InMemoryNestedSetStorageEndpoint());
  }

  /**
   * Computes and returns the fingerprint for the given NestedSet contents using the given {@link
   * SerializationContext}, while also associating the contents with the computed fingerprint in the
   * store. Recursively does the same for all transitive members (i.e. Object[] members) of the
   * provided contents.
   */
  @VisibleForTesting
  public FingerprintComputationResult computeFingerprintAndStore(
      Object[] contents, SerializationContext serializationContext)
      throws SerializationException, IOException {
    FingerprintComputationResult priorFingerprint = nestedSetCache.fingerprintForContents(contents);
    if (priorFingerprint != null) {
      return priorFingerprint;
    }

    // For every fingerprint computation, we need to use a new memoization table.  This is required
    // to guarantee that the same child will always have the same fingerprint - otherwise,
    // differences in memoization context could cause part of a child to be memoized in one
    // fingerprinting but not in the other.  We expect this clearing of memoization state to be a
    // major source of extra work over the naive serialization approach.  The same value may have to
    // be serialized many times across separate fingerprintings.
    SerializationContext newSerializationContext = serializationContext.getNewMemoizingContext();
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    CodedOutputStream codedOutputStream = CodedOutputStream.newInstance(byteArrayOutputStream);

    ImmutableList.Builder<ListenableFuture<Void>> futureBuilder = ImmutableList.builder();
    try {
      codedOutputStream.writeInt32NoTag(contents.length);
      for (Object child : contents) {
        if (child instanceof Object[]) {
          FingerprintComputationResult fingerprintComputationResult =
              computeFingerprintAndStore((Object[]) child, serializationContext);
          futureBuilder.add(fingerprintComputationResult.writeStatus());
          newSerializationContext.serialize(
              fingerprintComputationResult.fingerprint(), codedOutputStream);
        } else {
          newSerializationContext.serialize(child, codedOutputStream);
        }
      }
      codedOutputStream.flush();
    } catch (IOException e) {
      throw new SerializationException("Could not serialize NestedSet contents", e);
    }

    byte[] serializedBytes = byteArrayOutputStream.toByteArray();
    ByteString fingerprint =
        ByteString.copyFrom(Hashing.md5().hashBytes(serializedBytes).asBytes());
    futureBuilder.add(nestedSetStorageEndpoint.put(fingerprint, serializedBytes));

    // If this is a NestedSet<NestedSet>, serialization of the contents will itself have writes.
    ListenableFuture<Void> innerWriteFutures =
        newSerializationContext.createFutureToBlockWritingOn();
    if (innerWriteFutures != null) {
      futureBuilder.add(innerWriteFutures);
    }

    ListenableFuture<Void> writeFuture =
        Futures.whenAllComplete(futureBuilder.build())
            .call(() -> null, MoreExecutors.directExecutor());
    FingerprintComputationResult fingerprintComputationResult =
        FingerprintComputationResult.create(fingerprint, writeFuture);

    nestedSetCache.put(fingerprintComputationResult, contents);

    return fingerprintComputationResult;
  }

  /** Retrieves and deserializes the NestedSet contents associated with the given fingerprint. */
  public Object[] getContentsAndDeserialize(
      ByteString fingerprint, DeserializationContext deserializationContext)
      throws SerializationException, IOException {
    Object[] contents = nestedSetCache.contentsForFingerprint(fingerprint);
    if (contents != null) {
      return contents;
    }

    byte[] retrieved = nestedSetStorageEndpoint.get(fingerprint);
    if (retrieved == null) {
      throw new AssertionError("Fingerprint " + fingerprint + " not found in NestedSetStore");
    }

    CodedInputStream codedIn = CodedInputStream.newInstance(retrieved);
    DeserializationContext newDeserializationContext =
        deserializationContext.getNewMemoizingContext();

    int numberOfElements = codedIn.readInt32();
    Object[] dereferencedContents = new Object[numberOfElements];
    for (int i = 0; i < numberOfElements; i++) {
      Object deserializedElement = newDeserializationContext.deserialize(codedIn);
      dereferencedContents[i] =
          deserializedElement instanceof ByteString
              ? getContentsAndDeserialize(
                  (ByteString) deserializedElement, deserializationContext)
              : deserializedElement;
    }

    FingerprintComputationResult fingerprintComputationResult =
        FingerprintComputationResult.create(fingerprint, Futures.immediateFuture(null));
    nestedSetCache.put(fingerprintComputationResult, dereferencedContents);
    return dereferencedContents;
  }
}
