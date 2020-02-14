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
import com.google.common.base.Stopwatch;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.logging.Logger;
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

  private static final Logger logger = Logger.getLogger(NestedSetStore.class.getName());
  private static final Duration FETCH_FROM_STORAGE_LOGGING_THRESHOLD = Duration.ofSeconds(5);

  /** Stores fingerprint -> NestedSet associations. */
  public interface NestedSetStorageEndpoint {
    /**
     * Associates a fingerprint with the serialized representation of some NestedSet contents.
     * Returns a future that completes when the write completes.
     *
     * <p>It is the responsibility of the caller to deduplicate {@code put} calls, to avoid multiple
     * writes of the same fingerprint.
     */
    ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) throws IOException;

    /**
     * Retrieves the serialized bytes for the NestedSet contents associated with this fingerprint.
     *
     * <p>It is the responsibility of the caller to deduplicate {@code get} calls, to avoid multiple
     * fetches of the same fingerprint.
     */
    ListenableFuture<byte[]> get(ByteString fingerprint) throws IOException;
  }

  /** An in-memory {@link NestedSetStorageEndpoint} */
  @VisibleForTesting
  public static class InMemoryNestedSetStorageEndpoint implements NestedSetStorageEndpoint {
    private final ConcurrentHashMap<ByteString, byte[]> fingerprintToContents =
        new ConcurrentHashMap<>();

    @Override
    public ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
      fingerprintToContents.put(fingerprint, serializedBytes);
      return Futures.immediateFuture(null);
    }

    @Override
    public ListenableFuture<byte[]> get(ByteString fingerprint) {
      return Futures.immediateFuture(fingerprintToContents.get(fingerprint));
    }
  }

  /** An in-memory cache for fingerprint <-> NestedSet associations. */
  @VisibleForTesting
  public static class NestedSetCache {

    /**
     * Fingerprint to {@link NestedSet#children} cache.
     *
     * <p>The values in this cache are always {@code Object[]} or {@code
     * ListenableFuture<Object[]>}, the same as the children field in NestedSet. We avoid a common
     * wrapper object both for memory efficiency and because our cache eviction policy is based on
     * value GC, and wrapper objects would defeat that.
     */
    private final Cache<ByteString, Object> fingerprintToContents =
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
     * Returns children (an {@code Object[]} or a {@code ListenableFuture<Object[]>}) for NestedSet
     * contents associated with the given fingerprint if there was already one. Otherwise associates
     * {@code future} with {@code fingerprint} and returns null.
     *
     * <p>Since the associated future is used as the basis for equality comparisons for deserialized
     * nested sets, it is critical that multiple calls with the same fingerprint don't override the
     * association.
     */
    @VisibleForTesting
    @Nullable
    Object putIfAbsent(ByteString fingerprint, ListenableFuture<Object[]> future) {
      Object result;
      // Guava's Cache doesn't have a #putIfAbsent method, so we emulate it here.
      try {
        result = fingerprintToContents.get(fingerprint, () -> future);
      } catch (ExecutionException e) {
        throw new IllegalStateException(e);
      }
      if (result.equals(future)) {
        // This is the first request of this fingerprint. We should put it.
        putAsync(fingerprint, future);
        return null;
      }
      return result;
    }

    /**
     * Retrieves the fingerprint associated with the given NestedSet contents, or null if the given
     * contents are not known.
     */
    @VisibleForTesting
    @Nullable
    FingerprintComputationResult fingerprintForContents(Object[] contents) {
      return contentsToFingerprint.getIfPresent(contents);
    }

    /**
     * Associates the provided {@code fingerprint} and contents of the future, when it completes.
     *
     * <p>There may be a race between this call and calls to {@link #put}. Those races are benign,
     * since the fingerprint should be the same regardless. We may pessimistically end up having a
     * future to wait on for serialization that isn't actually necessary, but that isn't a big
     * concern.
     */
    private void putAsync(ByteString fingerprint, ListenableFuture<Object[]> futureContents) {
      futureContents.addListener(
          () -> {
            // There may already be an entry here, but it's better to put a fingerprint result with
            // an immediate future, since then later readers won't need to block unnecessarily. It
            // would be nice to sanity check the old value, but Cache#put doesn't provide it to us.
            try {
              contentsToFingerprint.put(
                  Futures.getDone(futureContents),
                  FingerprintComputationResult.create(fingerprint, Futures.immediateFuture(null)));

            } catch (ExecutionException e) {
              // Failure to fetch the NestedSet contents is unexpected, but the failed future can
              // be stored as the NestedSet children. This way the exception is only propagated if
              // the NestedSet is consumed (unrolled).
              BugReport.sendBugReport(
                  new IllegalStateException(
                      "Expected write for " + fingerprint + " to be complete", e));
            }
          },
          MoreExecutors.directExecutor());
    }

    // TODO(janakr): Currently, racing threads can overwrite each other's
    // fingerprintComputationResult, leading to confusion and potential performance drag. Fix this.
    public void put(FingerprintComputationResult fingerprintComputationResult, Object[] contents) {
      contentsToFingerprint.put(contents, fingerprintComputationResult);
      fingerprintToContents.put(fingerprintComputationResult.fingerprint(), contents);
    }
  }

  /** The result of a fingerprint computation, including the status of its storage. */
  @AutoValue
  abstract static class FingerprintComputationResult {
    static FingerprintComputationResult create(
        ByteString fingerprint, ListenableFuture<Void> writeStatus) {
      return new AutoValue_NestedSetStore_FingerprintComputationResult(fingerprint, writeStatus);
    }

    abstract ByteString fingerprint();

    @VisibleForTesting
    abstract ListenableFuture<Void> writeStatus();
  }

  private final NestedSetCache nestedSetCache;
  private final NestedSetStorageEndpoint nestedSetStorageEndpoint;
  private final Executor executor;

  /** Creates a NestedSetStore with the provided {@link NestedSetStorageEndpoint} as a backend. */
  @VisibleForTesting
  public NestedSetStore(NestedSetStorageEndpoint nestedSetStorageEndpoint) {
    this(nestedSetStorageEndpoint, new NestedSetCache(), MoreExecutors.directExecutor());
  }

  /**
   * Creates a NestedSetStore with the provided {@link NestedSetStorageEndpoint} and executor for
   * deserialization.
   */
  public NestedSetStore(NestedSetStorageEndpoint nestedSetStorageEndpoint, Executor executor) {
    this(nestedSetStorageEndpoint, new NestedSetCache(), executor);
  }

  @VisibleForTesting
  public NestedSetStore(
      NestedSetStorageEndpoint nestedSetStorageEndpoint,
      NestedSetCache nestedSetCache,
      Executor executor) {
    this.nestedSetStorageEndpoint = nestedSetStorageEndpoint;
    this.nestedSetCache = nestedSetCache;
    this.executor = executor;
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
   *
   * <p>We wish to serialize each nested set only once. However, this is not currently enforced, due
   * to the check-then-act race below, where we check nestedSetCache and then, significantly later,
   * insert a result into the cache. This is a bug, but since any thread that redoes unnecessary
   * work will return the {@link FingerprintComputationResult} containing its own futures, the
   * serialization work that must wait on remote storage writes to complete will wait on the correct
   * futures. Thus it is a performance bug, not a correctness bug.
   */
  // TODO(janakr): fix this, if for no other reason than to make the semantics cleaner.
  FingerprintComputationResult computeFingerprintAndStore(
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

  @SuppressWarnings("unchecked")
  private static ListenableFuture<Object[]> maybeWrapInFuture(Object contents) {
    if (contents instanceof Object[]) {
      return Futures.immediateFuture((Object[]) contents);
    }
    return (ListenableFuture<Object[]>) contents;
  }

  /**
   * Retrieves and deserializes the NestedSet contents associated with the given fingerprint.
   *
   * <p>We wish to only do one deserialization per fingerprint. This is enforced by the {@link
   * #nestedSetCache}, which is responsible for returning the actual contents or the canonical
   * future that will contain the results of the deserialization. If that future is not owned by the
   * current call of this method, it doesn't have to do anything further.
   *
   * <p>The return value is either an {@code Object[]} or a {@code ListenableFuture<Object[]>}.
   */
  // All callers will test on type and check return value if it's a future.
  @SuppressWarnings("FutureReturnValueIgnored")
  Object getContentsAndDeserialize(
      ByteString fingerprint, DeserializationContext deserializationContext) throws IOException {
    SettableFuture<Object[]> future = SettableFuture.create();
    Object contents = nestedSetCache.putIfAbsent(fingerprint, future);
    if (contents != null) {
      return contents;
    }
    ListenableFuture<byte[]> retrieved = nestedSetStorageEndpoint.get(fingerprint);
    Stopwatch fetchStopwatch = Stopwatch.createStarted();
    future.setFuture(
        Futures.transformAsync(
            retrieved,
            bytes -> {
              Duration fetchDuration = fetchStopwatch.elapsed();
              if (FETCH_FROM_STORAGE_LOGGING_THRESHOLD.compareTo(fetchDuration) < 0) {
                logger.info(
                    String.format(
                        "NestedSet fetch took: %dms, size: %dB",
                        fetchDuration.toMillis(), bytes.length));
              }

              CodedInputStream codedIn = CodedInputStream.newInstance(bytes);
              int numberOfElements = codedIn.readInt32();
              DeserializationContext newDeserializationContext =
                  deserializationContext.getNewMemoizingContext();

              // The elements of this list are futures for the deserialized values of these
              // NestedSet contents.  For direct members, the futures complete immediately and yield
              // an Object.  For transitive members (fingerprints), the futures complete with the
              // underlying fetch, and yield Object[]s.
              List<ListenableFuture<?>> deserializationFutures = new ArrayList<>();
              for (int i = 0; i < numberOfElements; i++) {
                Object deserializedElement = newDeserializationContext.deserialize(codedIn);
                if (deserializedElement instanceof ByteString) {
                  deserializationFutures.add(
                      maybeWrapInFuture(
                          getContentsAndDeserialize(
                              (ByteString) deserializedElement, deserializationContext)));
                } else {
                  deserializationFutures.add(Futures.immediateFuture(deserializedElement));
                }
              }

              return Futures.whenAllComplete(deserializationFutures)
                  .call(
                      () -> {
                        Object[] deserializedContents = new Object[deserializationFutures.size()];
                        for (int i = 0; i < deserializationFutures.size(); i++) {
                          deserializedContents[i] = Futures.getDone(deserializationFutures.get(i));
                        }
                        return deserializedContents;
                      },
                      executor);
            },
            executor));
    return future;
  }
}
