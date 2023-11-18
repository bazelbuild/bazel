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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationDependencyProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.function.Function;
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
 * <pre>{@code A -> fingerprint(FPb, C)}</pre>
 *
 * <p>On retrieval, A will be reconstructed by first retrieving A using its fingerprint, and then
 * recursively retrieving B using its fingerprint.
 */
public class NestedSetStore {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final Duration FETCH_FROM_STORAGE_LOGGING_THRESHOLD = Duration.ofSeconds(5);

  /**
   * Exception indicating that {@link NestedSetStorageEndpoint#get} was called with a fingerprint
   * that does not exist in the store.
   */
  public static final class MissingNestedSetException extends Exception {

    public MissingNestedSetException(ByteString fingerprint) {
      this(fingerprint, /*cause=*/ null);
    }

    public MissingNestedSetException(ByteString fingerprint, @Nullable Throwable cause) {
      super("No NestedSet data for " + fingerprint, cause);
    }
  }

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
     * <p>If the given fingerprint does not exist in the store, the returned future fails with a
     * {@link MissingNestedSetException}.
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
      return immediateFuture(null);
    }

    @Override
    public ListenableFuture<byte[]> get(ByteString fingerprint) {
      byte[] serializedBytes = fingerprintToContents.get(fingerprint);
      if (serializedBytes == null) {
        return immediateFailedFuture(new MissingNestedSetException(fingerprint));
      }
      return immediateFuture(serializedBytes);
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

    abstract ListenableFuture<Void> writeStatus();
  }

  public static final Function<SerializationDependencyProvider, ?> NO_CONTEXT = ctx -> "";

  private final NestedSetStorageEndpoint endpoint;
  private final Executor executor;
  private final NestedSetSerializationCache nestedSetCache;
  private final Function<SerializationDependencyProvider, ?> cacheContextFn;

  /**
   * Creates a NestedSetStore with the provided {@link NestedSetStorageEndpoint} and executor for
   * deserialization.
   *
   * <p>Takes a function that produces a caching context object from a {@link
   * SerializationDependencyProvider}. The context should work as described in {@link
   * NestedSetSerializationCache} to disambiguate different contents that have the same serialized
   * representation. If a one-to-one correspondence between contents and serialized representation
   * is guaranteed, use {@link #NO_CONTEXT}, which uses a constant object for the cache context.
   */
  public NestedSetStore(
      NestedSetStorageEndpoint endpoint,
      Executor executor,
      BugReporter bugReporter,
      Function<SerializationDependencyProvider, ?> cacheContextFn) {
    this(endpoint, executor, new NestedSetSerializationCache(bugReporter), cacheContextFn);
  }

  @VisibleForTesting
  NestedSetStore(
      NestedSetStorageEndpoint endpoint,
      Executor executor,
      NestedSetSerializationCache nestedSetCache,
      Function<SerializationDependencyProvider, ?> cacheContextFn) {
    this.endpoint = checkNotNull(endpoint);
    this.executor = checkNotNull(executor);
    this.nestedSetCache = checkNotNull(nestedSetCache);
    this.cacheContextFn = checkNotNull(cacheContextFn);
  }

  /** Creates a NestedSetStore with an in-memory storage backend and no caching context. */
  public static NestedSetStore inMemory() {
    return new NestedSetStore(
        new InMemoryNestedSetStorageEndpoint(),
        directExecutor(),
        BugReporter.defaultInstance(),
        NO_CONTEXT);
  }

  /**
   * Computes and returns the fingerprint for the given {@link NestedSet} contents using the given
   * {@link SerializationContext}, while also associating the contents with the computed fingerprint
   * in the store. Recursively does the same for all transitive {@code Object[]} members of the
   * provided contents.
   *
   * <p>We wish to compute a fingerprint for each array only once. However, this is not currently
   * enforced, due to the check-then-act race below, where we check {@link
   * NestedSetSerializationCache#fingerprintForContents} and then, significantly later, call {@link
   * NestedSetSerializationCache#putIfAbsent}. It is not straightforward to solve this with a
   * typical cache loader because the fingerprint computation is recursive, and cache loaders must
   * not attempt to update the cache while loading a result. Even if we duplicate fingerprint
   * computation, only one thread will end up calling {@link NestedSetStorageEndpoint#put} (the one
   * that wins the race to {@link NestedSetSerializationCache#putIfAbsent}).
   */
  FingerprintComputationResult computeFingerprintAndStore(
      Object[] contents, SerializationContext serializationContext)
      throws SerializationException, IOException {
    return computeFingerprintAndStore(
        contents, serializationContext, cacheContextFn.apply(serializationContext));
  }

  private FingerprintComputationResult computeFingerprintAndStore(
      Object[] contents, SerializationContext serializationContext, Object cacheContext)
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
              computeFingerprintAndStore((Object[]) child, serializationContext, cacheContext);
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
    SettableFuture<Void> localWriteFuture = SettableFuture.create();
    futureBuilder.add(localWriteFuture);

    // If this is a NestedSet<NestedSet>, serialization of the contents will itself have writes.
    ListenableFuture<Void> innerWriteFutures =
        newSerializationContext.createFutureToBlockWritingOn();
    if (innerWriteFutures != null) {
      futureBuilder.add(innerWriteFutures);
    }

    ListenableFuture<Void> writeFuture =
        Futures.whenAllSucceed(futureBuilder.build()).call(() -> null, directExecutor());
    FingerprintComputationResult result =
        FingerprintComputationResult.create(fingerprint, writeFuture);

    FingerprintComputationResult existingResult =
        nestedSetCache.putIfAbsent(contents, result, cacheContext);
    if (existingResult != null) {
      return existingResult; // Another thread won the fingerprint computation race.
    }

    // This fingerprint was not cached previously, so we must ensure that it is written to storage.
    localWriteFuture.setFuture(endpoint.put(fingerprint, serializedBytes));
    return result;
  }

  @SuppressWarnings("unchecked")
  private static ListenableFuture<Object[]> maybeWrapInFuture(Object contents) {
    if (contents instanceof Object[]) {
      return immediateFuture((Object[]) contents);
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
   * <p>The return value is either an {@code Object[]} or a {@code ListenableFuture<Object[]>},
   * which may be completed with a {@link MissingNestedSetException}.
   */
  Object getContentsAndDeserialize(
      ByteString fingerprint, DeserializationContext deserializationContext) throws IOException {
    return getContentsAndDeserialize(
        fingerprint, deserializationContext, cacheContextFn.apply(deserializationContext));
  }

  // All callers will test on type and check return value if it's a future.
  @SuppressWarnings("FutureReturnValueIgnored")
  private Object getContentsAndDeserialize(
      ByteString fingerprint, DeserializationContext deserializationContext, Object cacheContext)
      throws IOException {
    SettableFuture<Object[]> future = SettableFuture.create();
    Object contents = nestedSetCache.putFutureIfAbsent(fingerprint, future, cacheContext);
    if (contents != null) {
      return contents;
    }
    ListenableFuture<byte[]> retrieved = endpoint.get(fingerprint);
    Stopwatch fetchStopwatch = Stopwatch.createStarted();
    future.setFuture(
        Futures.transformAsync(
            retrieved,
            bytes -> {
              // The future should have failed with MissingNestedSetException if the fingerprint
              // was not present in the NestedSetStorageEndpoint.
              checkNotNull(bytes);

              Duration fetchDuration = fetchStopwatch.elapsed();
              if (FETCH_FROM_STORAGE_LOGGING_THRESHOLD.compareTo(fetchDuration) < 0) {
                logger.atInfo().log(
                    "NestedSet fetch took: %dms, size: %dB",
                    fetchDuration.toMillis(), bytes.length);
              }

              CodedInputStream codedIn = CodedInputStream.newInstance(bytes);
              int numberOfElements = codedIn.readInt32();
              DeserializationContext newDeserializationContext =
                  deserializationContext.getNewMemoizingContext();

              // The elements of this list are futures for the deserialized values of these
              // NestedSet contents. For direct members, the futures complete immediately and yield
              // an Object. For transitive members (fingerprints), the futures complete with the
              // underlying fetch, and yield Object[]s.
              ImmutableList.Builder<ListenableFuture<?>> deserializationFutures =
                  ImmutableList.builderWithExpectedSize(numberOfElements);
              for (int i = 0; i < numberOfElements; i++) {
                Object deserializedElement = newDeserializationContext.deserialize(codedIn);
                if (deserializedElement instanceof ByteString) {
                  Object innerContents =
                      getContentsAndDeserialize(
                          (ByteString) deserializedElement, deserializationContext, cacheContext);
                  deserializationFutures.add(maybeWrapInFuture(innerContents));
                } else {
                  deserializationFutures.add(Futures.immediateFuture(deserializedElement));
                }
              }

              return Futures.transform(
                  Futures.allAsList(deserializationFutures.build()), List::toArray, executor);
            },
            executor));
    return future;
  }
}
