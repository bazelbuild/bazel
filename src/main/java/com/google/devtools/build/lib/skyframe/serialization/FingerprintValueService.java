// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static java.util.concurrent.Executors.newSingleThreadExecutor;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;

/**
 * Bundles the components needed to store serialized values by fingerprint, the storage interface,
 * the cache and the hash function for computing fingerprints.
 */
public final class FingerprintValueService {
  private final Executor executor;
  private final FingerprintValueStore store;
  private final FingerprintValueCache cache;

  /**
   * The function used to generate fingerprints.
   *
   * <p>Used to derive {@link #fingerprintPlaceholder} and {@link #fingerprintLength}.
   */
  private final HashFunction hashFunction;

  private final ByteString fingerprintPlaceholder;
  private final int fingerprintLength;

  @VisibleForTesting
  public static FingerprintValueService createForTesting() {
    return createForTesting(
        FingerprintValueStore.inMemoryStore(), /* exerciseDeserializationForTesting= */ true);
  }

  @VisibleForTesting
  public static FingerprintValueService createForTesting(FingerprintValueStore store) {
    return createForTesting(store, /* exerciseDeserializationForTesting= */ true);
  }

  @VisibleForTesting
  public static FingerprintValueService createForTesting(
      boolean exerciseDeserializationForTesting) {
    return createForTesting(
        FingerprintValueStore.inMemoryStore(), exerciseDeserializationForTesting);
  }

  private static FingerprintValueService createForTesting(
      FingerprintValueStore store, boolean exerciseDeserializationForTesting) {
    return new FingerprintValueService(
        newSingleThreadExecutor(),
        store,
        new FingerprintValueCache(exerciseDeserializationForTesting),
        Hashing.murmur3_128());
  }

  public FingerprintValueService(
      Executor executor,
      FingerprintValueStore store,
      FingerprintValueCache cache,
      HashFunction hashFunction) {
    this.executor = executor;
    this.store = store;
    this.cache = cache;
    this.hashFunction = hashFunction;

    this.fingerprintPlaceholder = fingerprint(new byte[] {});
    this.fingerprintLength = fingerprintPlaceholder.size();
  }

  /** Delegates to {@link FingerprintValueStore#put}. */
  ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
    return store.put(fingerprint, serializedBytes);
  }

  /** Delegates to {@link FingerprintValueStore#get}. */
  ListenableFuture<byte[]> get(ByteString fingerprint) throws IOException {
    return store.get(fingerprint);
  }

  /** Delegates to {@link FingerprintValueCache#getOrClaimPutOperation}. */
  @Nullable
  Object getOrClaimPutOperation(
      Object obj, @Nullable Object distinguisher, ListenableFuture<PutOperation> putOperation) {
    return cache.getOrClaimPutOperation(obj, distinguisher, putOperation);
  }

  /** Delegates to {@link FingerprintValueCache#getOrClaimGetOperation}. */
  @Nullable
  Object getOrClaimGetOperation(
      ByteString fingerprint,
      @Nullable Object distinguisher,
      ListenableFuture<Object> getOperation) {
    return cache.getOrClaimGetOperation(fingerprint, distinguisher, getOperation);
  }

  /** Computes the fingerprint of {@code bytes}. */
  ByteString fingerprint(byte[] bytes) {
    return ByteString.copyFrom(hashFunction.hashBytes(bytes).asBytes());
  }

  /**
   * A placeholder fingerprint to use when the actual fingerprint is not yet available.
   *
   * <p>The placeholder has the same length as the real fingerprint so the real fingerprint can
   * overwrite the placeholder when it becomes available.
   */
  ByteString fingerprintPlaceholder() {
    return fingerprintPlaceholder;
  }

  /** The fixed length of fingerprints. */
  int fingerprintLength() {
    return fingerprintLength;
  }

  /**
   * Executor for chaining work on top of futures returned by {@link #put} or {@link #get}.
   *
   * <p>Those callbacks may be executing on RPC threads that should not be blocked.
   */
  Executor getExecutor() {
    return executor;
  }
}
