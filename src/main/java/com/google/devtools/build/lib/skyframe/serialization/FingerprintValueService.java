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

import static com.google.common.hash.Hashing.murmur3_128;
import static com.google.common.io.BaseEncoding.base16;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.Executors.newSingleThreadExecutor;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore.InMemoryFingerprintValueStore;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.WriteStatus;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisJsonLogWriter;
import com.google.devtools.build.lib.util.DecimalBucketer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * Bundles the components needed to store serialized values by fingerprint, the storage interface,
 * the cache and the hash function for computing fingerprints.
 */
public final class FingerprintValueService implements KeyValueWriter {

  /** A {@link Fingerprinter} implementation for non-production use. */
  public static final Fingerprinter NONPROD_FINGERPRINTER =
      input -> PackedFingerprint.fromBytesOffsetZeros(murmur3_128().hashBytes(input).asBytes());

  private final Executor executor;
  private final FingerprintValueStore store;
  private final FingerprintValueCache cache;

  /**
   * The function used to generate fingerprints.
   *
   * <p>Used to derive {@link #fingerprintPlaceholder} and {@link #fingerprintLength}.
   */
  private final Fingerprinter fingerprinter;

  @Nullable // When log writing is not turned on
  private final RemoteAnalysisJsonLogWriter jsonLogWriter;

  private final PackedFingerprint fingerprintPlaceholder;
  private final int fingerprintLength;

  private final DecimalBucketer getLatencyMicros = new DecimalBucketer();
  private final DecimalBucketer setLatencyMicros = new DecimalBucketer();

  @VisibleForTesting
  public static FingerprintValueService createForTesting() {
    return createForTesting(
        FingerprintValueStore.inMemoryStore(), FingerprintValueCache.SyncMode.NOT_LINKED);
  }

  /**
   * Returns an instance that uses a {@link FingerprintValueStore} that indicates a missing entry by
   * returning null, which is what analysis caching expects.
   */
  @VisibleForTesting
  public static FingerprintValueService createForAnalysisCacheTesting() {
    return createForTesting(new InMemoryFingerprintValueStore(true));
  }

  @VisibleForTesting
  public static FingerprintValueService createForTesting(FingerprintValueStore store) {
    return createForTesting(store, FingerprintValueCache.SyncMode.NOT_LINKED);
  }

  @VisibleForTesting
  public static FingerprintValueService createForTesting(FingerprintValueCache.SyncMode mode) {
    return createForTesting(FingerprintValueStore.inMemoryStore(), mode);
  }

  private static FingerprintValueService createForTesting(
      FingerprintValueStore store, FingerprintValueCache.SyncMode mode) {
    return new FingerprintValueService(
        newSingleThreadExecutor(),
        store,
        new FingerprintValueCache(mode),
        NONPROD_FINGERPRINTER,
        /* jsonLogWriter= */ null);
  }

  public FingerprintValueService(
      Executor executor,
      FingerprintValueStore store,
      FingerprintValueCache cache,
      Fingerprinter fingerprinter,
      @Nullable RemoteAnalysisJsonLogWriter jsonLogWriter) {
    this.executor = executor;
    this.store = store;
    this.cache = cache;
    this.fingerprinter = fingerprinter;
    this.jsonLogWriter = jsonLogWriter;

    this.fingerprintPlaceholder = fingerprint(new byte[] {});
    this.fingerprintLength = fingerprintPlaceholder.toBytes().length;
  }

  /**
   * Serializes a {@link SkyKey}, concatenates it with the {@link FrontierNodeVersion}, computes the
   * fingerprint, and returns the {@link PackedFingerprint}.
   */
  public static PackedFingerprint computeFingerprint(
      FingerprintValueService fingerprintValueService,
      ObjectCodecs codecs,
      SkyKey key,
      FrontierNodeVersion nodeVersion)
      throws InterruptedException, SerializationException {
    ListenableFuture<SerializationResult<ByteString>> serializedKey =
        codecs.serializeMemoizedAsync(fingerprintValueService, key, null);

    ListenableFuture<PackedFingerprint> fingerprintFuture =
        Futures.transform(
            serializedKey,
            k ->
                fingerprintValueService.fingerprint(
                    nodeVersion.concat(k.getObject().toByteArray())),
            // Keys are hopefully small enough that it's reasonable to not spawn off a separate task
            directExecutor());

    try {
      return fingerprintFuture.get();
    } catch (ExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), SerializationException.class);
      throw new IllegalStateException(e);
    }
  }

  public void shutdown() {
    store.shutdown();
  }

  /** Delegates to {@link FingerprintValueStore#put}. */
  @Override
  public WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes) {
    int serializedBytesLength = serializedBytes.length;
    Instant before = Instant.now();
    WriteStatus putStatus = store.put(fingerprint, serializedBytes);
    putStatus.addListener(
        () ->
            setLatencyMicros.add(
                TimeUnit.NANOSECONDS.toMicros(Duration.between(before, Instant.now()).toNanos())),
        directExecutor());
    if (jsonLogWriter == null) {
      return putStatus;
    }

    return jsonLogWriter.logWrite(
        putStatus,
        e -> {
          try (var entry = jsonLogWriter.startEntry("fvsPut")) {
            entry.addField("start", before);
            entry.addField("end", Instant.now());
            entry.addField("key", base16().lowerCase().encode(fingerprint.toBytes()));
            entry.addField("valueSize", serializedBytesLength);
            if (e != null) {
              entry.addField("exception", e.getMessage());
            }
          }
        });
  }

  public FingerprintValueStore.Stats getStats() {
    FingerprintValueStore.Stats storeStats = store.getStats();
    return new FingerprintValueStore.Stats(
        storeStats.valueBytesReceived(),
        storeStats.valueBytesSent(),
        storeStats.keyBytesSent(),
        storeStats.entriesWritten(),
        storeStats.entriesFound(),
        storeStats.entriesNotFound(),
        storeStats.getBatches(),
        storeStats.setBatches(),
        getLatencyMicros.getBuckets(),
        setLatencyMicros.getBuckets(),
        storeStats.getBatchLatencyMicros(),
        storeStats.setBatchLatencyMicros());
  }

  /** Delegates to {@link FingerprintValueStore#get}. */
  public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) throws IOException {
    Instant before = Instant.now();
    ListenableFuture<byte[]> result = store.get(fingerprint);
    if (jsonLogWriter != null) {
      result =
          Futures.transform(
              result,
              b -> {
                try (var entry = jsonLogWriter.startEntry("fvsGet")) {
                  entry.addField("start", before);
                  entry.addField("end", Instant.now());
                  entry.addField("key", base16().lowerCase().encode(fingerprint.toBytes()));
                  entry.addField("responseSize", b.length);
                }
                return b;
              },
              directExecutor());
    }
    result.addListener(
        () ->
            getLatencyMicros.add(
                TimeUnit.NANOSECONDS.toMicros(Duration.between(before, Instant.now()).toNanos())),
        directExecutor());
    return result;
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
      PackedFingerprint fingerprint,
      @Nullable Object distinguisher,
      ListenableFuture<Object> getOperation) {
    return cache.getOrClaimGetOperation(fingerprint, distinguisher, getOperation);
  }

  /** Computes the fingerprint of {@code bytes}. */
  @Override
  public PackedFingerprint fingerprint(byte[] bytes) {
    return fingerprinter.fingerprint(bytes);
  }

  /** Convenience overload of {@link #fingerprint(byte[])}. */
  @VisibleForTesting
  PackedFingerprint fingerprint(ByteString bytes) {
    return fingerprint(bytes.toByteArray());
  }

  /**
   * A placeholder fingerprint to use when the actual fingerprint is not yet available.
   *
   * <p>The placeholder has the same length as the real fingerprint so the real fingerprint can
   * overwrite the placeholder when it becomes available.
   */
  PackedFingerprint fingerprintPlaceholder() {
    return fingerprintPlaceholder;
  }

  /** The fixed length of fingerprints. */
  int fingerprintLength() {
    return fingerprintLength;
  }

  /**
   * Executor for scheduling work related to serializing and deserializing values from the
   * fingerprint value store.
   *
   * <p>Technically, this should be plumbed separately but for the time being, {@link
   * FingerprintValueService} is a convenient container for the {@link Executor}.
   */
  public Executor getExecutor() {
    return executor;
  }

  @VisibleForTesting
  public FingerprintValueStore getStoreForTesting() {
    return store;
  }
}
