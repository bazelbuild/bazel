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

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.immediateWriteStatus;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.WriteStatus;
import com.google.devtools.build.lib.util.Bucket;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/** Encapsulates fingerprint keyed bytes storage system. */
public interface FingerprintValueStore {
  /** Usage statistics. */
  record Stats(
      long valueBytesReceived,
      long valueBytesSent,
      long keyBytesSent,
      long entriesWritten,
      long entriesFound,
      long entriesNotFound,
      long getBatches,
      long setBatches,
      ImmutableList<Bucket> getLatencyMicros,
      ImmutableList<Bucket> setLatencyMicros,
      ImmutableList<Bucket> getBatchLatencyMicros,
      ImmutableList<Bucket> setBatchLatencyMicros) {}

  Stats EMPTY_STATS =
      new Stats(
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          ImmutableList.of(),
          ImmutableList.of(),
          ImmutableList.of(),
          ImmutableList.of());

  default Stats getStats() {
    return EMPTY_STATS;
  }

  default void shutdown() {}

  /**
   * Associates a fingerprint with the serialized representation of some object.
   *
   * <p>The caller should deduplicate {@code put} calls to avoid multiple writes of the same
   * fingerprint.
   *
   * @return a future that completes when the write completes
   */
  WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes);

  /**
   * Retrieves the serialized bytes associated with {@code fingerprint}.
   *
   * @return a future eventually containing the serialized bytes. If the fingerprint is missing, the
   *     future may contain null or a failed future, depending on the implementation.
   */
  ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) throws IOException;

  /**
   * {@link FingerprintValueStore#get} was called with a fingerprint that does not exist in the
   * store.
   */
  final class MissingFingerprintValueException extends Exception {

    public MissingFingerprintValueException(KeyBytesProvider fingerprint) {
      this(fingerprint, /* cause= */ null);
    }

    public MissingFingerprintValueException(
        KeyBytesProvider fingerprint, @Nullable Throwable cause) {
      super("No remote value for " + fingerprint, cause);
    }
  }

  static InMemoryFingerprintValueStore inMemoryStore() {
    return new InMemoryFingerprintValueStore();
  }

  /** An in-memory {@link FingerprintValueStore} for testing. */
  static class InMemoryFingerprintValueStore implements FingerprintValueStore {
    private static final ListenableFuture<byte[]> IMMEDIATE_NULL = immediateFuture((byte[]) null);

    public final ConcurrentMap<ByteString, ByteString> fingerprintToContents;

    private final boolean useNullForMissingValues;

    public InMemoryFingerprintValueStore() {
      this(/* useNullForMissingValues= */ false);
    }

    public InMemoryFingerprintValueStore(boolean useNullForMissingValues) {
      this(new ConcurrentHashMap<>(), useNullForMissingValues);
    }

    public InMemoryFingerprintValueStore(
        ConcurrentMap<ByteString, ByteString> kvMap, boolean useNullForMissingValues) {
      this.fingerprintToContents = kvMap;
      this.useNullForMissingValues = useNullForMissingValues;
    }

    @Override
    public WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes) {
      boolean wasNovel =
          (fingerprintToContents.put(
                  ByteString.copyFrom(fingerprint.toBytes()), ByteString.copyFrom(serializedBytes))
              == null);
      return immediateWriteStatus(wasNovel);
    }

    @Override
    public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) {
      ByteString serializedBytes =
          fingerprintToContents.get(ByteString.copyFrom(fingerprint.toBytes()));
      if (serializedBytes == null) {
        return useNullForMissingValues
            ? IMMEDIATE_NULL
            : immediateFailedFuture(new MissingFingerprintValueException(fingerprint));
      }
      return immediateFuture(serializedBytes.toByteArray());
    }

    @Nullable
    @CanIgnoreReturnValue
    public byte[] remove(KeyBytesProvider fingerprint) {
      ByteString result = fingerprintToContents.remove(ByteString.copyFrom(fingerprint.toBytes()));
      return result == null ? null : result.toByteArray();
    }

    public Iterable<ByteString> keys() {
      return ImmutableList.copyOf(fingerprintToContents.keySet());
    }
  }
}
