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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import com.google.devtools.build.lib.util.Bucket;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CancellationException;
import javax.annotation.Nullable;

/** Encapsulates fingerprint keyed bytes storage system. */
@SkybridgeInterface
public interface FingerprintValueStore {
  /** Usage statistics. */
  @SkybridgeInterface
  record Stats(
      long valueBytesReceived,
      long valueBytesSent,
      long keyBytesSent,
      long entriesWritten,
      long entriesFound,
      long entriesNotFound,
      long getBatches,
      long setBatches,
      List<Bucket> getLatencyMicros,
      List<Bucket> setLatencyMicros,
      List<Bucket> getBatchLatencyMicros,
      List<Bucket> setBatchLatencyMicros) {}

  @SuppressWarnings("JdkImmutableCollections") // Keep the SkybridgeInterface simple.
  Stats EMPTY_STATS = new Stats(0, 0, 0, 0, 0, 0, 0, 0, List.of(), List.of(), List.of(), List.of());

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
   * store, or the get operation was cancelled.
   */
  final class MissingFingerprintValueException extends Exception {

    public MissingFingerprintValueException(KeyBytesProvider fingerprint) {
      this(fingerprint, /* cause= */ null);
    }

    public MissingFingerprintValueException(
        KeyBytesProvider fingerprint, @Nullable Throwable cause) {
      super("No remote value for " + fingerprint, cause);
    }

    public MissingFingerprintValueException(CancellationException cause) {
      super("Fingerprint value fetch cancelled", checkNotNull(cause));
    }
  }

}
