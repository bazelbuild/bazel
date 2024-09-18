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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import com.google.common.util.concurrent.ListenableFuture;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** Encapsulates fingerprint keyed bytes storage system. */
public interface FingerprintValueStore {
  /**
   * Associates a fingerprint with the serialized representation of some object.
   *
   * <p>The caller should deduplicate {@code put} calls to avoid multiple writes of the same
   * fingerprint.
   *
   * @return a future that completes when the write completes
   */
  ListenableFuture<Void> put(PackedFingerprint fingerprint, byte[] serializedBytes);

  /**
   * Retrieves the serialized bytes associated with {@code fingerprint}.
   *
   * <p>If the given fingerprint does not exist in the store, the returned future fails with a
   * {@link MissingFingerprintValueException}.
   *
   * <p>The caller should deduplicate {@code get} calls to avoid multiple fetches of the same
   * fingerprint.
   */
  ListenableFuture<byte[]> get(PackedFingerprint fingerprint) throws IOException;

  /**
   * {@link FingerprintValueStore#get} was called with a fingerprint that does not exist in the
   * store.
   */
  final class MissingFingerprintValueException extends Exception {

    public MissingFingerprintValueException(PackedFingerprint fingerprint) {
      this(fingerprint, /* cause= */ null);
    }

    public MissingFingerprintValueException(
        PackedFingerprint fingerprint, @Nullable Throwable cause) {
      super("No remote value for " + fingerprint, cause);
    }
  }

  static InMemoryFingerprintValueStore inMemoryStore() {
    return new InMemoryFingerprintValueStore();
  }

  /** An in-memory {@link FingerprintValueStore} for testing. */
  static class InMemoryFingerprintValueStore implements FingerprintValueStore {
    private final ConcurrentHashMap<PackedFingerprint, byte[]> fingerprintToContents =
        new ConcurrentHashMap<>();

    @Override
    public ListenableFuture<Void> put(PackedFingerprint fingerprint, byte[] serializedBytes) {
      fingerprintToContents.put(fingerprint, serializedBytes);
      return immediateVoidFuture();
    }

    @Override
    public ListenableFuture<byte[]> get(PackedFingerprint fingerprint) {
      byte[] serializedBytes = fingerprintToContents.get(fingerprint);
      if (serializedBytes == null) {
        return immediateFailedFuture(new MissingFingerprintValueException(fingerprint));
      }
      return immediateFuture(serializedBytes);
    }
  }
}
