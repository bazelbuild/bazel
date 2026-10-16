// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/** An in-memory {@link FingerprintValueStore} for testing. */
public class InMemoryFingerprintValueStore implements FingerprintValueStore {
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
