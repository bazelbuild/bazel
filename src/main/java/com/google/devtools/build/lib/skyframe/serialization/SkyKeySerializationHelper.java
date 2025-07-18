// Copyright 2025 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/Apache-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
//
// limitations under the License.

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;

/** Helper utility for serializing and fingerprinting {@link SkyKey}s. */
public final class SkyKeySerializationHelper {

  private SkyKeySerializationHelper() {}

  /**
   * Serializes a {@link SkyKey} using the provided codecs and service, handles the write status
   * future, and returns the serialized {@link ByteString}.
   */
  private static ByteString serializeSkyKeyAndHandleStatus(
      ObjectCodecs codecs, FingerprintValueService fingerprintValueService, SkyKey key)
      throws SerializationException {
    SerializationResult<ByteString> keyBytesResult =
        codecs.serializeMemoizedAndBlocking(
            fingerprintValueService, key, /* profileCollector= */ null);
    ListenableFuture<Void> keyWriteStatus = keyBytesResult.getFutureToBlockWritesOn();
    if (keyWriteStatus != null) {
      // This write status future represents the storage write success of the
      // shared bytes contained in the SkyKey (e.g. anything with a value
      // sharing codec). Since the SkyKey is never deserialized -- they are only
      // ever serialized into a fingerprint or a cache key -- failure to write
      // dependent shared bytes to storage is not a serious concern, but let's
      // report them anyway, in case of real serialization bugs.
      Futures.addCallback(
          keyWriteStatus, FutureHelpers.FAILURE_REPORTING_CALLBACK, directExecutor());
    }
    return keyBytesResult.getObject();
  }

  /**
   * Serializes a {@link SkyKey}, concatenates it with the {@link FrontierNodeVersion}, computes the
   * fingerprint, and returns the {@link PackedFingerprint}.
   */
  public static PackedFingerprint computeFingerprint(
      ObjectCodecs codecs,
      FingerprintValueService fingerprintValueService,
      SkyKey key,
      FrontierNodeVersion version)
      throws SerializationException {
    ByteString serializedKey = serializeSkyKeyAndHandleStatus(codecs, fingerprintValueService, key);
    return fingerprintValueService.fingerprint(version.concat(serializedKey.toByteArray()));
  }
}
