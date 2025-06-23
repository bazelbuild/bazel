// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.SkyKeySerializationHelper;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.SnapshotClientId;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.IntVersion;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.util.Optional;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Unit tests for {@link AnalysisCacheInvalidator}. */
@RunWith(JUnit4.class)
public final class AnalysisCacheInvalidatorTest {

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();
  @Mock private RemoteAnalysisCacheClient mockAnalysisCacheClient;
  @Mock private ExtendedEventHandler mockEventHandler;

  private final ObjectCodecs objectCodecs = new ObjectCodecs();
  private final FrontierNodeVersion frontierNodeVersion = FrontierNodeVersion.CONSTANT_FOR_TESTING;
  private final FingerprintValueService fingerprintService =
      FingerprintValueService.createForTesting();

  @Test
  public void lookupKeysToInvalidate_emptyInput_returnsEmptySet() {
    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            mockEventHandler);
    assertThat(
            invalidator.lookupKeysToInvalidate(
                new RemoteAnalysisCachingState(frontierNodeVersion, ImmutableSet.of())))
        .isEmpty();
  }

  @Test
  public void lookupKeysToInvalidate_cacheHit_returnsEmptySet() throws Exception {
    TrivialKey key = new TrivialKey("hit_key");
    PackedFingerprint fingerprint =
        SkyKeySerializationHelper.computeFingerprint(
            objectCodecs, fingerprintService, key, frontierNodeVersion);

    // Simulate a cache hit by returning a non-empty response.
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(fingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.copyFromUtf8("some_value")));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                new RemoteAnalysisCachingState(frontierNodeVersion, ImmutableSet.of(key))))
        .isEmpty();
  }

  @Test
  public void lookupKeysToInvalidate_cacheMiss_returnsKey() throws Exception {
    TrivialKey key = new TrivialKey("miss_key");
    PackedFingerprint fingerprint =
        SkyKeySerializationHelper.computeFingerprint(
            objectCodecs, fingerprintService, key, frontierNodeVersion);

    // Simulate a cache miss by returning an empty response.
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(fingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.EMPTY));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                new RemoteAnalysisCachingState(frontierNodeVersion, ImmutableSet.of(key))))
        .containsExactly(key);
  }

  @Test
  public void lookupKeysToInvalidate_mixedHitAndMiss_returnsMissedKey() throws Exception {
    TrivialKey hitKey = new TrivialKey("hit_key_mixed");
    TrivialKey missKey = new TrivialKey("miss_key_mixed");

    PackedFingerprint hitFingerprint =
        SkyKeySerializationHelper.computeFingerprint(
            objectCodecs, fingerprintService, hitKey, frontierNodeVersion);
    PackedFingerprint missFingerprint =
        SkyKeySerializationHelper.computeFingerprint(
            objectCodecs, fingerprintService, missKey, frontierNodeVersion);

    // Simulate a cache hit _and_ miss for looking up multiple keys.
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(hitFingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.copyFromUtf8("some_value")));
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(missFingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.EMPTY));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                new RemoteAnalysisCachingState(
                    frontierNodeVersion, ImmutableSet.of(hitKey, missKey))))
        .containsExactly(missKey);
  }

  @Test
  public void lookupKeysToInvalidate_differentVersions_returnsAllKeys() throws Exception {
    TrivialKey key1 = new TrivialKey("key1");
    TrivialKey key2 = new TrivialKey("key2");

    var previousVersion =
        new FrontierNodeVersion(
            "123",
            HashCode.fromInt(42),
            IntVersion.of(9000),
            "distinguisher",
            /* useFakeStampData= */ true,
            Optional.of(new SnapshotClientId("for_testing", 123)));
    var currentVersion =
        new FrontierNodeVersion(
            "123",
            HashCode.fromInt(42),
            IntVersion.of(9001), // changed
            "distinguisher",
            /* useFakeStampData= */ true,
            Optional.of(new SnapshotClientId("for_testing", 123)));
    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            currentVersion,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                new RemoteAnalysisCachingState(previousVersion, ImmutableSet.of(key1, key2))))
        .containsExactly(key1, key2);

    // No RPCs should be sent.
    verify(mockAnalysisCacheClient, never()).lookup(any());
  }

  @AutoCodec
  @VisibleForSerialization
  record TrivialKey(String text) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      throw new UnsupportedOperationException();
    }
  }
}
