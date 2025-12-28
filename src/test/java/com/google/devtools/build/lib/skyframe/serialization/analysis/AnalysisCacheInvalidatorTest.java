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
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.LongVersionClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.SnapshotClientId;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.IntVersion;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Optional;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Unit tests for {@link AnalysisCacheInvalidator}. */
@RunWith(TestParameterInjector.class)
public final class AnalysisCacheInvalidatorTest {

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();
  @Mock private RemoteAnalysisCacheClient mockAnalysisCacheClient;
  @Mock private ExtendedEventHandler mockEventHandler;

  private final ObjectCodecs objectCodecs = new ObjectCodecs();
  private final FrontierNodeVersion frontierNodeVersion = FrontierNodeVersion.CONSTANT_FOR_TESTING;
  private final ClientId baseClientId = new SnapshotClientId("for_testing", 1);
  private final FingerprintValueService fingerprintService =
      FingerprintValueService.createForTesting();

  @Test
  public void lookupKeysToInvalidate_emptyInput_returnsEmptySet() throws Exception {
    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            baseClientId,
            mockEventHandler);
    assertThat(
            invalidator.lookupKeysToInvalidate(
                ImmutableSet.of(),
                new RemoteAnalysisCachingServerState(
                    frontierNodeVersion, new SnapshotClientId("for_testing", 2))))
        .isEmpty();
  }

  @Test
  public void lookupKeysToInvalidate_cacheHit_returnsEmptySet() throws Exception {
    TrivialKey key = new TrivialKey("hit_key");
    PackedFingerprint fingerprint =
        FingerprintValueService.computeFingerprint(
            fingerprintService, objectCodecs, key, frontierNodeVersion);

    // Simulate a cache hit by returning a non-empty response.
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(fingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.copyFromUtf8("some_value")));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            baseClientId,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                ImmutableSet.of(key),
                new RemoteAnalysisCachingServerState(
                    frontierNodeVersion, new SnapshotClientId("for_testing", 2))))
        .isEmpty();
  }

  @Test
  public void lookupKeysToInvalidate_cacheMiss_returnsKey() throws Exception {
    TrivialKey key = new TrivialKey("miss_key");
    PackedFingerprint fingerprint =
        FingerprintValueService.computeFingerprint(
            fingerprintService, objectCodecs, key, frontierNodeVersion);

    // Simulate a cache miss by returning an empty response.
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(fingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.EMPTY));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            baseClientId,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                ImmutableSet.of(key),
                new RemoteAnalysisCachingServerState(
                    frontierNodeVersion, new SnapshotClientId("for_testing", 2))))
        .containsExactly(key);
  }

  @Test
  public void lookupKeysToInvalidate_mixedHitAndMiss_returnsMissedKey() throws Exception {
    TrivialKey hitKey = new TrivialKey("hit_key_mixed");
    TrivialKey missKey = new TrivialKey("miss_key_mixed");

    PackedFingerprint hitFingerprint =
        FingerprintValueService.computeFingerprint(
            fingerprintService, objectCodecs, hitKey, frontierNodeVersion);
    PackedFingerprint missFingerprint =
        FingerprintValueService.computeFingerprint(
            fingerprintService, objectCodecs, missKey, frontierNodeVersion);

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
            baseClientId,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                ImmutableSet.of(hitKey, missKey),
                new RemoteAnalysisCachingServerState(
                    frontierNodeVersion, new SnapshotClientId("for_testing", 2))))
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
            baseClientId,
            mockEventHandler);

    assertThat(
            invalidator.lookupKeysToInvalidate(
                ImmutableSet.of(key1, key2),
                new RemoteAnalysisCachingServerState(
                    previousVersion, new SnapshotClientId("for_testing", 2))))
        .containsExactly(key1, key2);

    // No RPCs should be sent.
    verify(mockAnalysisCacheClient, never()).lookup(any());
  }

  private enum ClientIdTestCase {
    NEWER_CLIENT_ID_CACHE_MISS_INVALIDATES(
        new SnapshotClientId("for_testing", 2),
        new SnapshotClientId("for_testing", 1),
        /* expectedInvalidated= */ true),
    OLDER_CLIENT_ID_CACHE_MISS_INVALIDATES(
        new SnapshotClientId("for_testing", 1),
        new SnapshotClientId("for_testing", 2),
        /* expectedInvalidated= */ true),
    SAME_CLIENT_ID_CACHE_MISS_DOES_NOT_INVALIDATE_ANYTHING(
        new SnapshotClientId("for_testing", 1),
        new SnapshotClientId("for_testing", 1),
        /* expectedInvalidated= */ false),
    SAME_LONG_VERSION_CLIENT_ID_CACHE_MISS_DOES_NOT_INVALIDATE_ANYTHING(
        new LongVersionClientId(123456789),
        new LongVersionClientId(123456789),
        /* expectedInvalidated= */ false),
    DIFFERENT_LONG_VERSION_CLIENT_ID_CACHE_MISS_INVALIDATES(
        new LongVersionClientId(123456789),
        new LongVersionClientId(123456788),
        /* expectedInvalidated= */ true),
    DIFFERENT_CLIENT_ID_SUBCLASS_CACHE_MISS_INVALIDATES(
        new LongVersionClientId(123456789),
        new SnapshotClientId("for_testing", 1),
        /* expectedInvalidated= */ true);

    private final ClientId currentClientId;
    private final ClientId previousClientId;
    private final boolean expectedInvalidated;

    ClientIdTestCase(
        ClientId currentClientId, ClientId previousClientId, boolean expectedInvalidated) {
      this.currentClientId = currentClientId;
      this.previousClientId = previousClientId;
      this.expectedInvalidated = expectedInvalidated;
    }
  }

  @Test
  public void lookupKeysToInvalidate_clientIdComparison(@TestParameter ClientIdTestCase testCase)
      throws Exception {
    TrivialKey key = new TrivialKey("key");
    PackedFingerprint packedFingerprint =
        FingerprintValueService.computeFingerprint(
            fingerprintService, objectCodecs, key, frontierNodeVersion);
    when(mockAnalysisCacheClient.lookup(ByteString.copyFrom(packedFingerprint.toBytes())))
        .thenReturn(immediateFuture(ByteString.EMPTY));

    AnalysisCacheInvalidator invalidator =
        new AnalysisCacheInvalidator(
            mockAnalysisCacheClient,
            objectCodecs,
            fingerprintService,
            /* currentVersion= */ frontierNodeVersion,
            testCase.currentClientId,
            mockEventHandler);

    ImmutableSet<SkyKey> keysToInvalidate =
        invalidator.lookupKeysToInvalidate(
            ImmutableSet.of(key),
            new RemoteAnalysisCachingServerState(frontierNodeVersion, testCase.previousClientId));

    if (testCase.expectedInvalidated) {
      assertThat(keysToInvalidate).containsExactly(key);
    } else {
      assertThat(keysToInvalidate).isEmpty();
    }
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
