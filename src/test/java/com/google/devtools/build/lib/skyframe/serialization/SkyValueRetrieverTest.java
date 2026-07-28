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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.skyframe.serialization.DependOnFutureShim.ObservedFutureStatus.DONE;
import static com.google.devtools.build.lib.skyframe.serialization.DependOnFutureShim.ObservedFutureStatus.NOT_DONE;
import static com.google.devtools.build.lib.skyframe.serialization.ExampleValue.exampleValueCodec;
import static com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion.CONSTANT_FOR_TESTING;
import static com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.Restart.RESTART;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.FakeInvalidationDataHelper.prependFakeInvalidationData;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.DependOnFutureShim.ObservedFutureStatus;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.PeerFailedException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.NoCachedData;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalContext;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievedValue;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.WaitingForCacheServiceResponse;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.WaitingForLookupContinuation;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.SnapshotClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.LookupResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.LookupResultImpl;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheClient;
import com.google.devtools.build.lib.skyframe.serialization.analysis.proto.MissReason;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.IntVersion;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class SkyValueRetrieverTest {
  /** Default implementation that errors if any keys are requested. */
  private static final EnvironmentForUtilities NO_LOOKUP_ENVIRONMENT =
      new EnvironmentForUtilities(
          k -> {
            throw new IllegalStateException("no requests expected, got " + k);
          });

  private ObjectCodecs codecs = new ObjectCodecs();

  private static SkyValueRetriever createSkyValueRetriever(
      FingerprintValueService fingerprintValueService,
      ObjectCodecs codecs,
      FrontierNodeVersion frontierNodeVersion) {
    return new SkyValueRetriever(fingerprintValueService, codecs, frontierNodeVersion);
  }

  private enum InitialQueryCases {
    IMMEDIATE_EMPTY_VALUE,
    IMMEDIATE_MISSING_VALUE,
    FUTURE_VALUE
  }


  @Test
  public void initialQueryState_withAnalysisCacheService_progressesToWaiting(
      @TestParameter InitialQueryCases testCase) throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var data = new HashMap<ByteString, ByteString>();
    var captured = new ArrayList<SettableFuture<LookupResult>>();
    RemoteAnalysisCacheClient analysisCacheClient =
        switch (testCase) {
          case IMMEDIATE_EMPTY_VALUE, IMMEDIATE_MISSING_VALUE ->
              createFakeAnalysisCacheClient(data);
          // The capturing client places a SettableFuture into `captured`. Not setting it is
          // sufficient to elicit a Restart.RESTART response.
          case FUTURE_VALUE -> createCapturingAnalysisCacheClient(captured::add);
        };

    var key = new TrivialKey("a");
    SerializationResult<ByteString> keyBytes =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, key);
    assertThat(keyBytes.getFutureToBlockWritesOn()).isNull();

    if (testCase.equals(InitialQueryCases.IMMEDIATE_EMPTY_VALUE)) {
      assertThat(
              fingerprintValueService
                  .put(fingerprintValueService.fingerprint(keyBytes.getObject()), new byte[0])
                  .get())
          .isTrue();
    }

    RetrievalContext state = new RetrievalContext();

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    if (testCase.equals(InitialQueryCases.FUTURE_VALUE)) {
      assertThat(state.getState()).isInstanceOf(WaitingForCacheServiceResponse.class);
      assertThat(result).isEqualTo(RESTART);
    } else {
      result =
          maybeWaitForAnalysisCacheService(
              fingerprintValueService, analysisCacheClient, state, key, result);
      assertThat(state.getState()).isInstanceOf(NoCachedData.class);
      assertThat(((NoCachedData) result).reason()).isEqualTo(MissReason.MISS_REASON_SKYVALUE_MISS);
    }
  }


  @Test
  public void waitingForCacheServiceResponse_returnsValue() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    var key = new TrivialKey("a");
    var value = new TrivialValue("abc");

    uploadKeyValuePair(key, value, fingerprintValueService, analysisCacheServiceData);

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    result =
        maybeWaitForAnalysisCacheService(
            fingerprintValueService, analysisCacheClient, state, key, result);

    assertThat(((RetrievedValue) result).value()).isEqualTo(value);
    assertThat(state.getState()).isInstanceOf(RetrievedValue.class);
  }

  private RetrievalResult maybeWaitForAnalysisCacheService(
      FingerprintValueService fingerprintValueService,
      RemoteAnalysisCacheClient analysisCacheClient,
      RetrievalContext state,
      SkyKey key,
      RetrievalResult previousResult)
      throws SerializationException, ExecutionException, InterruptedException {
    if (state.getState()
        instanceof WaitingForCacheServiceResponse(ListenableFuture<LookupResult> futureResult)) {
      // There's a race condition here due to the RequestBatcher's response handling executor.
      // Most of the time, the test thread will outrace the executor and require a restart, but
      // RequestBatcher could occasionally outrace this thread.

      // Waits for the future to complete and simulates a restart.
      var unused = futureResult.get();
      return createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
          .tryRetrieve(
              NO_LOOKUP_ENVIRONMENT,
              SkyValueRetrieverTest::dependOnFutureImpl,
              analysisCacheClient,
              key,
              state);
    }
    return previousResult;
  }


  private PackedFingerprint fingerprintObject(
      FingerprintValueService fingerprintValueService, Object o) throws Exception {
    @SuppressWarnings("unchecked") // codec() returns ObjectCodec<?>
    var codec =
        (ObjectCodec<Object>) codecs.getCodecRegistry().getCodecDescriptorForObject(o).codec();
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    outputStream.write((byte) 0); // Assume the object is not compressed, see maybeCompressBytes()
    CodedOutputStream cos = CodedOutputStream.newInstance(outputStream);
    codec.serialize(codecs.getSerializationContextForTesting(), o, cos);
    cos.flush();
    return fingerprintValueService.fingerprint(outputStream.toByteArray());
  }

  @Test
  public void missingReferencedValue_resultsInObjectMiss() throws Exception {
    InMemoryFingerprintValueStore store = new InMemoryFingerprintValueStore(true);
    var fingerprintValueService = FingerprintValueService.createForTesting(store);
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    ValueWithReferenceCodec codec = new ValueWithReferenceCodec();
    codecs = codecs.withCodecOverridesForTesting(ImmutableList.of(codec));

    var key = new TrivialKey("a");
    var v1 = new ValueWithReference(1, null);
    var v2 = new ValueWithReference(2, v1);
    var v3 = new ValueWithReference(3, v2);
    var v4 = new ValueWithReference(4, v3);

    PackedFingerprint skyValueFingerprint =
        uploadKeyValuePair(key, v4, fingerprintValueService, analysisCacheServiceData);
    PackedFingerprint v1Fingerprint = fingerprintObject(fingerprintValueService, v1);

    store.remove(v1Fingerprint);
    SerializationException e =
        assertThrows(
            SerializationException.class,
            () ->
                createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
                    .tryRetrieve(
                        NO_LOOKUP_ENVIRONMENT,
                        SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                        analysisCacheClient,
                        key,
                        state));
    assertThat(e.getReason()).isEqualTo(MissReason.MISS_REASON_REFERENCED_OBJECT_MISS);

    // Also check just in case that if we remove the SkyValue entry, we get a SKYVALUE_MISS
    store.remove(skyValueFingerprint);
    analysisCacheServiceData.remove(ByteString.copyFrom(skyValueFingerprint.toBytes()));
    RetrievalContext state2 = new RetrievalContext();
    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state2);
    assertThat(((NoCachedData) result).reason()).isEqualTo(MissReason.MISS_REASON_SKYVALUE_MISS);
  }

  @Test
  public void tryRetrieve_withoutRestarts_returnsValue() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    codecs = codecs.withCodecOverridesForTesting(ImmutableList.of(new TrivialValueSharingCodec()));

    var key = new TrivialKey("a");
    var value = new TrivialValue("abc");
    uploadKeyValuePair(key, value, fingerprintValueService, analysisCacheServiceData);

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state);

    assertThat(((RetrievedValue) result).value()).isEqualTo(value);
  }

  @Test
  public void tryRetrieve_withInvalidMissReason_defaultsToUnspecified() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    RetrievalContext state = new RetrievalContext();
    var key = new TrivialKey("a");

    RemoteAnalysisCacheClient analysisCacheClient = mock(RemoteAnalysisCacheClient.class);
    when(analysisCacheClient.lookup(any()))
        .thenReturn(
            immediateFuture(
                new LookupResultImpl(
                    /* value= */ new byte[0], /* missReason= */ 999))); // invalid miss reason

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    result =
        maybeWaitForAnalysisCacheService(
            fingerprintValueService, analysisCacheClient, state, key, result);

    assertThat(state.getState()).isInstanceOf(NoCachedData.class);
    assertThat(((NoCachedData) result).reason()).isEqualTo(MissReason.MISS_REASON_UNSPECIFIED);
  }

  @Test
  public void tryRetrieve_withSkyframeRestart_completes() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    var key = new ExampleKey("a");
    var value = new ExampleValue(key, 10);
    uploadKeyValuePair(key, value, fingerprintValueService, analysisCacheServiceData);

    var capturedKey = new SkyKey[1];

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                new EnvironmentForUtilities(
                    k -> {
                      assertThat(capturedKey[0]).isNull();
                      capturedKey[0] = k;
                      return null;
                    }),
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isEqualTo(RESTART);
    assertThat(capturedKey[0]).isEqualTo(key);
    assertThat(state.getState()).isInstanceOf(WaitingForLookupContinuation.class);

    result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                new EnvironmentForUtilities(
                    k -> {
                      assertThat(k).isEqualTo(key);
                      return value;
                    }),
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state);

    assertThat(((RetrievedValue) result).value()).isEqualTo(value);
  }

  @Test
  public void retrievalError_throwsException() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var state = new RetrievalContext();
    var captured = new ArrayList<SettableFuture<LookupResult>>();
    RemoteAnalysisCacheClient analysisCacheClient =
        createCapturingAnalysisCacheClient(captured::add);

    var key = new TrivialKey("a");

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isEqualTo(RESTART);
    assertThat(state.getState()).isInstanceOf(WaitingForCacheServiceResponse.class);
    assertThat(captured).hasSize(1);

    var error = new IOException();
    captured.get(0).setException(error);

    var thrown =
        assertThrows(
            SerializationException.class,
            () ->
                createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
                    .tryRetrieve(
                        NO_LOOKUP_ENVIRONMENT,
                        SkyValueRetrieverTest::dependOnFutureImpl,
                        analysisCacheClient,
                        key,
                        state));

    assertThat(thrown).hasMessageThat().contains("getting cache response for " + key);
    assertThat(thrown).hasCauseThat().hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void retrievalCancelled_returnsNoCachedData() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var state = new RetrievalContext();
    var captured = new ArrayList<SettableFuture<LookupResult>>();
    RemoteAnalysisCacheClient analysisCacheClient =
        createCapturingAnalysisCacheClient(captured::add);

    var key = new TrivialKey("a");

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isEqualTo(RESTART);
    assertThat(state.getState()).isInstanceOf(WaitingForCacheServiceResponse.class);
    assertThat(captured).hasSize(1);

    captured.get(0).cancel(false);

    result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                NO_LOOKUP_ENVIRONMENT,
                SkyValueRetrieverTest::dependOnFutureImpl,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isInstanceOf(NoCachedData.class);
    assertThat(state.getState()).isInstanceOf(NoCachedData.class);
    assertThat(((NoCachedData) result).reason()).isEqualTo(MissReason.MISS_REASON_UNSPECIFIED);
  }

  @Test
  public void skyframeLookupError_throwsException() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    var key = new ExampleKey("a");
    var value = new ExampleValue(key, 10);
    uploadKeyValuePair(key, value, fingerprintValueService, analysisCacheServiceData);

    var capturedKey = new SkyKey[1];

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                new EnvironmentForUtilities(
                    k -> {
                      assertThat(capturedKey[0]).isNull();
                      capturedKey[0] = k;
                      return null;
                    }),
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isEqualTo(RESTART);
    assertThat(capturedKey[0]).isEqualTo(key);
    assertThat(state.getState()).isInstanceOf(WaitingForLookupContinuation.class);

    var error = new Exception();

    var thrown =
        assertThrows(
            SerializationException.class,
            () ->
                createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
                    .tryRetrieve(
                        new EnvironmentForUtilities(
                            k -> {
                              assertThat(k).isEqualTo(key);
                              return error;
                            }),
                        SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                        analysisCacheClient,
                        key,
                        state));

    assertThat(thrown)
        .hasMessageThat()
        .contains("skyframe dependency error during deserialization for " + key);
    assertThat(thrown).hasCauseThat().isInstanceOf(SkyframeDependencyException.class);
    assertThat(thrown).hasCauseThat().hasCauseThat().isSameInstanceAs(error);
  }

  @Test
  public void skyframeLookupError_marksOtherLookupsAbandoned() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    var key = new TrivialKey("a");

    var lookupKey0 = new ExampleKey("a");
    var lookupKey1 = new ExampleKey("b");
    var multiLookupValue =
        new MultiLookupValue(new ExampleValue(lookupKey0, 3), new ExampleValue(lookupKey1, 5));
    uploadKeyValuePair(key, multiLookupValue, fingerprintValueService, analysisCacheServiceData);

    var capturedKeys = new ArrayList<SkyKey>();

    RetrievalResult result =
        createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
            .tryRetrieve(
                new EnvironmentForUtilities(
                    k -> {
                      capturedKeys.add(k);
                      return null;
                    }),
                SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                analysisCacheClient,
                key,
                state);

    assertThat(result).isEqualTo(RESTART);
    assertThat(capturedKeys).containsExactly(lookupKey0, lookupKey1).inOrder();
    assertThat(state.getState()).isInstanceOf(WaitingForLookupContinuation.class);

    var lookups =
        ImmutableList.copyOf(
            ((WaitingForLookupContinuation) state.getState())
                .continuation()
                .getSkyframeLookupsForTesting());
    assertThat(lookups).hasSize(2);

    var error = new Exception();
    var thrown =
        assertThrows(
            SerializationException.class,
            () ->
                createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
                    .tryRetrieve(
                        new EnvironmentForUtilities(
                            k -> {
                              assertThat(k).isEqualTo(lookupKey0);
                              return error;
                            }),
                        SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                        analysisCacheClient,
                        key,
                        state));
    assertThat(thrown)
        .hasMessageThat()
        .contains("skyframe dependency error during deserialization for " + key);
    assertThat(thrown).hasCauseThat().isInstanceOf(SkyframeDependencyException.class);
    assertThat(thrown).hasCauseThat().hasCauseThat().isSameInstanceAs(error);

    var thrownByLookup0 = assertThrows(ExecutionException.class, lookups.get(0)::get).getCause();
    assertThat(thrownByLookup0).isInstanceOf(SkyframeDependencyException.class);
    assertThat(thrownByLookup0).hasCauseThat().isSameInstanceAs(error);

    var thrownByLookup1 = assertThrows(ExecutionException.class, lookups.get(1)::get).getCause();
    assertThat(thrownByLookup1).isInstanceOf(PeerFailedException.class);
    assertThat(thrownByLookup1).hasCauseThat().isSameInstanceAs(thrownByLookup0);
  }

  @Test
  public void exceptionWhileWaitingForResult_throwsException() throws Exception {
    var fingerprintValueService = FingerprintValueService.createForAnalysisCacheTesting();
    var analysisCacheServiceData = new HashMap<ByteString, ByteString>();
    var state = new RetrievalContext();
    RemoteAnalysisCacheClient analysisCacheClient =
        createFakeAnalysisCacheClient(analysisCacheServiceData);

    codecs = codecs.withCodecOverridesForTesting(ImmutableList.of(new FaultyTrivialValueCodec()));

    var key = new TrivialKey("k");
    var value = new TrivialValue("v");
    uploadKeyValuePair(key, value, fingerprintValueService, analysisCacheServiceData);

    var thrown =
        assertThrows(
            SerializationException.class,
            () ->
                createSkyValueRetriever(fingerprintValueService, codecs, CONSTANT_FOR_TESTING)
                    .tryRetrieve(
                        NO_LOOKUP_ENVIRONMENT,
                        SkyValueRetrieverTest::alwaysDoneDependOnFuture,
                        analysisCacheClient,
                        key,
                        state));

    assertThat(thrown).hasMessageThat().contains("waiting for deserialization result for " + key);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("error setting value");
  }

  @Test
  public void frontierNodeVersions_areEqual_ifTupleComponentsAreEqual() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifTopLevelConfigChecksumIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "CHANGED",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifBlazeInstallMD5IsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(9000),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifStarlarkSemanticsIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {4, 5, 6},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifEvaluatingVersionIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(10000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifDistinguisherIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "changed",
            true,
            Optional.empty());
    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areNotEqual_ifUseFakeStampDataIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            false,
            Optional.empty());
    assertThat(first.getPrecomputedFingerprint()).isNotEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void frontierNodeVersions_areEqual_evenIfSnapshotIsDifferent() {
    var first =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.of(new SnapshotClientId("changed", 123)));
    var second =
        new FrontierNodeVersion(
            "foo",
            HashCode.fromInt(42),
            new byte[] {1, 2, 3},
            IntVersion.of(9000),
            "distinguisher",
            true,
            Optional.empty());

    assertThat(first.getPrecomputedFingerprint()).isEqualTo(second.getPrecomputedFingerprint());
    assertThat(first).isEqualTo(second);
  }

  @CanIgnoreReturnValue
  private PackedFingerprint uploadKeyValuePair(
      SkyKey key,
      SkyValue value,
      FingerprintValueService fingerprintValueService,
      @Nullable Map<ByteString, ByteString> analysisCacheServiceData)
      throws SerializationException, InterruptedException, ExecutionException {
    return uploadKeyValuePair(
        key, CONSTANT_FOR_TESTING, value, fingerprintValueService, analysisCacheServiceData);
  }

  @CanIgnoreReturnValue
  private PackedFingerprint uploadKeyValuePair(
      SkyKey key,
      FrontierNodeVersion version,
      SkyValue value,
      FingerprintValueService fingerprintValueService,
      @Nullable Map<ByteString, ByteString> analysisCacheServiceData)
      throws SerializationException, InterruptedException, ExecutionException {
    SerializationResult<ByteString> keyBytes =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, key);
    ListenableFuture<?> writeStatus = keyBytes.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      var unused = writeStatus.get();
    }

    SerializationResult<ByteString> valueBytes =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, value);
    writeStatus = keyBytes.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      var unused = writeStatus.get();
    }

    var keyFingerprint =
        fingerprintValueService.fingerprint(version.concat(keyBytes.getObject().toByteArray()));

    if (analysisCacheServiceData != null) {
      analysisCacheServiceData.put(
          ByteString.copyFrom(keyFingerprint.toBytes()), valueBytes.getObject());
    } else {
      var unused =
          fingerprintValueService
              .put(
                  keyFingerprint, prependFakeInvalidationData(valueBytes.getObject()).toByteArray())
              .get();
    }

    return keyFingerprint;
  }

  private static RemoteAnalysisCacheClient createFakeAnalysisCacheClient(
      Map<ByteString, ByteString> data) {
    RemoteAnalysisCacheClient result = mock(RemoteAnalysisCacheClient.class);
    when(result.lookup(any()))
        .thenAnswer(
            invocation -> {
              ByteString key = ByteString.copyFrom((byte[]) invocation.getArgument(0));
              ByteString value = data.getOrDefault(key, ByteString.empty());
              return immediateFuture(
                  new LookupResultImpl(
                      value.toByteArray(),
                      value.isEmpty()
                          ? MissReason.MISS_REASON_SKYVALUE_MISS.getNumber()
                          : MissReason.MISS_REASON_UNSPECIFIED.getNumber()));
            });

    return result;
  }

  /**
   * Creates a {@link RequestBatcher} that emits a {@link SettableFuture} per request.
   *
   * <p>The client sets the {@link SettableFuture} to complete the request.
   */
  private static RemoteAnalysisCacheClient createCapturingAnalysisCacheClient(
      Consumer<SettableFuture<LookupResult>> capturer) {
    RemoteAnalysisCacheClient result = mock(RemoteAnalysisCacheClient.class);

    when(result.lookup(any()))
        .thenAnswer(
            invocation -> {
              var settable = SettableFuture.<LookupResult>create();
              capturer.accept(settable);
              return settable;
            });

    return result;
  }

  private static ObservedFutureStatus dependOnFutureImpl(ListenableFuture<?> future) {
    return future.isDone() ? DONE : NOT_DONE;
  }

  private static ObservedFutureStatus alwaysDoneDependOnFuture(ListenableFuture<?> future) {
    // Although the in-memory FingerprintValueStore is synchronous, the returned bytes are
    // processed asynchronously on an executor. There are 3 places where this may be called and 2
    // where the future might still be unset.
    //
    // 1. At the end of WaitingForFutureValueBytes, there's a wait for the
    //    SkyframeLookupContinuation to become available. That happens on the executor
    //    thread that processes the shared bytes.
    // 2. At the end of WaitingForLookupContinuation, there's a small wait for the final result.
    //    This wait corresponds to setting the shared value in the parent and happens on the
    //    executor thread after 1 so the caller might observe an unset future.
    try {
      var unused = future.get();
    } catch (ExecutionException | InterruptedException e) {
      // Exceptions are ignored here but handled by the next state, which always calls getDone.
    }
    return DONE;
  }

  @AutoCodec
  @VisibleForSerialization
  record TrivialKey(String text) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      throw new UnsupportedOperationException();
    }
  }

  private static final record ValueWithReference(int id, ValueWithReference ref)
      implements SkyValue {}

  private static final class ValueWithReferenceBuilder
      implements DeferredValue<ValueWithReference> {
    private final int id;
    private ValueWithReference ref;

    private ValueWithReferenceBuilder(int id) {
      this.id = id;
    }

    private static void setRef(ValueWithReferenceBuilder builder, Object ref) {
      builder.ref = (ValueWithReference) ref;
    }

    @Override
    public ValueWithReference call() {
      return new ValueWithReference(id, ref);
    }
  }

  private static final class ValueWithReferenceCodec
      extends DeferredObjectCodec<ValueWithReference> {
    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<? extends ValueWithReference> getEncodedClass() {
      return ValueWithReference.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ValueWithReference obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj.id);
      if (obj.ref == null) {
        codedOut.writeBoolNoTag(false);
      } else {
        codedOut.writeBoolNoTag(true);
        context.putSharedValue(obj.ref, null, this, codedOut);
      }
    }

    @Override
    public DeferredValue<ValueWithReference> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int id = codedIn.readInt32();
      if (!codedIn.readBool()) {
        var simpleResult = SimpleDeferredValue.<ValueWithReference>create();
        SimpleDeferredValue.set(simpleResult, new ValueWithReference(id, null));
        return simpleResult;
      }

      ValueWithReferenceBuilder sharedResult = new ValueWithReferenceBuilder(id);
      context.getSharedValue(codedIn, null, this, sharedResult, ValueWithReferenceBuilder::setRef);
      return sharedResult;
    }
  }

  @AutoCodec
  @VisibleForSerialization
  record TrivialValue(String text) implements SkyValue {}

  private static final DeferredObjectCodec<TrivialValue> TRIVIAL_SKY_VALUE_CODEC =
      new SkyValueRetrieverTest_TrivialValue_AutoCodec();

  private static final class TrivialValueSharingCodec extends DeferredObjectCodec<TrivialValue> {
    @Override
    public Class<TrivialValue> getEncodedClass() {
      return TrivialValue.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(
        SerializationContext context, TrivialValue value, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(value, /* distinguisher= */ null, TRIVIAL_SKY_VALUE_CODEC, codedOut);
    }

    @Override
    public DeferredValue<TrivialValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = SimpleDeferredValue.<TrivialValue>create();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          TRIVIAL_SKY_VALUE_CODEC,
          builder,
          SimpleDeferredValue::set);
      return builder;
    }
  }

  private static final class FaultyTrivialValueCodec extends DeferredObjectCodec<TrivialValue> {
    @Override
    public Class<TrivialValue> getEncodedClass() {
      return TrivialValue.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(
        SerializationContext context, TrivialValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(obj, /* distinguisher= */ null, TRIVIAL_SKY_VALUE_CODEC, codedOut);
    }

    @Override
    public DeferredValue<TrivialValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = SimpleDeferredValue.<TrivialValue>create();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          TRIVIAL_SKY_VALUE_CODEC,
          builder,
          (b, v) -> {
            throw new SerializationException("error setting value");
          });
      return builder;
    }
  }

  /** Value that requires multiple Skyframe lookups to deserialize. */
  private record MultiLookupValue(ExampleValue value1, ExampleValue value2) implements SkyValue {}

  @Keep // used reflectively
  private static final class MultiLookupValueCodec extends DeferredObjectCodec<MultiLookupValue> {
    @Override
    public Class<MultiLookupValue> getEncodedClass() {
      return MultiLookupValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, MultiLookupValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.value1(), /* distinguisher= */ null, exampleValueCodec(), codedOut);
      context.putSharedValue(
          obj.value2(), /* distinguisher= */ null, exampleValueCodec(), codedOut);
    }

    @Override
    public DeferredValue<MultiLookupValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new MultiLookupValueBuilder();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          exampleValueCodec(),
          builder,
          MultiLookupValueBuilder::setValue1);
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          exampleValueCodec(),
          builder,
          MultiLookupValueBuilder::setValue2);
      return builder;
    }

    private static class MultiLookupValueBuilder implements DeferredValue<MultiLookupValue> {
      private ExampleValue value1;
      private ExampleValue value2;

      private void setValue1(Object obj) {
        this.value1 = (ExampleValue) obj;
      }

      private void setValue2(Object obj) {
        this.value2 = (ExampleValue) obj;
      }

      @Override
      public MultiLookupValue call() {
        return new MultiLookupValue(value1, value2);
      }
    }
  }
}
