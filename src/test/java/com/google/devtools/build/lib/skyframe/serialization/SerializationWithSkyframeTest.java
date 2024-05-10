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
import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NestedArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.GetRecordingStore;
import com.google.devtools.build.lib.skyframe.serialization.testutils.GetRecordingStore.GetRequest;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class SerializationWithSkyframeTest {
  private final ObjectCodecs codecs =
      new ObjectCodecs(
          AutoRegistry.get()
              .getBuilder()
              .add(new NotNestedSetCodec(new NestedArrayCodec()))
              .build());
  private final GetRecordingStore recordingStore = new GetRecordingStore();
  private FingerprintValueService fingerprintValueService =
      FingerprintValueService.createForTesting(recordingStore);

  @Test
  public void serializeWithOneSkyKey(
      @TestParameter boolean missingValue, @TestParameter boolean injectSkyframeError)
      throws Exception {
    var key = new ExampleKey("test");
    var value = new ExampleValue(key, 10);

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, value);
    assertThat(serialized.getFutureToBlockWritesOn()).isNull();

    // Deserialization always returns a future because there is a Skyframe lookup. The future is
    // always done because there are no shared values to wait on.
    var continuation =
        (SkyframeLookupContinuation)
            Futures.getDone(
                (ListenableFuture<?>)
                    codecs.deserializeWithSkyframe(
                        fingerprintValueService, serialized.getObject()));

    if (missingValue) {
      // The continuation must resume, returning null, because the value is not in the injected
      // Skyframe environment.
      assertThat(processWithEntries(continuation, ImmutableMap.of())).isNull();
    }

    if (injectSkyframeError) {
      Exception error = new Exception("error");
      SkyframeDependencyException thrown =
          assertThrows(
              SkyframeDependencyException.class,
              () -> processWithEntries(continuation, ImmutableMap.of(key, error)));
      assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
    } else {
      // Injects the key-value pair into the environment. The continuation produces a result.
      ListenableFuture<?> futureValue =
          processWithEntries(continuation, ImmutableMap.of(key, value));
      // The result is done because there are no shared values.
      assertThat(Futures.getDone(futureValue)).isSameInstanceAs(value);
    }
  }

  @Test
  public void serializeWithTwoSkyKeys() throws Exception {
    var key1 = new ExampleKey("key1");
    var value1 = new ExampleValue(key1, 10);
    var key2 = new ExampleKey("key2");
    var value2 = new ExampleValue(key2, 20);

    var value = ImmutableList.<ExampleValue>of(value1, value2);

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, value);
    assertThat(serialized.getFutureToBlockWritesOn()).isNull();

    // Deserialization always returns a future because there is a Skyframe lookup. The future is
    // always done because there are no shared values to wait on.
    var continuation =
        (SkyframeLookupContinuation)
            Futures.getDone(
                (ListenableFuture<?>)
                    codecs.deserializeWithSkyframe(
                        fingerprintValueService, serialized.getObject()));

    // Evaluates the continuation as if `key1` is already present but `key2` is missing. It returns
    // null, requesting a restart.
    assertThat(processWithEntries(continuation, ImmutableMap.of(key1, value1))).isNull();

    ListenableFuture<?> futureValue =
        processWithEntries(continuation, ImmutableMap.of(key1, value1, key2, value2));
    // The future is done because there are no shared values.
    assertThat(Futures.getDone(futureValue)).isEqualTo(value);
  }

  @Test
  public void serializeInSharedValue(@TestParameter boolean fetchError) throws Exception {
    var key = new ExampleKey("test");
    var value = new ExampleValue(key, 1);
    var sharedValue = new SharedExampleValue(value);

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, sharedValue);
    ListenableFuture<Void> writeStatus = serialized.getFutureToBlockWritesOn();
    assertThat(writeStatus.get()).isNull();

    var futureResult =
        (ListenableFuture<?>)
            codecs.deserializeWithSkyframe(fingerprintValueService, serialized.getObject());
    assertThat(futureResult.isDone()).isFalse(); // not done because the fetch is blocked

    GetRequest request = recordingStore.takeFirstRequest();
    if (fetchError) {
      var error = new IOException();
      request.response().setException(error);
      var thrown = assertThrows(ExecutionException.class, futureResult::get);
      assertThat(thrown).hasCauseThat().isSameInstanceAs(error);
    } else {
      request.complete(); // completes the fetch
      var continuation = (SkyframeLookupContinuation) futureResult.get();
      ListenableFuture<?> futureValue =
          processWithEntries(continuation, ImmutableMap.of(key, value));
      // There may be a small amount of bookkeeping work in the shared values deserialization
      // threads so the following call may block.
      assertThat(futureValue.get()).isEqualTo(sharedValue);
    }
  }

  /**
   * Error scenarios for {@link #serializeWithCrossValueSharing}.
   *
   * <p>In this scenario, there are two concurrent deserializations, one for {@code subject0} and
   * one for {@code subject1} and they share a value. The test is arranged so that deserialization
   * of {@code subject0} owns the shared value. The scenarios here exercise the propagation of the
   * error from {@code subject0} to {@code subject1}.
   */
  static enum CrossError {
    NO_ERROR,
    /** Error occurs in {@link FingerprintValueStore#get}. */
    FETCH_ERROR,
    /** Error occurs in a Skyframe lookup. */
    SKY_VALUE_ERROR
  }

  @Test
  public void serializeWithCrossValueSharing(@TestParameter CrossError crossError)
      throws Exception {
    var key0 = new ExampleKey("key0");
    var value0 = new ExampleValue(key0, 1);
    var sharedValue0 = new SharedExampleValue(value0);

    var key1 = new ExampleKey("key1");
    var value1 = new ExampleValue(key1, 3);
    var sharedValue1 = new SharedExampleValue(value1);

    SharedExampleValue subject0 = sharedValue0;
    var subject1 = ImmutableList.<SharedExampleValue>of(sharedValue0, sharedValue1);

    var serializedBytes = new ArrayList<ByteString>();
    for (Object sharedValue : ImmutableList.of(subject0, subject1)) {
      SerializationResult<ByteString> serialized =
          codecs.serializeMemoizedAndBlocking(fingerprintValueService, sharedValue);
      ListenableFuture<Void> writeStatus = serialized.getFutureToBlockWritesOn();
      assertThat(writeStatus.get()).isNull();
      serializedBytes.add(serialized.getObject());
    }

    // Deserializing subject0 first makes futureResult0 own deserialization of value0.
    var futureResult0 =
        (ListenableFuture<?>)
            codecs.deserializeWithSkyframe(fingerprintValueService, serializedBytes.get(0));
    // As subject1 deserializes, it'll try to deserialize value0 but see that its deserialization is
    // already owned by another thread.
    var futureResult1 =
        (ListenableFuture<?>)
            codecs.deserializeWithSkyframe(fingerprintValueService, serializedBytes.get(1));

    var getRequest0 = recordingStore.takeFirstRequest();
    var getRequest1 = recordingStore.takeFirstRequest();

    // Completes the fetch associated with sharedValue1, but leaves sharedValue0 blocked.
    getRequest1.complete();
    // This allows futureResult1 to progress to its continuation. Although value0 is blocked, it is
    // owned by futureResult0 and doesn't interfere with futureResult1's continuation.
    var continuation1 = (SkyframeLookupContinuation) futureResult1.get();
    // On the other hand, futureResult0 is always blocked here because getRequest0 is blocked.
    assertThat(futureResult0.isDone()).isFalse();

    ListenableFuture<?> futureValue1 =
        processWithEntries(continuation1, ImmutableMap.of(key1, value1));
    // key1, value1 is the only Skyframe entry needed, so no restart. value0 is being deserialized
    // concurrently under futureResult0.
    assertThat(futureValue1).isNotNull();

    switch (crossError) {
      case NO_ERROR:
        {
          // Unblocks the remaining blocked fetch, enabling the next step of deserialization of
          // value0.
          getRequest0.complete();
          var continuation0 = (SkyframeLookupContinuation) futureResult0.get();

          assertThat(futureValue1.isDone()).isFalse(); // subject1 is still blocked on value0

          ListenableFuture<?> futureValue0 =
              processWithEntries(continuation0, ImmutableMap.of(key0, value0));
          assertThat(futureValue0).isNotNull(); // continuation0 only requires key0, value0.

          // Deserialization is fully unblocked.
          assertThat(Futures.allAsList(futureValue0, futureValue1).get())
              .containsExactly(subject0, subject1)
              .inOrder();
          break;
        }
      case FETCH_ERROR:
        {
          var fetchError = new IOException();
          getRequest0.response().setException(fetchError);
          var thrown0 = assertThrows(ExecutionException.class, futureResult0::get);
          assertThat(thrown0).hasCauseThat().isSameInstanceAs(fetchError);

          var thrown1 = assertThrows(ExecutionException.class, futureValue1::get);
          assertThat(thrown1).hasCauseThat().isSameInstanceAs(fetchError);
          break;
        }
      case SKY_VALUE_ERROR:
        {
          getRequest0.complete();
          var continuation0 = (SkyframeLookupContinuation) futureResult0.get();

          Exception skyValueError = new Exception("failed");
          var thrown0 =
              assertThrows(
                  SkyframeDependencyException.class,
                  () -> processWithEntries(continuation0, ImmutableMap.of(key0, skyValueError)));
          assertThat(thrown0).hasCauseThat().isSameInstanceAs(skyValueError);

          var thrown1 = assertThrows(ExecutionException.class, futureValue1::get);
          assertThat(Throwables.getRootCause(thrown1)).isSameInstanceAs(skyValueError);
          break;
        }
    }
  }

  @Test
  public void randomNestedData(@TestParameter({"10", "20", "50"}) int size) throws Exception {
    // Doesn't use the GetRecordingStore because it is too complex to use in this test.
    this.fingerprintValueService = FingerprintValueService.createForTesting();

    Random random = new Random(0);
    var entries = new HashMap<SkyKey, Object>();
    var subject =
        NotNestedSet.createRandom(
            random,
            size,
            size,
            rng -> {
              ExampleKey key = new ExampleKey(Integer.toString(rng.nextInt(1000)));
              Object value = entries.get(key);
              if (value != null) {
                return value;
              }
              value = new ExampleValue(key, rng.nextInt());
              entries.put(key, value);
              return value;
            });

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, subject);
    ListenableFuture<Void> writeStatus = serialized.getFutureToBlockWritesOn();
    assertThat(writeStatus.get()).isNull();

    var futureResult =
        (ListenableFuture<?>)
            codecs.deserializeWithSkyframe(fingerprintValueService, serialized.getObject());

    var continuation = (SkyframeLookupContinuation) futureResult.get();
    ListenableFuture<?> futureValue = processWithEntries(continuation, entries);

    assertThat(dumpStructureWithEquivalenceReduction(futureValue.get()))
        .isEqualTo(dumpStructureWithEquivalenceReduction(subject));
  }

  @Nullable
  private static ListenableFuture<?> processWithEntries(
      SkyframeLookupContinuation continuation, Map<SkyKey, Object> entries)
      throws InterruptedException, SkyframeDependencyException {
    return continuation.process(new EnvironmentForUtilities(entries::get));
  }

  private record ExampleKey(String name) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      throw new UnsupportedOperationException();
    }
  }

  /** A class that is deserialized using {@link DeserializationContext#getSkyValue}. */
  private record ExampleValue(ExampleKey key, int x) implements SkyValue {}

  private record SharedExampleValue(ExampleValue value) {}

  private static final class ExampleKeyCodec extends LeafObjectCodec<ExampleKey> {
    private static final ExampleKeyCodec INSTANCE = new ExampleKeyCodec();

    @Override
    public Class<ExampleKey> getEncodedClass() {
      return ExampleKey.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, ExampleKey key, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(key.name(), stringCodec(), codedOut);
    }

    @Override
    public ExampleKey deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new ExampleKey(context.deserializeLeaf(codedIn, stringCodec()));
    }
  }

  private static ExampleKeyCodec exampleKeyCodec() {
    return ExampleKeyCodec.INSTANCE;
  }

  private static final class ExampleValueCodec extends DeferredObjectCodec<ExampleValue> {
    private static final ExampleValueCodec INSTANCE = new ExampleValueCodec();

    @Override
    public Class<ExampleValue> getEncodedClass() {
      return ExampleValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ExampleValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.key(), exampleKeyCodec(), codedOut);
    }

    @Override
    public DeferredValue<ExampleValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ExampleKey key = context.deserializeLeaf(codedIn, exampleKeyCodec());
      var builder = new ExampleValueBuilder();
      context.getSkyValue(key, builder, ExampleValueBuilder::setValue);
      return builder;
    }
  }

  private static ExampleValueCodec exampleValueCodec() {
    return ExampleValueCodec.INSTANCE;
  }

  private static final class ExampleValueBuilder implements DeferredValue<ExampleValue> {
    private ExampleValue value;

    @Override
    public ExampleValue call() {
      return value;
    }

    private static void setValue(ExampleValueBuilder builder, Object obj) {
      builder.value = (ExampleValue) obj;
    }
  }

  @Keep // used reflectively
  private static final class SharedExampleValueCodec
      extends DeferredObjectCodec<SharedExampleValue> {
    @Override
    public Class<SharedExampleValue> getEncodedClass() {
      return SharedExampleValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, SharedExampleValue value, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          value.value(), /* distinguisher= */ null, exampleValueCodec(), codedOut);
    }

    @Override
    public DeferredValue<SharedExampleValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new SharedExampleValueBuilder();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          exampleValueCodec(),
          builder,
          SharedExampleValueBuilder::setValue);
      return builder;
    }
  }

  private static final class SharedExampleValueBuilder
      implements DeferredValue<SharedExampleValue> {
    private ExampleValue value;

    @Override
    public SharedExampleValue call() {
      return new SharedExampleValue(value);
    }

    private static void setValue(SharedExampleValueBuilder builder, Object obj) {
      builder.value = (ExampleValue) obj;
    }
  }
}
