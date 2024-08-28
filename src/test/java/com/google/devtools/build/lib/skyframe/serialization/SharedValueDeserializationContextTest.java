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
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.createRandomLeafArray;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.getFieldOffset;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableFutureTask;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NestedArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetDeferredCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.GetRecordingStore;
import com.google.devtools.build.lib.skyframe.serialization.testutils.GetRecordingStore.GetRequest;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class SharedValueDeserializationContextTest {
  private static final int CONCURRENCY = 20;

  private final ForkJoinPool executor = new ForkJoinPool(CONCURRENCY);
  private final Random rng = new Random(0);

  @Test
  @TestParameters("{size: 2, useDeferredCodec: false}")
  @TestParameters("{size: 4, useDeferredCodec: false}")
  @TestParameters("{size: 4, useDeferredCodec: true}")
  @TestParameters("{size: 8, useDeferredCodec: false}")
  @TestParameters("{size: 16, useDeferredCodec: false}")
  @TestParameters("{size: 16, useDeferredCodec: true}")
  @TestParameters("{size: 32, useDeferredCodec: false}")
  @TestParameters("{size: 64, useDeferredCodec: false}")
  @TestParameters("{size: 128, useDeferredCodec: false}")
  public void codec_roundTrips(int size, boolean useDeferredCodec) throws Exception {
    new SerializationTester(NotNestedSet.createRandom(rng, size, size, Random::nextInt))
        .addCodec(
            useDeferredCodec
                ? new NotNestedSetDeferredCodec(new NestedArrayCodec())
                : getDefaultNotNestedSetCodec())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(
            SharedValueDeserializationContextTest::verifyDeserializedNotNestedSet)
        .runTests();
  }

  @Test
  public void getsShouldBeConcurrent() throws Exception {
    // When deserializing a value, multiple calls to `FingerprintValueStore.get` may occur. These
    // should not block each other.

    GetRecordingStore store = new GetRecordingStore();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(store);
    ObjectCodecs codecs = createObjectCodecs();

    NotNestedSet subject =
        new NotNestedSet(
            new Object[] {
              createRandomLeafArray(rng, Random::nextInt),
              createRandomLeafArray(rng, Random::nextInt),
              createRandomLeafArray(rng, Random::nextInt)
            });

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(
            fingerprintValueService, subject, /* profileCollector= */ null);
    ListenableFuture<Void> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      // If it is asynchronous, writing should complete without throwing any exceptions.
      assertThat(writeStatus.get()).isNull();
    }

    ListenableFuture<Object> result =
        deserializeWithExecutor(codecs, fingerprintValueService, serialized.getObject());

    // There are 4 nested arrays. The top-level one and its 3 child arrays. The child arrays aren't
    // requested until the top-level array is requested. Completes the top-level request.
    store.takeFirstRequest().complete();

    // The 3 child requests should become available. Since none of them are complete, they must be
    // concurrent.
    ArrayList<GetRequest> childGets = new ArrayList<>(3);
    for (int i = 0; i < 3; i++) {
      childGets.add(store.takeFirstRequest());
    }

    // Since the child requests have not been satisfied, the result can't be done yet.
    assertThat(result.isDone()).isFalse();

    // Completes the child requests and verifies the result.
    for (GetRequest request : childGets) {
      request.complete();
    }
    verifyDeserializedNotNestedSet(subject, (NotNestedSet) result.get());
  }

  private static class NotNestedSetContainer {
    private static final long FIRST_OFFSET;
    private static final long SECOND_OFFSET;

    private NotNestedSet first;
    private NotNestedSet second;

    private NotNestedSetContainer() {}

    private NotNestedSetContainer(NotNestedSet first, NotNestedSet second) {
      this.first = first;
      this.second = second;
    }

    static {
      try {
        FIRST_OFFSET = getFieldOffset(NotNestedSetContainer.class, "first");
        SECOND_OFFSET = getFieldOffset(NotNestedSetContainer.class, "second");
      } catch (ReflectiveOperationException e) {
        throw new ExceptionInInitializerError(e);
      }
    }
  }

  /** Selects the {@link AsyncDeserializationContext#deserialize} overload. */
  private enum DeserializeOverloadSelector {
    OFFSET,
    SETTER,
    OFFSET_WITH_DONE_CALLBACK
  }

  /** Codec that observes futures through {@link AsyncObjectCodec#deserialize} overloads. */
  private static final class NotNestedSetContainerCodec
      extends AsyncObjectCodec<NotNestedSetContainer> {
    private final DeserializeOverloadSelector overloadSelector;
    private final NotNestedSet expectedFirst;
    private final NotNestedSet expectedSecond;

    private NotNestedSetContainerCodec(
        DeserializeOverloadSelector overloadSelector,
        NotNestedSet expectedFirst,
        NotNestedSet expectedSecond) {
      this.overloadSelector = overloadSelector;
      this.expectedFirst = expectedFirst;
      this.expectedSecond = expectedSecond;
    }

    @Override
    public Class<NotNestedSetContainer> getEncodedClass() {
      return NotNestedSetContainer.class;
    }

    @Override
    public void serialize(
        SerializationContext context, NotNestedSetContainer container, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(container.first, codedOut);
      context.serialize(container.second, codedOut);
    }

    @Override
    public NotNestedSetContainer deserializeAsync(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      NotNestedSetContainer value = new NotNestedSetContainer();
      context.registerInitialValue(value);
      // The additional verifications in the code below are redundant with the ones performed by the
      // SerializationTester except that they occur at the moment the context provides the values by
      // callback. This enables verification that the provided values are fully deserialized as soon
      // as they are set, as required by the specification.
      switch (overloadSelector) {
        case OFFSET:
          context.deserialize(codedIn, value, NotNestedSetContainer.FIRST_OFFSET);
          context.deserialize(codedIn, value, NotNestedSetContainer.SECOND_OFFSET);
          break;
        case SETTER:
          context.deserialize(
              codedIn,
              value,
              (container, untypedFirst) -> {
                NotNestedSet first = (NotNestedSet) untypedFirst;
                container.first = first;
                verifyDeserializedNotNestedSet(expectedFirst, first);
              });
          context.deserialize(
              codedIn,
              value,
              (container, untypedSecond) -> {
                NotNestedSet second = (NotNestedSet) untypedSecond;
                container.second = second;
                verifyDeserializedNotNestedSet(expectedSecond, second);
              });
          break;
        case OFFSET_WITH_DONE_CALLBACK:
          context.deserialize(
              codedIn,
              value,
              NotNestedSetContainer.FIRST_OFFSET,
              () -> verifyDeserializedNotNestedSet(expectedFirst, value.first));
          context.deserialize(
              codedIn,
              value,
              NotNestedSetContainer.SECOND_OFFSET,
              () -> verifyDeserializedNotNestedSet(expectedSecond, value.second));
          break;
      }
      return value;
    }
  }

  @Test
  public void valueDependsOnFuture(
      @TestParameter DeserializeOverloadSelector overloadSelector,
      @TestParameter boolean doesSecondAliasFirst)
      throws Exception {
    // Exercises the case where AsyncDeserializationContext.deserialize overloads are called and the
    // result is a future. In the special case where `doesSecondAliasFirst` = true, `subject.second`
    // is a backreference to the first, which exercises the case where a future is added to the
    // memoization table.

    NotNestedSetContainer subject;
    if (doesSecondAliasFirst) {
      subject =
          new NotNestedSetContainer(
              NotNestedSet.createRandom(rng, 4, 4, Random::nextInt),
              NotNestedSet.createRandom(rng, 4, 4, Random::nextInt));
    } else {
      NotNestedSet contained = NotNestedSet.createRandom(rng, 5, 5, Random::nextInt);
      subject = new NotNestedSetContainer(contained, contained);
    }
    new SerializationTester(subject)
        .addCodec(getDefaultNotNestedSetCodec())
        .addCodec(new NotNestedSetContainerCodec(overloadSelector, subject.first, subject.second))
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(
            SharedValueDeserializationContextTest::verifyDeserializedNotNestedSetContainer)
        .runTests();
  }

  @Test
  public void internedValueWithSharedElement() throws Exception {
    new SerializationTester(InternedValue.create(101), InternedValue.create(45678))
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .runTests();
  }

  private static class InternedValue {
    private Integer value;

    private static InternedValue create(int value) {
      InternedValue result = new InternedValue();
      result.value = value;
      return result;
    }

    @Override
    public int hashCode() {
      return value;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof InternedValue that) {
        return Objects.equals(value, that.value);
      }
      return false;
    }
  }

  @Keep
  private static class InternedValueCodec extends InterningObjectCodec<InternedValue> {
    @Override
    public Class<InternedValue> getEncodedClass() {
      return InternedValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, InternedValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.value, /* distinguisher= */ null, DeferredIntegerCodec.INSTANCE, codedOut);
    }

    @Override
    public InternedValue deserializeInterned(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      InternedValue value = new InternedValue();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredIntegerCodec.INSTANCE,
          value,
          (parent, v) -> parent.value = (Integer) v);
      return value;
    }

    @Override
    @SuppressWarnings("CanIgnoreReturnValueSuggester") // fake implementation just returns input
    public InternedValue intern(InternedValue interned) {
      checkNotNull(interned.value);
      return interned;
    }
  }

  private static class DeferredIntegerCodec extends DeferredObjectCodec<Integer> {
    private static final DeferredIntegerCodec INSTANCE = new DeferredIntegerCodec();

    @Override
    public Class<Integer> getEncodedClass() {
      return Integer.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(SerializationContext context, Integer obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj);
    }

    @Override
    public DeferredValue<Integer> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int value = codedIn.readInt32();
      return () -> value;
    }
  }

  private ListenableFuture<Object> deserializeWithExecutor(
      ObjectCodecs codecs, FingerprintValueService fingerprintValueService, ByteString data) {
    var task =
        ListenableFutureTask.create(
            () -> codecs.deserializeMemoizedAndBlocking(fingerprintValueService, data));
    executor.execute(task);
    return task;
  }

  private static void verifyDeserializedNotNestedSet(
      NotNestedSet original, NotNestedSet deserialized) {
    assertThat(dumpStructureWithEquivalenceReduction(deserialized))
        .isEqualTo(dumpStructureWithEquivalenceReduction(original));
  }

  private static void verifyDeserializedNotNestedSetContainer(
      NotNestedSetContainer original, NotNestedSetContainer deserialized) {
    verifyDeserializedNotNestedSet(original.first, deserialized.first);
    verifyDeserializedNotNestedSet(original.second, deserialized.second);
  }

  private static ObjectCodecs createObjectCodecs() {
    return new ObjectCodecs(
        AutoRegistry.get().getBuilder().add(getDefaultNotNestedSetCodec()).build());
  }

  private static ObjectCodec<NotNestedSet> getDefaultNotNestedSetCodec() {
    return new NotNestedSetCodec(new NestedArrayCodec());
  }
}
