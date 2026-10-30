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
import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.createRandomLeafArray;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;
import static com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableFutureTask;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NestedArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetDeferredCodec;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.MissingSharedValueBytesException;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
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
    var compressionService = new CompressionServiceImpl();
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
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
    ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      // If it is asynchronous, writing should complete without throwing any exceptions.
      writeStatus.get();
    }

    ListenableFuture<Object> result =
        deserializeWithExecutor(
            codecs, compressionService, fingerprintValueService, serialized.getObject());

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

  private static NotNestedSet createDeterministicNotNestedSet() {
    byte[] bytes = new byte[1000];
    new Random(42).nextBytes(bytes);
    return new NotNestedSet(new Object[] {bytes});
  }

  @Test
  public void chunkedSharedValue_roundTrips() throws Exception {
    int testChunkSize = 600;
    ChunkedValueSerialization.setChunkSizeForTesting(testChunkSize);
    try {
      var compressionService = new CompressionServiceImpl();
      var readCount = new AtomicInteger();
      var store =
          new InMemoryFingerprintValueStore() {
            @Override
            public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) {
              readCount.incrementAndGet();
              return super.get(fingerprint);
            }
          };
      FingerprintValueService fingerprintValueService =
          FingerprintValueService.createForTesting(store);
      ObjectCodecs codecs = createObjectCodecs();

      NotNestedSet subject = createDeterministicNotNestedSet();

      SerializationResult<ByteString> serialized =
          codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
      ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
      if (writeStatus != null) {
        writeStatus.get();
      }

      // Verify that there are exactly 3 entries: 1 reference blob + 2 chunks.
      assertThat(store.fingerprintToContents).hasSize(3);

      ListenableFuture<Object> result =
          deserializeWithExecutor(
              codecs, compressionService, fingerprintValueService, serialized.getObject());

      verifyDeserializedNotNestedSet(subject, (NotNestedSet) result.get());
      // Deserialization reads the 1 reference blob plus exactly 2 chunks.
      assertThat(readCount.get()).isEqualTo(3);
    } finally {
      ChunkedValueSerialization.resetChunkSizeForTesting();
    }
  }

  @Test
  public void chunkedSharedValue_missingChunk_reportedAsMissing() throws Exception {
    int testChunkSize = 600;
    ChunkedValueSerialization.setChunkSizeForTesting(testChunkSize);
    try {
      var compressionService = new CompressionServiceImpl();
      var store = new InMemoryFingerprintValueStore(/* useNullForMissingValues= */ true);
      FingerprintValueService fingerprintValueService =
          FingerprintValueService.createForTesting(store);
      ObjectCodecs codecs = createObjectCodecs();

      NotNestedSet subject = createDeterministicNotNestedSet();

      SerializationResult<ByteString> serialized =
          codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
      ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
      if (writeStatus != null) {
        writeStatus.get();
      }

      assertThat(store.fingerprintToContents).hasSize(3);

      // Find the chunk fingerprints from the stored reference blob.
      ByteString refBlob = null;
      for (ByteString content : store.fingerprintToContents.values()) {
        if (ChunkedValueSerialization.isChunked(content.toByteArray())) {
          refBlob = content;
          break;
        }
      }
      assertThat(refBlob).isNotNull();
      ImmutableList<PackedFingerprint> chunkFingerprints =
          ChunkedValueSerialization.parseChunkFingerprints(refBlob.toByteArray());
      assertThat(chunkFingerprints).hasSize(2);

      // Remove one chunk from the store to simulate a missing chunk.
      PackedFingerprint missingChunk = chunkFingerprints.get(0);
      store.fingerprintToContents.remove(ByteString.copyFrom(missingChunk.toBytes()));

      ListenableFuture<Object> result =
          deserializeWithExecutor(
              codecs, compressionService, fingerprintValueService, serialized.getObject());

      var thrown =
          (MissingSharedValueBytesException)
              assertThrows(ExecutionException.class, result::get).getCause();
      assertThat(thrown).hasMessageThat().isEqualTo("Missing shared value bytes");
    } finally {
      ChunkedValueSerialization.resetChunkSizeForTesting();
    }
  }

  private static class NotNestedSetContainer {
    private NotNestedSet first;
    private NotNestedSet second;

    private NotNestedSetContainer() {}

    private NotNestedSetContainer(NotNestedSet first, NotNestedSet second) {
      this.first = first;
      this.second = second;
    }
  }

  /** Selects the {@link AsyncDeserializationContext#deserialize} overload. */
  private enum DeserializeOverloadSelector {
    SETTER,
    ARRAY_ELEMENT_WITH_DONE_CALLBACK
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
        case ARRAY_ELEMENT_WITH_DONE_CALLBACK:
          {
            // Buffers the elements and populates the container only once all callbacks have run.
            // If any callback were skipped, the container would remain unpopulated and the
            // SerializationTester's verification would fail.
            Object[] elements = new Object[2];
            AtomicInteger remaining = new AtomicInteger(2);
            Runnable done =
                () -> {
                  if (remaining.decrementAndGet() == 0) {
                    value.first = (NotNestedSet) elements[0];
                    value.second = (NotNestedSet) elements[1];
                    verifyDeserializedNotNestedSet(expectedFirst, value.first);
                    verifyDeserializedNotNestedSet(expectedSecond, value.second);
                  }
                };
            context.deserializeArrayElement(codedIn, elements, 0, done);
            context.deserializeArrayElement(codedIn, elements, 1, done);
            break;
          }
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

  @Test
  public void
      distinctClassesWithIdenticalSerializedBytes_haveDistinctFingerprintsAndDeserializeCorrectly()
          throws Exception {
    var compressionService = new CompressionServiceImpl();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(FingerprintValueCache.SyncMode.LINKED);
    ObjectCodecs codecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .add(new ContainerACodec())
                .add(new ContainerBCodec())
                .build());

    ContainerA first = new ContainerA(new SharedElement(42));
    ContainerB second = new ContainerB(new OtherSharedElement(42));

    // Both SharedElement(42) and OtherSharedElement(42) serialize to identical bytes (int32 42).
    SerializationResult<ByteString> serializedFirst =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, first);
    if (serializedFirst.getFutureToBlockWritesOn() != null) {
      serializedFirst.getFutureToBlockWritesOn().get();
    }

    SerializationResult<ByteString> serializedSecond =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, second);
    if (serializedSecond.getFutureToBlockWritesOn() != null) {
      serializedSecond.getFutureToBlockWritesOn().get();
    }

    // Because the codecs have different class names, the resulting fingerprints must differ.
    assertThat(serializedFirst.getObject()).isNotEqualTo(serializedSecond.getObject());

    // Deserializing `second` retrieves from the cache or deserializes without ClassCastException.
    ListenableFuture<Object> deserializedSecond =
        deserializeWithExecutor(
            codecs, compressionService, fingerprintValueService, serializedSecond.getObject());
    assertThat(((ContainerB) deserializedSecond.get()).element.value).isEqualTo(42);

    ListenableFuture<Object> deserializedFirst =
        deserializeWithExecutor(
            codecs, compressionService, fingerprintValueService, serializedFirst.getObject());
    assertThat(((ContainerA) deserializedFirst.get()).element.value).isEqualTo(42);
  }

  @Test
  public void missingSharedValueData_producesSpecificError() throws Exception {
    GetRecordingStore store = new GetRecordingStore();
    var compressionService = new CompressionServiceImpl();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(store);
    ObjectCodecs codecs = createObjectCodecs();

    // This subject results in exactly one shared value, storing the leaf array.
    var subject = new NotNestedSet(new Object[] {createRandomLeafArray(rng, Random::nextInt)});

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
    ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      // If it is asynchronous, writing should complete without throwing any exceptions.
      writeStatus.get();
    }

    ListenableFuture<Object> result =
        deserializeWithExecutor(
            codecs, compressionService, fingerprintValueService, serialized.getObject());

    // Completes the request for shared value bytes with null bytes, indicating missing data.
    store.takeFirstRequest().completeWithNullBytes();

    var thrown =
        (MissingSharedValueBytesException)
            assertThrows(ExecutionException.class, result::get).getCause();
    assertThat(thrown).hasMessageThat().isEqualTo("Missing shared value bytes");
  }

  @Test
  public void getSharedValue_missingBytesFromCache_notifiesLookupCollector() throws Exception {
    GetRecordingStore store = new GetRecordingStore();
    var compressionService = new CompressionServiceImpl();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(store);
    ObjectCodecs codecs = createObjectCodecs();

    // This subject results in exactly one shared value, storing the leaf array.
    var subject = new NotNestedSet(new Object[] {createRandomLeafArray(rng, Random::nextInt)});

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
    ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      // If the write is asynchronous, writing should complete without throwing any exceptions.
      writeStatus.get();
    }

    @SuppressWarnings("unchecked")
    var result =
        (ListenableFuture<SkyframeLookupContinuation>)
            SharedValueDeserializationContext.deserializeWithSkyframe(
                codecs.getCodecRegistry(),
                ImmutableClassToInstanceMap.of(),
                compressionService,
                fingerprintValueService,
                serialized.getObject().newCodedInput());

    // Completes the request for shared value bytes with null bytes, indicating missing data.
    store.takeFirstRequest().completeWithNullBytes();

    // The following get call hangs if the missing bytes are not propagated.
    var thrown =
        (MissingSharedValueBytesException)
            assertThrows(ExecutionException.class, () -> result.get(WAIT_TIMEOUT_SECONDS, SECONDS))
                .getCause();
    assertThat(thrown).hasMessageThat().isEqualTo("Missing shared value bytes");
  }

  private enum FailureKind {
    SERIALIZATION_EXCEPTION {
      @Override
      Throwable create() {
        return new SerializationException("boom");
      }
    },
    RUNTIME_EXCEPTION {
      @Override
      Throwable create() {
        return new IllegalStateException("boom");
      }
    },
    ERROR {
      @Override
      Throwable create() {
        return new AssertionError("boom");
      }
    };

    abstract Throwable create();
  }

  private enum PostFetchFailureStage {
    DEFERRED_CALL,
    SETTER
  }

  @Test
  public void
      getSharedValue_failureAfterNotifyFetchDone_failsOperationWithoutDoubleDecrementingCollector(
          @TestParameter FailureKind failureKind,
          @TestParameter PostFetchFailureStage stage,
          @TestParameter boolean nested)
          throws Exception {
    GetRecordingStore store = new GetRecordingStore();
    var compressionService = new CompressionServiceImpl();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(store);

    Throwable failure = failureKind.create();
    ObjectCodecs codecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .add(new ThrowingContainerCodec(stage, failure))
                .build());

    Object[] firstContents =
        nested
            ? new Object[] {createRandomLeafArray(rng, Random::nextInt)}
            : createRandomLeafArray(rng, Random::nextInt);
    Object[] secondContents = createRandomLeafArray(rng, Random::nextInt);
    var subject =
        new NotNestedSetContainer(
            new NotNestedSet(firstContents), new NotNestedSet(secondContents));

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
    ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      writeStatus.get();
    }

    @SuppressWarnings("unchecked")
    var continuationFuture =
        (ListenableFuture<SkyframeLookupContinuation>)
            SharedValueDeserializationContext.deserializeWithSkyframe(
                codecs.getCodecRegistry(),
                ImmutableClassToInstanceMap.of(),
                compressionService,
                fingerprintValueService,
                serialized.getObject().newCodedInput());

    GetRequest firstRequest = store.takeFirstRequest();
    GetRequest secondRequest = store.takeFirstRequest();

    // Complete first's request (and its nested child request if nested) while second's request is
    // still pending. The failure thrown after notifyFetchDone() must not call onFailure() and
    // double-decrement SkyframeLookupCollector.
    firstRequest.complete();
    if (nested) {
      assertThat(continuationFuture.isDone()).isFalse();
      store.takeFirstRequest().complete();
    }
    assertThat(continuationFuture.isDone()).isFalse();

    // Completing second's request finishes the lookup collector; the resulting deserialization
    // future fails with the exact exception rather than hanging.
    secondRequest.complete();
    SkyframeLookupContinuation continuation = continuationFuture.get(WAIT_TIMEOUT_SECONDS, SECONDS);
    ListenableFuture<?> resultFuture = continuation.process(/* env= */ null);
    Throwable thrown =
        assertThrows(
                ExecutionException.class, () -> resultFuture.get(WAIT_TIMEOUT_SECONDS, SECONDS))
            .getCause();
    assertThat(thrown).isSameInstanceAs(failure);
  }

  @Test
  public void sharedValueIsDecompressed(@TestParameter boolean compress) throws Exception {
    GetRecordingStore store = new GetRecordingStore();
    var compressionService = new CompressionServiceImpl();
    FingerprintValueService fingerprintValueService =
        FingerprintValueService.createForTesting(store);
    ObjectCodecs codecs = createObjectCodecs();
    byte[] bytes = new byte[compress ? 2000 : 1000];
    NotNestedSet subject =
        new NotNestedSet(
            new Object[] {
              bytes,
            });

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(compressionService, fingerprintValueService, subject);
    ListenableFuture<?> writeStatus = serialized.getFutureToBlockWritesOn();
    if (writeStatus != null) {
      writeStatus.get();
    }

    ListenableFuture<Object> result =
        deserializeWithExecutor(
            codecs, compressionService, fingerprintValueService, serialized.getObject());

    ImmutableList<byte[]> storeValues =
        ImmutableList.copyOf(store.getFingerprintToContents().values());
    assertThat(storeValues).hasSize(1);
    assertThat(storeValues.get(0)).hasLength(compress ? 24 : 1007);

    store.takeFirstRequest().complete();
    verifyDeserializedNotNestedSet(subject, (NotNestedSet) result.get());
  }

  /**
   * A substitute for {@link Integer} but declared <b>not</b> to be a <a
   * href="https://openjdk.org/jeps/401">value class</a> (as {@link Integer} is becoming).
   */
  private static class SharedElement {
    private final int value;

    private SharedElement(int value) {
      this.value = value;
    }

    @Override
    public int hashCode() {
      return value;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof SharedElement that) {
        return this.value == that.value;
      }
      return false;
    }
  }

  private static class InternedValue {
    private SharedElement value;

    private static InternedValue create(int value) {
      InternedValue result = new InternedValue();
      result.value = new SharedElement(value);
      return result;
    }

    @Override
    public int hashCode() {
      return value.hashCode();
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
          obj.value, /* distinguisher= */ null, DeferredSharedElementCodec.INSTANCE, codedOut);
    }

    @Override
    public InternedValue deserializeInterned(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      InternedValue value = new InternedValue();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredSharedElementCodec.INSTANCE,
          value,
          (parent, v) -> parent.value = (SharedElement) v);
      return value;
    }

    @Override
    @SuppressWarnings("CanIgnoreReturnValueSuggester") // fake implementation just returns input
    public InternedValue intern(InternedValue interned) {
      checkNotNull(interned.value);
      return interned;
    }
  }

  private static class DeferredSharedElementCodec extends DeferredObjectCodec<SharedElement> {
    private static final DeferredSharedElementCodec INSTANCE = new DeferredSharedElementCodec();

    @Override
    public Class<SharedElement> getEncodedClass() {
      return SharedElement.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(
        SerializationContext context, SharedElement obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj.value);
    }

    @Override
    public DeferredValue<SharedElement> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int value = codedIn.readInt32();
      return () -> new SharedElement(value);
    }
  }

  private static class OtherSharedElement {
    private final int value;

    private OtherSharedElement(int value) {
      this.value = value;
    }
  }

  private static class DeferredOtherSharedElementCodec
      extends DeferredObjectCodec<OtherSharedElement> {
    private static final DeferredOtherSharedElementCodec INSTANCE =
        new DeferredOtherSharedElementCodec();

    @Override
    public Class<OtherSharedElement> getEncodedClass() {
      return OtherSharedElement.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(
        SerializationContext context, OtherSharedElement obj, CodedOutputStream codedOut)
        throws IOException {
      codedOut.writeInt32NoTag(obj.value);
    }

    @Override
    public DeferredValue<OtherSharedElement> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn) throws IOException {
      int value = codedIn.readInt32();
      return () -> new OtherSharedElement(value);
    }
  }

  private static class ContainerA {
    private SharedElement element;

    private ContainerA(SharedElement element) {
      this.element = element;
    }
  }

  private static class ContainerACodec extends DeferredObjectCodec<ContainerA> {
    @Override
    public Class<ContainerA> getEncodedClass() {
      return ContainerA.class;
    }

    @Override
    public void serialize(SerializationContext context, ContainerA obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.element, /* distinguisher= */ null, DeferredSharedElementCodec.INSTANCE, codedOut);
    }

    @Override
    public DeferredValue<ContainerA> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ContainerA container = new ContainerA(null);
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredSharedElementCodec.INSTANCE,
          container,
          (parent, v) -> parent.element = (SharedElement) v);
      return () -> container;
    }
  }

  private static class ContainerB {
    private OtherSharedElement element;

    private ContainerB(OtherSharedElement element) {
      this.element = element;
    }
  }

  private static class ContainerBCodec extends DeferredObjectCodec<ContainerB> {
    @Override
    public Class<ContainerB> getEncodedClass() {
      return ContainerB.class;
    }

    @Override
    public void serialize(SerializationContext context, ContainerB obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.element,
          /* distinguisher= */ null,
          DeferredOtherSharedElementCodec.INSTANCE,
          codedOut);
    }

    @Override
    public DeferredValue<ContainerB> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ContainerB container = new ContainerB(null);
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredOtherSharedElementCodec.INSTANCE,
          container,
          (parent, v) -> parent.element = (OtherSharedElement) v);
      return () -> container;
    }
  }

  private ListenableFuture<Object> deserializeWithExecutor(
      ObjectCodecs codecs,
      CompressionService compressionService,
      FingerprintValueService fingerprintValueService,
      ByteString data) {
    var task =
        ListenableFutureTask.create(
            () ->
                codecs.deserializeMemoizedAndBlocking(
                    compressionService, fingerprintValueService, data));
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

  private static final class ThrowingContainerCodec
      extends DeferredObjectCodec<NotNestedSetContainer> {
    private final PostFetchFailureStage stage;
    private final Throwable failure;
    private final NestedArrayCodec nestedArrayCodec = new NestedArrayCodec();
    private final DeferredObjectCodec<Object[]> firstArrayCodec;

    private ThrowingContainerCodec(PostFetchFailureStage stage, Throwable failure) {
      this.stage = stage;
      this.failure = failure;
      this.firstArrayCodec =
          new DeferredObjectCodec<Object[]>() {
            @Override
            public boolean autoRegister() {
              return false;
            }

            @Override
            public Class<Object[]> getEncodedClass() {
              return Object[].class;
            }

            @Override
            public void serialize(
                SerializationContext context, Object[] obj, CodedOutputStream codedOut)
                throws SerializationException, IOException {
              nestedArrayCodec.serialize(context, obj, codedOut);
            }

            @Override
            public DeferredValue<Object[]> deserializeDeferred(
                AsyncDeserializationContext context, CodedInputStream codedIn)
                throws SerializationException, IOException {
              DeferredValue<Object[]> inner =
                  nestedArrayCodec.deserializeDeferred(context, codedIn);
              return () -> {
                if (stage == PostFetchFailureStage.DEFERRED_CALL) {
                  throwFailure(failure);
                }
                return inner.call();
              };
            }
          };
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<NotNestedSetContainer> getEncodedClass() {
      return NotNestedSetContainer.class;
    }

    @Override
    public void serialize(
        SerializationContext context, NotNestedSetContainer container, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          container.first.getContents(), /* distinguisher= */ null, firstArrayCodec, codedOut);
      context.putSharedValue(
          container.second.getContents(), /* distinguisher= */ null, nestedArrayCodec, codedOut);
    }

    @Override
    public DeferredValue<NotNestedSetContainer> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var container = new NotNestedSetContainer();
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          firstArrayCodec,
          container,
          (c, v) -> {
            if (stage == PostFetchFailureStage.SETTER) {
              throwFailure(failure);
            }
            c.first = new NotNestedSet((Object[]) v);
          });
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          nestedArrayCodec,
          container,
          (c, v) -> c.second = new NotNestedSet((Object[]) v));
      return () -> container;
    }

    private static void throwFailure(Throwable failure) throws SerializationException {
      throwIfInstanceOf(failure, SerializationException.class);
      throwIfInstanceOf(failure, RuntimeException.class);
      throwIfInstanceOf(failure, Error.class);
      throw new AssertionError(failure);
    }
  }
}
