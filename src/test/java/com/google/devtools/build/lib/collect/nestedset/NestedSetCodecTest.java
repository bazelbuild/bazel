// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.InMemoryNestedSetStorageEndpoint;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.MissingNestedSetException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.NestedSetCache;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.NestedSetStorageEndpoint;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.charset.Charset;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

/** Tests for {@link NestedSet} serialization. */
@RunWith(JUnit4.class)
public final class NestedSetCodecTest {

  @Test
  public void testAutoCodecedCodec() throws Exception {
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get().getBuilder().setAllowDefaultCodec(true).build(),
            ImmutableClassToInstanceMap.of());
    NestedSetCodecTestUtils.checkCodec(objectCodecs, false, false);
  }

  @Test
  public void testCodecWithInMemoryNestedSetStore() throws Exception {
    ObjectCodecs objectCodecs = createCodecs(NestedSetStore.inMemory());
    NestedSetCodecTestUtils.checkCodec(objectCodecs, true, true);
  }

  @Test
  public void onlyOneReadPerArray() throws Exception {
    NestedSet<String> base = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    NestedSet<String> top = NestedSetBuilder.fromNestedSet(base).add("c").build();

    AtomicInteger reads = new AtomicInteger();
    NestedSetStorageEndpoint endpoint =
        new NestedSetStorageEndpoint() {
          final InMemoryNestedSetStorageEndpoint delegate = new InMemoryNestedSetStorageEndpoint();

          @Override
          public ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
            return delegate.put(fingerprint, serializedBytes);
          }

          @Override
          public ListenableFuture<byte[]> get(ByteString fingerprint) {
            reads.incrementAndGet();
            return delegate.get(fingerprint);
          }
        };

    ObjectCodecs serializer = createCodecs(new NestedSetStore(endpoint));
    ByteString serializedBase = serializer.serializeMemoizedAndBlocking(base).getObject();
    ByteString serializedTop = serializer.serializeMemoizedAndBlocking(top).getObject();

    // When deserializing top, we should perform 2 reads, one for each array in [[a, b], c].
    ObjectCodecs deserializer = createCodecs(new NestedSetStore(endpoint));
    NestedSet<?> deserializedTop = (NestedSet<?>) deserializer.deserializeMemoized(serializedTop);
    assertThat(deserializedTop.toList()).containsExactly("a", "b", "c");
    assertThat(reads.get()).isEqualTo(2);

    // When deserializing base, we should not need to perform any additional reads since we have
    // already read [a, b] and it is still in memory.
    GcFinalization.awaitFullGc();
    NestedSet<?> deserializedBase = (NestedSet<?>) deserializer.deserializeMemoized(serializedBase);
    assertThat(deserializedBase.toList()).containsExactly("a", "b");
    assertThat(reads.get()).isEqualTo(2);
  }

  @Test
  public void missingNestedSetException_hiddenUntilNestedSetIsConsumed() throws Exception {
    NestedSetStorageEndpoint storageEndpoint =
        new NestedSetStorageEndpoint() {
          @Override
          public ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
            return immediateVoidFuture();
          }

          @Override
          public ListenableFuture<byte[]> get(ByteString fingerprint) {
            return immediateFailedFuture(
                new MissingNestedSetException(ByteString.copyFromUtf8("fingerprint")));
          }
        };
    ObjectCodecs serializer = createCodecs(new NestedSetStore(storageEndpoint));
    ObjectCodecs deserializer = createCodecs(new NestedSetStore(storageEndpoint));

    NestedSet<?> serialized = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    SerializationResult<ByteString> result = serializer.serializeMemoizedAndBlocking(serialized);
    Object deserialized = deserializer.deserializeMemoized(result.getObject());

    assertThat(deserialized).isInstanceOf(NestedSet.class);
    assertThrows(
        MissingNestedSetException.class, ((NestedSet<?>) deserialized)::toListInterruptibly);
  }

  @Test
  public void unexpectedException_hiddenUntilNestedSetIsConsumed() throws Exception {
    NestedSetStorageEndpoint storageEndpoint =
        new NestedSetStorageEndpoint() {
          @Override
          public ListenableFuture<Void> put(ByteString fingerprint, byte[] serializedBytes) {
            return immediateVoidFuture();
          }

          @Override
          public ListenableFuture<byte[]> get(ByteString fingerprint) {
            return immediateFailedFuture(new RuntimeException("Something went wrong"));
          }
        };
    ObjectCodecs serializer = createCodecs(new NestedSetStore(storageEndpoint));
    ObjectCodecs deserializer = createCodecs(new NestedSetStore(storageEndpoint));

    NestedSet<?> serialized = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    SerializationResult<ByteString> result = serializer.serializeMemoizedAndBlocking(serialized);
    Object deserialized = deserializer.deserializeMemoized(result.getObject());

    assertThat(deserialized).isInstanceOf(NestedSet.class);
    Exception e = assertThrows(RuntimeException.class, ((NestedSet<?>) deserialized)::toList);
    assertThat(e).hasMessageThat().contains("Something went wrong");
  }

  /**
   * Tests that serialization of a {@code NestedSet<NestedSet<String>>} waits on the writes of the
   * inner NestedSets.
   */
  @Test
  public void testNestedNestedSetSerialization() throws Exception {
    NestedSetStorageEndpoint mockStorage = mock(NestedSetStorageEndpoint.class);
    SettableFuture<Void> innerWrite = SettableFuture.create();
    SettableFuture<Void> outerWrite = SettableFuture.create();
    when(mockStorage.put(any(), any()))
        // The write of the inner NestedSet {"a", "b"}
        .thenReturn(innerWrite)
        // The write of the inner NestedSet {"c", "d"}
        .thenReturn(innerWrite)
        // The write of the outer NestedSet {{"a", "b"}, {"c", "d"}}
        .thenReturn(outerWrite);
    ObjectCodecs objectCodecs = createCodecs(new NestedSetStore(mockStorage));

    NestedSet<NestedSet<String>> nestedNestedSet =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b"),
            NestedSetBuilder.create(Order.STABLE_ORDER, "c", "d"));

    SerializationResult<ByteString> result =
        objectCodecs.serializeMemoizedAndBlocking(nestedNestedSet);
    outerWrite.set(null);
    assertThat(result.getFutureToBlockWritesOn().isDone()).isFalse();
    innerWrite.set(null);
    assertThat(result.getFutureToBlockWritesOn().isDone()).isTrue();
  }

  @Test
  public void testNestedNestedSetsWithCommonDependencyWaitOnSameInnerFuture() throws Exception {
    NestedSetStorageEndpoint mockStorage = mock(NestedSetStorageEndpoint.class);
    SettableFuture<Void> sharedInnerWrite = SettableFuture.create();
    SettableFuture<Void> outerWrite = SettableFuture.create();
    when(mockStorage.put(any(), any()))
        // The write of the shared inner NestedSet {"a", "b"}
        .thenReturn(sharedInnerWrite)
        // The write of the inner NestedSet {"c", "d"}
        .thenReturn(immediateVoidFuture())
        // The write of the outer NestedSet {{"a", "b"}, {"c", "d"}}
        .thenReturn(outerWrite)
        // The write of the inner NestedSet {"e", "f"}
        .thenReturn(immediateVoidFuture());
    ObjectCodecs objectCodecs = createCodecs(new NestedSetStore(mockStorage));

    NestedSet<String> sharedInnerNestedSet = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    NestedSet<NestedSet<String>> nestedNestedSet1 =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            sharedInnerNestedSet,
            NestedSetBuilder.create(Order.STABLE_ORDER, "c", "d"));
    NestedSet<NestedSet<String>> nestedNestedSet2 =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            sharedInnerNestedSet,
            NestedSetBuilder.create(Order.STABLE_ORDER, "e", "f"));

    SerializationResult<ByteString> result1 =
        objectCodecs.serializeMemoizedAndBlocking(nestedNestedSet1);
    SerializationResult<ByteString> result2 =
        objectCodecs.serializeMemoizedAndBlocking(nestedNestedSet2);
    outerWrite.set(null);
    assertThat(result1.getFutureToBlockWritesOn().isDone()).isFalse();
    assertThat(result2.getFutureToBlockWritesOn().isDone()).isFalse();
    sharedInnerWrite.set(null);
    assertThat(result1.getFutureToBlockWritesOn().isDone()).isTrue();
    assertThat(result2.getFutureToBlockWritesOn().isDone()).isTrue();
  }

  @Test
  public void testSingletonNestedSetSerializedWithoutStore() throws Exception {
    NestedSetStore mockNestedSetStore = mock(NestedSetStore.class);
    when(mockNestedSetStore.computeFingerprintAndStore(any(), any()))
        .thenThrow(new AssertionError("NestedSetStore should not have been used"));

    ObjectCodecs objectCodecs = createCodecs(mockNestedSetStore);
    NestedSet<String> singletonNestedSet =
        new NestedSetBuilder<String>(Order.STABLE_ORDER).add("a").build();
    objectCodecs.serialize(singletonNestedSet);
  }

  @Test
  public void cacheEntryHasLifetimeOfContents() {
    NestedSetCache cache = new NestedSetCache();
    Object[] contents = new Object[0];
    ByteString fingerprint = ByteString.copyFrom(new byte[2]);
    cache.put(
        NestedSetStore.FingerprintComputationResult.create(fingerprint, immediateVoidFuture()),
        contents);
    GcFinalization.awaitFullGc();
    assertThat(cache.putIfAbsent(fingerprint, immediateFuture(null))).isEqualTo(contents);
    WeakReference<Object[]> weakRef = new WeakReference<>(contents);
    contents = null;
    fingerprint = null;
    GcFinalization.awaitClear(weakRef);
  }

  @Test
  public void testDeserializationInParallel() throws Exception {
    NestedSetStorageEndpoint nestedSetStorageEndpoint =
        Mockito.spy(new InMemoryNestedSetStorageEndpoint());
    NestedSetCache emptyNestedSetCache = mock(NestedSetCache.class);
    NestedSetStore nestedSetStore =
        new NestedSetStore(
            nestedSetStorageEndpoint, emptyNestedSetCache, MoreExecutors.directExecutor());

    ObjectCodecs objectCodecs = createCodecs(nestedSetStore);

    NestedSet<String> subset1 =
        new NestedSetBuilder<String>(Order.STABLE_ORDER).add("a").add("b").build();
    SettableFuture<byte[]> subset1Future = SettableFuture.create();
    NestedSet<String> subset2 =
        new NestedSetBuilder<String>(Order.STABLE_ORDER).add("c").add("d").build();
    SettableFuture<byte[]> subset2Future = SettableFuture.create();
    NestedSet<String> set =
        new NestedSetBuilder<String>(Order.STABLE_ORDER)
            .addTransitive(subset1)
            .addTransitive(subset2)
            .build();

    // We capture the arguments to #put() during serialization, so as to correctly mock results for
    // #get()
    ArgumentCaptor<ByteString> fingerprintCaptor = ArgumentCaptor.forClass(ByteString.class);
    ByteString fingerprint =
        nestedSetStore
            .computeFingerprintAndStore(
                (Object[]) set.getChildren(), objectCodecs.getSerializationContext())
            .fingerprint();
    Mockito.verify(nestedSetStorageEndpoint, Mockito.times(3))
        .put(fingerprintCaptor.capture(), any());
    Mockito.doReturn(subset1Future)
        .when(nestedSetStorageEndpoint)
        .get(fingerprintCaptor.getAllValues().get(0));
    Mockito.doReturn(subset2Future)
        .when(nestedSetStorageEndpoint)
        .get(fingerprintCaptor.getAllValues().get(1));
    when(emptyNestedSetCache.putIfAbsent(any(), any())).thenAnswer(invocation -> null);

    @SuppressWarnings("unchecked")
    ListenableFuture<Object[]> deserializationFuture =
        (ListenableFuture<Object[]>)
            nestedSetStore.getContentsAndDeserialize(
                fingerprint, objectCodecs.getDeserializationContext());
    // At this point, we expect deserializationFuture to be waiting on both of the underlying
    // fetches, which should have both been started.
    assertThat(deserializationFuture.isDone()).isFalse();
    Mockito.verify(nestedSetStorageEndpoint, Mockito.times(3)).get(any());

    // Once the underlying fetches complete, we expect deserialization to complete.
    subset1Future.set(ByteString.copyFrom("mock bytes", Charset.defaultCharset()).toByteArray());
    subset2Future.set(ByteString.copyFrom("mock bytes", Charset.defaultCharset()).toByteArray());
    assertThat(deserializationFuture.isDone()).isTrue();
  }

  @Test
  public void racingDeserialization() throws Exception {
    NestedSetStorageEndpoint nestedSetStorageEndpoint = mock(NestedSetStorageEndpoint.class);
    NestedSetCache nestedSetCache = Mockito.spy(new NestedSetCache());
    NestedSetStore nestedSetStore =
        new NestedSetStore(
            nestedSetStorageEndpoint, nestedSetCache, MoreExecutors.directExecutor());
    DeserializationContext deserializationContext = mock(DeserializationContext.class);
    ByteString fingerprint = ByteString.copyFromUtf8("fingerprint");
    // Future never completes, so we don't have to exercise that code in NestedSetStore.
    SettableFuture<byte[]> storageFuture = SettableFuture.create();
    when(nestedSetStorageEndpoint.get(fingerprint)).thenReturn(storageFuture);
    CountDownLatch fingerprintRequested = new CountDownLatch(2);
    Mockito.doAnswer(
            invocation -> {
              fingerprintRequested.countDown();
              @SuppressWarnings("unchecked")
              ListenableFuture<Object[]> result =
                  (ListenableFuture<Object[]>) invocation.callRealMethod();
              fingerprintRequested.await();
              return result;
            })
        .when(nestedSetCache)
        .putIfAbsent(Mockito.eq(fingerprint), any());
    AtomicReference<ListenableFuture<Object[]>> asyncResult = new AtomicReference<>();
    Thread asyncThread =
        new Thread(
            () -> {
              try {
                @SuppressWarnings("unchecked")
                ListenableFuture<Object[]> asyncContents =
                    (ListenableFuture<Object[]>)
                        nestedSetStore.getContentsAndDeserialize(
                            fingerprint, deserializationContext);
                asyncResult.set(asyncContents);
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
            });
    asyncThread.start();
    @SuppressWarnings("unchecked")
    ListenableFuture<Object[]> result =
        (ListenableFuture<Object[]>)
            nestedSetStore.getContentsAndDeserialize(fingerprint, deserializationContext);
    asyncThread.join();
    Mockito.verify(nestedSetStorageEndpoint, times(1)).get(Mockito.eq(fingerprint));
    assertThat(result).isSameInstanceAs(asyncResult.get());
    assertThat(result.isDone()).isFalse();
  }

  @Test
  public void bugInRacingSerialization() throws Exception {
    NestedSetStorageEndpoint nestedSetStorageEndpoint = mock(NestedSetStorageEndpoint.class);
    NestedSetCache nestedSetCache = Mockito.spy(new NestedSetCache());
    NestedSetStore nestedSetStore =
        new NestedSetStore(
            nestedSetStorageEndpoint, nestedSetCache, MoreExecutors.directExecutor());
    SerializationContext serializationContext = mock(SerializationContext.class);
    Object[] contents = {new Object()};
    when(serializationContext.getNewMemoizingContext()).thenReturn(serializationContext);
    when(nestedSetStorageEndpoint.put(any(), any()))
        .thenAnswer(invocation -> SettableFuture.create());
    CountDownLatch fingerprintRequested = new CountDownLatch(2);
    Mockito.doAnswer(
            invocation -> {
              fingerprintRequested.countDown();
              NestedSetStore.FingerprintComputationResult result =
                  (NestedSetStore.FingerprintComputationResult) invocation.callRealMethod();
              assertThat(result).isNull();
              fingerprintRequested.await();
              return null;
            })
        .when(nestedSetCache)
        .fingerprintForContents(contents);
    AtomicReference<NestedSetStore.FingerprintComputationResult> asyncResult =
        new AtomicReference<>();
    Thread asyncThread =
        new Thread(
            () -> {
              try {
                asyncResult.set(
                    nestedSetStore.computeFingerprintAndStore(contents, serializationContext));
              } catch (IOException | SerializationException e) {
                throw new IllegalStateException(e);
              }
            });
    asyncThread.start();
    NestedSetStore.FingerprintComputationResult result =
        nestedSetStore.computeFingerprintAndStore(contents, serializationContext);
    asyncThread.join();
    // TODO(janakr): This should be one fetch, but we currently do two.
    Mockito.verify(nestedSetStorageEndpoint, times(2)).put(any(), any());
    // TODO(janakr): These should be the same element.
    assertThat(result).isNotEqualTo(asyncResult.get());
  }

  @Test
  public void writeFuturesWaitForTransitiveWrites() throws Exception {
    NestedSetStorageEndpoint mockWriter = mock(NestedSetStorageEndpoint.class);
    NestedSetStore store = new NestedSetStore(mockWriter);
    SerializationContext mockSerializationContext = mock(SerializationContext.class);
    when(mockSerializationContext.getNewMemoizingContext()).thenReturn(mockSerializationContext);

    SettableFuture<Void> bottomReadFuture = SettableFuture.create();
    SettableFuture<Void> middleReadFuture = SettableFuture.create();
    SettableFuture<Void> topReadFuture = SettableFuture.create();
    when(mockWriter.put(any(), any()))
        .thenReturn(bottomReadFuture, middleReadFuture, topReadFuture);

    NestedSet<String> bottom =
        NestedSetBuilder.<String>stableOrder().add("bottom1").add("bottom2").build();
    NestedSet<String> middle =
        NestedSetBuilder.<String>stableOrder()
            .add("middle1")
            .add("middle2")
            .addTransitive(bottom)
            .build();
    NestedSet<String> top =
        NestedSetBuilder.<String>stableOrder()
            .add("top1")
            .add("top2")
            .addTransitive(middle)
            .build();

    ListenableFuture<Void> bottomWriteFuture =
        NestedSetCodecTestUtils.writeToStoreFuture(store, bottom, mockSerializationContext);
    ListenableFuture<Void> middleWriteFuture =
        NestedSetCodecTestUtils.writeToStoreFuture(store, middle, mockSerializationContext);
    ListenableFuture<Void> topWriteFuture =
        NestedSetCodecTestUtils.writeToStoreFuture(store, top, mockSerializationContext);
    assertThat(bottomWriteFuture.isDone()).isFalse();
    assertThat(middleWriteFuture.isDone()).isFalse();
    assertThat(topWriteFuture.isDone()).isFalse();

    topReadFuture.set(null);
    middleReadFuture.set(null);
    assertThat(bottomWriteFuture.isDone()).isFalse();
    assertThat(middleWriteFuture.isDone()).isFalse();
    assertThat(topWriteFuture.isDone()).isFalse();

    bottomReadFuture.set(null);
    assertThat(bottomWriteFuture.isDone()).isTrue();
    assertThat(middleWriteFuture.isDone()).isTrue();
    assertThat(topWriteFuture.isDone()).isTrue();
  }

  private static ObjectCodecs createCodecs(NestedSetStore store) {
    return new ObjectCodecs(
        AutoRegistry.get()
            .getBuilder()
            .setAllowDefaultCodec(true)
            .add(new NestedSetCodecWithStore(store))
            .build(),
        /*dependencies=*/ ImmutableClassToInstanceMap.of());
  }
}
