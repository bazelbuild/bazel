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

import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.NestedSetStorageEndpoint;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.protobuf.ByteString;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link NestedSet} serialization. */
@RunWith(JUnit4.class)
public class NestedSetCodecTest {
  @Before
  public void setUp() {
    SerializationConstants.shouldSerializeNestedSet = true;
  }

  @After
  public void tearDown() {
    SerializationConstants.shouldSerializeNestedSet = false;
  }

  @Test
  public void testAutoCodecedCodec() throws Exception {
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get().getBuilder().setAllowDefaultCodec(true).build(), ImmutableMap.of());
    NestedSetCodecTestUtils.checkCodec(objectCodecs, false);
  }

  @Test
  public void testCodecWithInMemoryNestedSetStore() throws Exception {
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .setAllowDefaultCodec(true)
                .add(new NestedSetCodecWithStore<>(NestedSetStore.inMemory()))
                .build(),
            ImmutableMap.of());
    NestedSetCodecTestUtils.checkCodec(objectCodecs, true);
  }

  /**
   * Tests that serialization of a {@code NestedSet<NestedSet<String>>} waits on the writes of the
   * inner NestedSets.
   */
  @Test
  public void testNestedNestedSetSerialization() throws Exception {
    NestedSetStorageEndpoint mockStorage = Mockito.mock(NestedSetStorageEndpoint.class);
    SettableFuture<Void> innerWrite = SettableFuture.create();
    SettableFuture<Void> outerWrite = SettableFuture.create();
    Mockito.when(mockStorage.put(Mockito.any(), Mockito.any()))
        // The write of the inner NestedSet {"a", "b"}
        .thenReturn(innerWrite)
        // The write of the inner NestedSet {"c", "d"}
        .thenReturn(innerWrite)
        // The write of the outer NestedSet {{"a", "b"}, {"c", "d"}}
        .thenReturn(outerWrite);
    NestedSetStore nestedSetStore = new NestedSetStore(mockStorage);
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .setAllowDefaultCodec(true)
                .add(new NestedSetCodecWithStore<>(nestedSetStore))
                .build(),
            ImmutableMap.of());

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
    NestedSetStorageEndpoint mockStorage = Mockito.mock(NestedSetStorageEndpoint.class);
    SettableFuture<Void> sharedInnerWrite = SettableFuture.create();
    SettableFuture<Void> outerWrite = SettableFuture.create();
    Mockito.when(mockStorage.put(Mockito.any(), Mockito.any()))
        // The write of the shared inner NestedSet {"a", "b"}
        .thenReturn(sharedInnerWrite)
        // The write of the inner NestedSet {"c", "d"}
        .thenReturn(Futures.immediateFuture(null))
        // The write of the outer NestedSet {{"a", "b"}, {"c", "d"}}
        .thenReturn(outerWrite)
        // The write of the inner NestedSet {"e", "f"}
        .thenReturn(Futures.immediateFuture(null));
    NestedSetStore nestedSetStore = new NestedSetStore(mockStorage);
    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .setAllowDefaultCodec(true)
                .add(new NestedSetCodecWithStore<>(nestedSetStore))
                .build(),
            ImmutableMap.of());

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
    NestedSetStore mockNestedSetStore = Mockito.mock(NestedSetStore.class);
    Mockito.when(mockNestedSetStore.computeFingerprintAndStore(Mockito.any(), Mockito.any()))
        .thenThrow(new AssertionError("NestedSetStore should not have been used"));

    ObjectCodecs objectCodecs =
        new ObjectCodecs(
            AutoRegistry.get()
                .getBuilder()
                .setAllowDefaultCodec(true)
                .add(new NestedSetCodecWithStore<>(mockNestedSetStore))
                .build());
    NestedSet<String> singletonNestedSet = new NestedSet<>(Order.STABLE_ORDER, "a");
    objectCodecs.serialize(singletonNestedSet);
  }
}
