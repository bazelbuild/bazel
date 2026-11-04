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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.protobuf.ByteString;
import java.util.List;
import net.starlark.java.eval.ImmutableSingletonStarlarkList;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.RegularImmutableStarlarkList;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class StarlarkListCodecTest {
  private final ObjectCodecs standardCodecs = new ObjectCodecs(AutoRegistry.get());

  private StarlarkList<Object> createUnfrozenMutableList() {
    return StarlarkList.of(Mutability.create("unfrozen"), "alpha", "beta", "gamma");
  }

  @Test
  public void testSkylarkListTypes_roundtripOk() throws SerializationException {
    StarlarkList.SerializableListSupplier<Object> supplier =
        () -> ImmutableList.of("alpha", "beta", "gamma");
    StarlarkList<Object> emptyMutable = StarlarkList.newList(Mutability.create("empty"));
    emptyMutable.mutability().freeze();
    StarlarkList<Object> singletonMutable =
        StarlarkList.of(Mutability.create("singleton"), "alpha");
    singletonMutable.mutability().freeze();
    StarlarkList<Object> regularMutable = createUnfrozenMutableList();
    regularMutable.mutability().freeze(); // must freeze mutable lists before serializing

    ImmutableList<StarlarkList<Object>> emptyLists =
        ImmutableList.of(
            StarlarkList.empty(), emptyMutable, StarlarkList.lazyImmutable(ImmutableList::of));
    for (StarlarkList<Object> aList : emptyLists) {
      ByteString serialized = standardCodecs.serialize(aList);
      var deserialized = standardCodecs.deserialize(serialized);
      assertThat(deserialized).isSameInstanceAs(StarlarkList.empty());
    }

    ImmutableList<StarlarkList<Object>> singletonLists =
        ImmutableList.of(
            StarlarkList.of(null, "alpha"),
            singletonMutable,
            StarlarkList.lazyImmutable(() -> ImmutableList.of("alpha")));
    for (StarlarkList<Object> aList : singletonLists) {
      ByteString serialized = standardCodecs.serialize(aList);
      var deserialized = standardCodecs.deserialize(serialized);
      assertThat(deserialized).isInstanceOf(ImmutableSingletonStarlarkList.class);
      assertThat(deserialized).isEqualTo(aList);
    }

    ImmutableList<StarlarkList<Object>> regularLists =
        ImmutableList.of(
            regularMutable,
            StarlarkList.of(null, "alpha", "beta", "gamma"),
            StarlarkList.lazyImmutable(supplier));
    for (StarlarkList<Object> aList : regularLists) {
      ByteString serialized = standardCodecs.serialize(aList);
      var deserialized = standardCodecs.deserialize(serialized);
      assertThat(deserialized).isInstanceOf(RegularImmutableStarlarkList.class);
      assertThat(deserialized).isEqualTo(aList);
    }
  }

  @Test
  public void emptyLists_serializeToSameBytes() throws Exception {
    StarlarkList<Object> emptyMutable = StarlarkList.newList(Mutability.create("empty"));
    emptyMutable.mutability().freeze();
    ByteString canonicalBytes = standardCodecs.serialize(StarlarkList.empty());
    assertThat(standardCodecs.serialize(emptyMutable)).isEqualTo(canonicalBytes);
    assertThat(standardCodecs.serialize(StarlarkList.lazyImmutable(ImmutableList::of)))
        .isEqualTo(canonicalBytes);
  }

  @Test
  public void testUnfrozenMutableList_serializeThrows() {
    StarlarkList<Object> unfrozen = createUnfrozenMutableList();
    SerializationException e =
        assertThrows(SerializationException.class, () -> standardCodecs.serialize(unfrozen));
    assertThat(e).hasMessageThat().contains("Trying to serialize unfrozen StarlarkList");
  }

  @Test
  public void listWithSpareCapacity_serializesOnlyActiveElements() throws Exception {
    Mutability mu = Mutability.create("spare");
    StarlarkList<Object> withSpareCapacity = StarlarkList.newList(mu);
    ImmutableList<Object> contents = ImmutableList.of("a", "b", "c", "d", "e");
    for (Object elem : contents) {
      withSpareCapacity.addElement(elem);
    }
    mu.freeze();
    // Precondition: the backing array is larger than the list.
    assertThat(withSpareCapacity.elems().length).isGreaterThan(withSpareCapacity.size());

    StarlarkList<Object> exactCapacity = StarlarkList.immutableCopyOf(contents);
    assertThat(exactCapacity.elems()).hasLength(exactCapacity.size());

    ByteString serialized = standardCodecs.serialize(withSpareCapacity);
    assertThat(serialized).isEqualTo(standardCodecs.serialize(exactCapacity));
    var deserialized = standardCodecs.deserialize(serialized);
    assertThat(deserialized).isInstanceOf(RegularImmutableStarlarkList.class);
    assertThat(deserialized).isEqualTo(withSpareCapacity);
  }

  @Test
  public void optimizingMemoryLayout_doesNotResultInDifferentSerializedBytes() throws Exception {
    Mutability foo = Mutability.create("foo");
    StarlarkList<StarlarkInt> wrapped =
        StarlarkList.copyOf(foo, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2)));
    foo.freeze();
    ByteString beforeByteString = standardCodecs.serialize(wrapped);
    StarlarkList<StarlarkInt> optimized = wrapped.unsafeOptimizeMemoryLayout();
    ByteString afterByteString = standardCodecs.serialize(wrapped);
    ByteString optimizedByteString = standardCodecs.serialize(optimized);
    assertThat(beforeByteString).isEqualTo(afterByteString);
    assertThat(beforeByteString).isEqualTo(optimizedByteString);
  }

  @Test
  public void lazyImmutableList_materializationAndOptimizationProduceSameSerializedBytes()
      throws Exception {
    StarlarkList<StarlarkInt> lazyList =
        StarlarkList.lazyImmutable(() -> ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2)));
    ByteString unmaterializedBytes = standardCodecs.serialize(lazyList);
    // Serialization materializes the list; unsafeOptimizeMemoryLayout() returns the canonical list.
    StarlarkList<StarlarkInt> optimized = lazyList.unsafeOptimizeMemoryLayout();
    ByteString materializedBytes = standardCodecs.serialize(lazyList);
    ByteString optimizedBytes = standardCodecs.serialize(optimized);
    assertThat(unmaterializedBytes).isEqualTo(materializedBytes);
    assertThat(unmaterializedBytes).isEqualTo(optimizedBytes);
  }

  @Test
  public void cyclicLists_roundtripOk() throws Exception {
    Mutability mu1 = Mutability.create("cycle1");
    StarlarkList<Object> cyclicSingleton = StarlarkList.newList(mu1);
    cyclicSingleton.addElement(cyclicSingleton);
    mu1.freeze();

    var deserializedSingleton =
        (List<?>)
            standardCodecs.deserializeMemoized(standardCodecs.serializeMemoized(cyclicSingleton));
    assertThat(deserializedSingleton).isInstanceOf(ImmutableSingletonStarlarkList.class);
    assertThat(deserializedSingleton).hasSize(1);
    assertThat(deserializedSingleton.get(0)).isSameInstanceAs(deserializedSingleton);

    Mutability mu2 = Mutability.create("cycle2");
    StarlarkList<Object> cyclicRegular = StarlarkList.of(mu2, "head");
    cyclicRegular.addElement(cyclicRegular);
    mu2.freeze();

    var deserializedRegular =
        (List<?>)
            standardCodecs.deserializeMemoized(standardCodecs.serializeMemoized(cyclicRegular));
    assertThat(deserializedRegular).isInstanceOf(RegularImmutableStarlarkList.class);
    assertThat(deserializedRegular).hasSize(2);
    assertThat(deserializedRegular.get(0)).isEqualTo("head");
    assertThat(deserializedRegular.get(1)).isSameInstanceAs(deserializedRegular);
  }
}
