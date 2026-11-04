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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructure;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class DeferredNestedSetCodecTest {

  private static final CompressionService COMPRESSION_SERVICE = new CompressionServiceImpl();

  @Test
  public void empty() throws Exception {
    new SerializationTester(
            Order.STABLE_ORDER.emptySet(),
            Order.COMPILE_ORDER.emptySet(),
            Order.LINK_ORDER.emptySet(),
            Order.NAIVE_LINK_ORDER.emptySet())
        .addCodec(new DeferredNestedSetCodec())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .runTests();
  }

  @Test
  public void singleton() throws Exception {
    try (var _ = NestedSetOwnerScope.enter("owner")) {
      new SerializationTester(
              NestedSetBuilder.stableOrder().add("A").build(),
              NestedSetBuilder.compileOrder().add("B").build(),
              NestedSetBuilder.linkOrder().add("C").build(),
              NestedSetBuilder.naiveLinkOrder().add("D").build())
          .addCodec(new DeferredNestedSetCodec())
          .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
          .setVerificationFunction(DeferredNestedSetCodecTest::verifyUsingShallowEquals)
          .runTests();
    }
  }

  @Test
  public void array() throws Exception {
    try (var _ = NestedSetOwnerScope.enter("owner")) {
      new SerializationTester(
              NestedSetBuilder.stableOrder().addAll(ImmutableList.of(1, 2, 3)).build(),
              NestedSetBuilder.compileOrder().addAll(ImmutableList.of("A", "B", "C")).build(),
              NestedSetBuilder.linkOrder().addAll(ImmutableList.of(5.56, 3.14, 10, 20)).build(),
              NestedSetBuilder.naiveLinkOrder()
                  .addAll(ImmutableList.of("one", "two", "three", "four", "five"))
                  .build())
          .addCodec(new DeferredNestedSetCodec())
          .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
          .setVerificationFunction(DeferredNestedSetCodecTest::verifyUsingShallowEquals)
          .runTests();
    }
  }

  @Test
  public void diamond() throws Exception {
    NestedSet<Object> top;
    try (var _ = NestedSetOwnerScope.enter("owner")) {
      var root = NestedSetBuilder.stableOrder().addAll(ImmutableList.of(1, 2)).build();
      var left = NestedSetBuilder.stableOrder().add("left").addTransitive(root).build();
      var right = NestedSetBuilder.stableOrder().add("right").addTransitive(root).build();
      top =
          NestedSetBuilder.stableOrder()
              .addAll(ImmutableList.of("this", "is", "the", "top"))
              .addTransitive(left)
              .addTransitive(right)
              .build();
    }

    var fingerprintValueService = FingerprintValueService.createForTesting();
    var codecs =
        new ObjectCodecs(AutoRegistry.get().getBuilder().add(new DeferredNestedSetCodec()).build());

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(COMPRESSION_SERVICE, fingerprintValueService, top);
    ListenableFuture<?> futureToBlockWritesOn = serialized.getFutureToBlockWritesOn();
    if (futureToBlockWritesOn != null) {
      var unused = futureToBlockWritesOn.get();
    }
    ByteString bytes = serialized.getObject();

    NestedSet<?> deserialized =
        (NestedSet<?>)
            codecs.deserializeMemoizedAndBlocking(
                COMPRESSION_SERVICE, fingerprintValueService, bytes);
    // Since dumpStructure doesn't perform equivalence reduction, equivalence here means the diamond
    // reference structure was preserved by deserialization.
    assertThat(dumpStructure(top)).isEqualTo(dumpStructure(deserialized));
  }

  @Test
  public void roundTrip_preservesOwnerAndIndex() throws Exception {
    NestedSet<String> singleton;
    NestedSet<Integer> array;
    try (var _ = NestedSetOwnerScope.enter("//pkg:target")) {
      singleton = NestedSetBuilder.<String>stableOrder().add("A").build();
      array = NestedSetBuilder.<Integer>stableOrder().addAll(ImmutableList.of(1, 2, 3)).build();
    }

    NestedSet<?> deserializedSingleton = roundTrip(singleton);
    NestedSet<?> deserializedArray = roundTrip(array);

    assertThat(deserializedSingleton.getOwner()).isEqualTo("//pkg:target");
    assertThat(deserializedSingleton.getIndex()).isEqualTo(0);
    assertThat(deserializedSingleton).isEqualTo(singleton);
    assertThat(deserializedSingleton.hashCode()).isEqualTo(singleton.hashCode());
    assertThat(deserializedSingleton.toList()).containsExactly("A");
    assertThat(deserializedArray.getOwner()).isEqualTo("//pkg:target");
    assertThat(deserializedArray.getIndex()).isEqualTo(1);
    assertThat(deserializedArray).isEqualTo(array);
    assertThat(deserializedArray.hashCode()).isEqualTo(array.hashCode());
    assertThat(deserializedArray.toList()).containsExactly(1, 2, 3).inOrder();
  }

  @Test
  public void serialize_withoutOwner_throws() {
    NestedSet<Integer> original =
        NestedSetBuilder.<Integer>stableOrder().addAll(ImmutableList.of(1, 2, 3)).build();
    assertThat(original.getOwner()).isNull();

    SerializationException e =
        assertThrows(SerializationException.class, () -> roundTrip(original));
    assertThat(e).hasMessageThat().contains("NestedSet without owner should not be serialized");
  }

  @Test
  public void serialize_withoutOwner_allowedWhenCheckDisabled() throws Exception {
    NestedSet<Integer> original =
        NestedSetBuilder.<Integer>stableOrder().addAll(ImmutableList.of(1, 2, 3)).build();
    assertThat(original.getOwner()).isNull();

    DeferredNestedSetCodec.checkNestedSetOwner.set(false);
    try {
      NestedSet<?> deserialized = roundTrip(original);
      assertThat(deserialized.getOwner()).isNull();
      assertThat(deserialized.toList()).containsExactly(1, 2, 3).inOrder();
    } finally {
      DeferredNestedSetCodec.checkNestedSetOwner.set(true);
    }
  }

  /**
   * Regression test for b/542300922.
   *
   * <p>A Skycache reader can hold both a deserialized copy of a target's set and a copy from
   * analyzing the target locally. When both are fields of struct-like elements of an outer set, the
   * outer set used to keep both elements, which produced duplicate actions.
   */
  @Test
  public void deserializedAndLocalCopies_deduplicateInsideOuterSet() throws Exception {
    NestedSet<String> writerSet;
    try (var _ = NestedSetOwnerScope.enter("//pkg:t")) {
      writerSet = NestedSetBuilder.<String>stableOrder().add("a.js").add("b.js").build();
    }
    NestedSet<?> deserialized = roundTrip(writerSet);
    NestedSet<String> localSet;
    try (var _ = NestedSetOwnerScope.enter("//pkg:t")) {
      localSet = NestedSetBuilder.<String>stableOrder().add("a.js").add("b.js").build();
    }
    assertThat(deserialized).isNotSameInstanceAs(localSet);

    // Like depset([struct(files = d) for d in ...]).to_list().
    NestedSet<StructLike> outer =
        NestedSetBuilder.<StructLike>stableOrder()
            .add(new StructLike(deserialized))
            .add(new StructLike(localSet))
            .build();

    assertThat(outer.toList()).containsExactly(new StructLike(localSet));
  }

  /** Stands in for a Starlark struct, whose equality delegates to its field values. */
  private record StructLike(Object files) {}

  private static NestedSet<?> roundTrip(NestedSet<?> original) throws Exception {
    var fingerprintValueService = FingerprintValueService.createForTesting();
    var codecs =
        new ObjectCodecs(AutoRegistry.get().getBuilder().add(new DeferredNestedSetCodec()).build());
    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(COMPRESSION_SERVICE, fingerprintValueService, original);
    ListenableFuture<?> futureToBlockWritesOn = serialized.getFutureToBlockWritesOn();
    if (futureToBlockWritesOn != null) {
      var _ = futureToBlockWritesOn.get();
    }
    return (NestedSet<?>)
        codecs.deserializeMemoizedAndBlocking(
            COMPRESSION_SERVICE, fingerprintValueService, serialized.getObject());
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static void verifyUsingShallowEquals(NestedSet original, NestedSet deserialized) {
    assertThat(original.shallowEquals(deserialized)).isTrue();
  }
}
