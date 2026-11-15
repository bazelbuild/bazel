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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSet;
import net.starlark.java.eval.Tuple;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link StarlarkSetCodec}. */
@RunWith(TestParameterInjector.class)
public final class StarlarkSetCodecTest {
  private final ObjectCodecs standardCodecs = new ObjectCodecs(AutoRegistry.get());

  @Test
  public void testCodec() throws Exception {
    Tuple aliasedTuple = Tuple.of("x", StarlarkInt.of(1));
    new SerializationTester(
            StarlarkSet.empty(),
            StarlarkSet.immutableCopyOf(ImmutableList.of("only")),
            StarlarkSet.immutableCopyOf(ImmutableList.of("c", "a", "b")),
            StarlarkSet.immutableCopyOf(
                ImmutableList.of(aliasedTuple, Tuple.of(aliasedTuple, StarlarkInt.of(2)))),
            frozenMutableCopyOf(createElementsInReverseOrder(12)))
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(StarlarkSetCodecTest::verifyDeserialization)
        .runTests();
  }

  @Test
  public void canonicalizesAllSetRepresentations(@TestParameter({"0", "1", "2", "12"}) int size)
      throws Exception {
    ImmutableList<String> elements = createElementsInReverseOrder(size);
    StarlarkSet<String> frozenMutable = frozenMutableCopyOf(elements);
    StarlarkSet<String> immutable = StarlarkSet.immutableCopyOf(elements);

    ByteString mutableBytes = standardCodecs.serialize(frozenMutable);
    ByteString immutableBytes = standardCodecs.serialize(immutable);
    assertThat(mutableBytes).isEqualTo(immutableBytes);

    for (StarlarkSet<String> candidate : ImmutableList.of(frozenMutable, immutable)) {
      var deserialized =
          (StarlarkSet<?>) standardCodecs.deserialize(standardCodecs.serialize(candidate));
      if (size == 0) {
        assertThat(deserialized).isSameInstanceAs(StarlarkSet.empty());
      }
      assertThat(deserialized.isImmutable()).isTrue();
      assertThat((Collection<?>) deserialized).containsExactlyElementsIn(elements).inOrder();
    }
  }

  @Test
  public void optimizingMemoryLayout_doesNotResultInDifferentSerializedBytes(
      @TestParameter({"0", "1", "2", "12"}) int size) throws Exception {
    StarlarkSet<String> set = frozenMutableCopyOf(createElementsInReverseOrder(size));

    ByteString beforeBytes = standardCodecs.serialize(set);
    Object optimized = set.unsafeOptimizeMemoryLayout();
    ByteString afterBytes = standardCodecs.serialize(set);
    ByteString optimizedBytes = standardCodecs.serialize(optimized);

    assertThat(beforeBytes).isEqualTo(afterBytes);
    assertThat(beforeBytes).isEqualTo(optimizedBytes);
  }

  @Test
  public void unfrozenSet_serializeThrows() {
    StarlarkSet<String> unfrozen =
        StarlarkSet.copyOf(Mutability.create("unfrozen"), ImmutableList.of("a", "b"));
    SerializationException e =
        assertThrows(SerializationException.class, () -> standardCodecs.serialize(unfrozen));
    assertThat(e).hasMessageThat().contains("Trying to serialize unfrozen StarlarkSet");
  }

  private static <E> StarlarkSet<E> frozenMutableCopyOf(Collection<E> elements) {
    Mutability mu = Mutability.create("frozen");
    StarlarkSet<E> set = StarlarkSet.copyOf(mu, elements);
    mu.freeze();
    return set;
  }

  private static ImmutableList<String> createElementsInReverseOrder(int size) {
    ImmutableList.Builder<String> elements = ImmutableList.builder();
    for (int i = size - 1; i >= 0; i--) {
      elements.add("elem_" + i);
    }
    return elements.build();
  }

  // Check for order.
  private static void verifyDeserialization(
      StarlarkSet<Object> subject, StarlarkSet<Object> deserialized) {
    assertThat(deserialized.isImmutable()).isTrue();
    assertThat((Set<?>) deserialized).isEqualTo(ImmutableSet.copyOf(subject));
    assertThat((List<?>) ImmutableList.copyOf(deserialized))
        .containsExactlyElementsIn(ImmutableList.copyOf(subject))
        .inOrder();
  }
}
