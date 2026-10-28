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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NestedArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ImmutableListCodec}. */
@RunWith(TestParameterInjector.class)
public class ImmutableListCodecTest {
  @Test
  public void smoke(@TestParameter boolean memoize) throws Exception {
    SerializationTester tester =
        new SerializationTester(
                ImmutableList.of(),
                ImmutableList.of("foo"),
                ImmutableList.of("bar", "baz"),
                ImmutableList.of(1, 2, 3, 4, 5))
            .setVerificationFunction(
                (VerificationFunction<ImmutableList<?>>)
                    (original, deserialized) -> {
                      assertThat(deserialized).containsExactlyElementsIn(original).inOrder();
                      // The hidden Guava subclass is preserved.
                      assertThat(deserialized.getClass()).isEqualTo(original.getClass());
                      // Guards against Guava ever precomputing state from the initial elements.
                      assertThat(deserialized).isEqualTo(original);
                      assertThat(deserialized.hashCode()).isEqualTo(original.hashCode());
                    });
    if (memoize) {
      tester.makeMemoizing();
    }
    tester.runTests();
  }

  @Test
  public void cycleThroughList_preservesIdentity(@TestParameter({"1", "2", "3"}) int size)
      throws Exception {
    // The list is registered before its elements are deserialized, so an element that refers back
    // to the list, here via a mutable container, resolves to the same instance.
    ArrayList<Object> container = new ArrayList<>();
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    builder.add(container);
    for (int i = 1; i < size; i++) {
      builder.add("element" + i);
    }
    ImmutableList<Object> list = builder.build();
    container.add(list);

    new SerializationTester(container)
        .makeMemoizing()
        .setVerificationFunction(
            (VerificationFunction<ArrayList<Object>>)
                (original, deserialized) -> {
                  assertThat(deserialized).hasSize(1);
                  ImmutableList<?> deserializedList = (ImmutableList<?>) deserialized.get(0);
                  assertThat(deserializedList.getClass()).isEqualTo(list.getClass());
                  assertThat(deserializedList).hasSize(size);
                  assertThat(deserializedList.get(0)).isSameInstanceAs(deserialized);
                  for (int i = 1; i < size; i++) {
                    assertThat(deserializedList.get(i)).isEqualTo("element" + i);
                  }
                })
        .runTests();
  }

  @Test
  public void elementsCompletingAsynchronously_roundTrip(@TestParameter({"1", "2", "3"}) int size)
      throws Exception {
    // NotNestedSetCodec uses shared values, so with future blocking enabled, these elements are set
    // into the list asynchronously.
    Random rng = new Random(0);
    NotNestedSet shared = NotNestedSet.createRandom(rng, 4, 4, Random::nextInt);
    ImmutableList.Builder<NotNestedSet> builder = ImmutableList.builder();
    for (int i = 0; i < size; i++) {
      builder.add(i == 0 ? NotNestedSet.createRandom(rng, 4, 4, Random::nextInt) : shared);
    }
    ImmutableList<NotNestedSet> subject = builder.build();

    // Wrapped in a list because the ImmutableList overload of the constructor treats the list as
    // multiple subjects.
    new SerializationTester(ImmutableList.of(subject))
        .addCodec(new NotNestedSetCodec(new NestedArrayCodec()))
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(
            (VerificationFunction<ImmutableList<NotNestedSet>>)
                (original, deserialized) -> {
                  assertThat(deserialized.getClass()).isEqualTo(original.getClass());
                  assertThat(deserialized).hasSize(original.size());
                  for (int i = 0; i < original.size(); i++) {
                    assertThat(dumpStructureWithEquivalenceReduction(deserialized.get(i)))
                        .isEqualTo(dumpStructureWithEquivalenceReduction(original.get(i)));
                  }
                  for (int i = 2; i < original.size(); i++) {
                    assertThat(deserialized.get(i)).isSameInstanceAs(deserialized.get(1));
                  }
                  // The elements were set after the list was constructed. Guards against Guava
                  // ever precomputing state, like the hash code, from the initial elements.
                  ArrayList<NotNestedSet> copy = new ArrayList<>(deserialized);
                  assertThat(deserialized).isEqualTo(copy);
                  assertThat(deserialized.hashCode()).isEqualTo(copy.hashCode());
                })
        .runTests();
  }
}
