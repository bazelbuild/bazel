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
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;

import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NestedArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.NotNestedSet.NotNestedSetCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ArrayListCodec}. */
@RunWith(TestParameterInjector.class)
public final class ArrayListCodecTest {
  @Test
  public void smokeTest(@TestParameter boolean memoize) throws Exception {
    SerializationTester tester =
        new SerializationTester(
                new ArrayList<>(),
                new ArrayList<>(Arrays.asList("one", "two", "three")),
                new ArrayList<>(Arrays.asList((String) null)),
                new ArrayList<>(Arrays.asList(null, "memberOne", null)),
                new ArrayList<>(Arrays.asList(1, 2, 3, null, 4, 5, 6)))
            .setVerificationFunction(
                (VerificationFunction<ArrayList<?>>)
                    (deserialized, subject) ->
                        assertThat(deserialized).containsExactlyElementsIn(subject).inOrder());
    if (memoize) {
      tester.makeMemoizing();
    }
    tester.runTests();
  }

  @Test
  public void selfReferentialList_preservesIdentity() throws Exception {
    // The list is registered before its elements are deserialized, so a backreference to the list
    // from one of its own elements resolves to the same instance.
    ArrayList<Object> subject = new ArrayList<>();
    subject.add("before");
    subject.add(subject);
    subject.add("after");

    new SerializationTester(subject)
        .makeMemoizing()
        .setVerificationFunction(
            (VerificationFunction<ArrayList<Object>>)
                (deserialized, original) -> {
                  assertThat(deserialized).hasSize(3);
                  assertThat(deserialized.get(0)).isEqualTo("before");
                  assertThat(deserialized.get(1)).isSameInstanceAs(deserialized);
                  assertThat(deserialized.get(2)).isEqualTo("after");
                })
        .runTests();
  }

  @Test
  public void elementsCompletingAsynchronously_roundTrip() throws Exception {
    // NotNestedSetCodec uses shared values, so with future blocking enabled, these elements are set
    // into the list asynchronously.
    Random rng = new Random(0);
    NotNestedSet shared = NotNestedSet.createRandom(rng, 4, 4, Random::nextInt);
    ArrayList<NotNestedSet> subject = new ArrayList<>();
    subject.add(NotNestedSet.createRandom(rng, 4, 4, Random::nextInt));
    subject.add(null);
    subject.add(shared);
    subject.add(shared);

    new SerializationTester(subject)
        .addCodec(new NotNestedSetCodec(new NestedArrayCodec()))
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(
            (VerificationFunction<ArrayList<NotNestedSet>>)
                (deserialized, original) -> {
                  assertThat(deserialized).hasSize(original.size());
                  assertThat(deserialized.get(1)).isNull();
                  for (int i : new int[] {0, 2, 3}) {
                    assertThat(dumpStructureWithEquivalenceReduction(deserialized.get(i)))
                        .isEqualTo(dumpStructureWithEquivalenceReduction(original.get(i)));
                  }
                  assertThat(deserialized.get(3)).isSameInstanceAs(deserialized.get(2));
                })
        .runTests();
  }
}
