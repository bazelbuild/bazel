// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import java.util.Comparator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableSortedSetCodec}. */
@RunWith(JUnit4.class)
public class ImmutableSortedSetCodecTest {
  @SerializationConstant @VisibleForSerialization
  static final Comparator<String> LENGTH_COMPARATOR = Comparator.comparingInt(String::length);

  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            ImmutableSortedSet.of(),
            ImmutableSortedSet.of("a", "b", "d"),
            ImmutableSortedSet.orderedBy(Comparator.<String>naturalOrder()).add("a", "c").build(),
            ImmutableSortedSet.orderedBy(LENGTH_COMPARATOR).add("abc", "defg", "h", "").build())
        // Check for order and comparator.
        .setVerificationFunction(
            (VerificationFunction<ImmutableSortedSet<String>>)
                (deserialized, subject) -> {
                  assertThat(deserialized).isEqualTo(subject);
                  assertThat(deserialized).containsExactlyElementsIn(subject).inOrder();
                  assertThat(deserialized.comparator()).isEqualTo(subject.comparator());
                })
        .runTests();
  }

  @Test
  public void unknowComparatorThrows() {
    assertThrows(
        SerializationException.class,
        () ->
            RoundTripping.roundTrip(
                ImmutableSortedSet.orderedBy(Comparator.comparingInt(String::length))
                    .add("a", "bcd", "ef")));
  }
}
