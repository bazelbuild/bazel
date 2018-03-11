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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableMapCodec}. */
@RunWith(JUnit4.class)
public class ImmutableMapCodecTest {
  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            ImmutableMap.of(),
            ImmutableMap.of("A", "//foo:A"),
            ImmutableMap.of("B", "//foo:B"),
            ImmutableSortedMap.of(),
            ImmutableSortedMap.of("A", "//foo:A"),
            ImmutableSortedMap.of("B", "//foo:B"),
            ImmutableSortedMap.reverseOrder().put("a", "b").put("c", "d").build())
        // Check for order.
        .setVerificationFunction(
            (VerificationFunction<ImmutableMap<?, ?>>)
                (deserialized, subject) -> {
                  assertThat(deserialized).isEqualTo(subject);
                  assertThat(deserialized).containsExactlyEntriesIn(subject).inOrder();
                })
        .runTests();
  }

  @Test
  public void unnaturallySortedMapComesBackUnsortedInCorrectOrder() throws Exception {
    ImmutableMap<?, ?> deserialized =
        TestUtils.roundTrip(
            ImmutableSortedMap.reverseOrder().put("a", "b").put("c", "d").build(),
            ImmutableMap.of());
    assertThat(deserialized).isInstanceOf(ImmutableMap.class);
    assertThat(deserialized).isNotInstanceOf(ImmutableSortedMap.class);
    assertThat(deserialized).containsExactly("c", "d", "a", "b").inOrder();
  }
}
