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

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MultimapCodec}. */
@RunWith(JUnit4.class)
public class MultimapCodecTest {
  @Test
  public void testMultimap() throws Exception {
    LinkedHashMultimap<String, String> linkedHashMultimap = LinkedHashMultimap.create();
    linkedHashMultimap.put("A", "//foo:B");
    new SerializationTester(
            ImmutableMultimap.of(),
            ImmutableMultimap.of("A", "//foo:A"),
            ImmutableMultimap.builder().putAll("B", "//foo:B1", "//foo:B2").build(),
            linkedHashMultimap)
        .runTests();
  }

  @Test
  public void testImmutableListMultimap() throws Exception {
    new SerializationTester(
            ImmutableListMultimap.builder().putAll("A", "a", "b", "c", "d", "e").build())
        .setVerificationFunction(
            (VerificationFunction<ImmutableListMultimap<String, String>>)
                (deserialized, subject) ->
                    assertThat(deserialized.get("A"))
                        .containsExactly("a", "b", "c", "d", "e")
                        .inOrder())
        .runTests();
  }

  @Test
  public void testImmutableSetMultimap() throws Exception {
    new SerializationTester(
            ImmutableSetMultimap.builder().putAll("A", "a", "b").putAll("A", "b", "c").build())
        .setVerificationFunction(
            (VerificationFunction<ImmutableSetMultimap<String, String>>)
                (deserialized, subject) ->
                    assertThat(deserialized.get("A")).containsExactly("a", "b", "c"))
        .runTests();
  }
}
