// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.util.Map;
import net.starlark.java.eval.Dict;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DictCodec}. */
@RunWith(JUnit4.class)
public class DictCodecTest {
  @Test
  public void testCodec() throws Exception {
    Dict<Object, Object> aliasedInnerDict =
        Dict.immutableCopyOf(ImmutableMap.of("1", "2", "3", "4"));
    new SerializationTester(
            Dict.empty(),
            Dict.immutableCopyOf(
                ImmutableMap.of("1", "2", "3", Dict.immutableCopyOf(ImmutableMap.of("4", "5")))),
            Dict.immutableCopyOf(ImmutableMap.of("10", aliasedInnerDict, "20", aliasedInnerDict)))
        .makeMemoizing()
        .setVerificationFunction(DictCodecTest::verifyDeserialization)
        .runTests();
  }

  // Check for order.
  private static void verifyDeserialization(
      Map<Object, Object> deserialized, Dict<Object, Object> subject) {
    assertThat(deserialized).isEqualTo(subject);
    assertThat(deserialized).containsExactlyEntriesIn(subject).inOrder();
  }
}
