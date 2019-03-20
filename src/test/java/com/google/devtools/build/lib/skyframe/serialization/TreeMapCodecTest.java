// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MapCodec}. */
@RunWith(JUnit4.class)
public class TreeMapCodecTest {
  @Test
  public void smoke() throws Exception {
    TreeMap<String, String> map1 = new TreeMap<>();
    map1.put("a", "first");
    map1.put("b", null);
    TreeMap<String, String> map2 = new TreeMap<>();
    map2.put("c", null);
    map2.put("a", "second");
    new SerializationTester(map1, map2, Collections.unmodifiableMap(map1))
        .setVerificationFunction(
            (VerificationFunction<Map<String, String>>)
                (original, deserialized) ->
                    assertThat(deserialized).containsExactlyEntriesIn(original).inOrder())
        .runTests();
  }
}
