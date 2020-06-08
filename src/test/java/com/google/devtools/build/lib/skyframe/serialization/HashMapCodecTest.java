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

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link HashMapCodec}. */
@RunWith(JUnit4.class)
public final class HashMapCodecTest {
  @Test
  public void smoke() throws Exception {
    HashMap<String, String> map1 = new HashMap<>();
    map1.put("a", "first");
    map1.put("b", null);
    LinkedHashMap<String, String> map2 = new LinkedHashMap<>();
    map2.put("c", null);
    map2.put("a", "second");
    HashMap<String, String> emptyMap = new HashMap<>();
    List<Map<String, String>> deserializedMaps = new ArrayList<>();
    // Put in empty map twice to make sure codec doesn't return same empty object.
    new SerializationTester(map1, map2, emptyMap, emptyMap)
        .setVerificationFunction(
            (VerificationFunction<Map<String, String>>)
                (original, deserialized) -> {
                  assertThat(deserialized).containsExactlyEntriesIn(original).inOrder();
                  deserializedMaps.add(deserialized);
                })
        .runTests();
    assertThat(deserializedMaps).hasSize(4);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        if (i == j) {
          continue;
        }
        assertThat(deserializedMaps.get(i)).isNotSameInstanceAs(deserializedMaps.get(j));
      }
    }
  }
}
