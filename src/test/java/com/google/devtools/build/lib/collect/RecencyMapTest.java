// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RecencyMap}. */
@RunWith(JUnit4.class)
public final class RecencyMapTest {

  @Test
  public void put_newKeys_preservesInsertionOrder() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);
    map.put("b", 2);
    map.put("c", 3);

    assertThat(map).containsExactly("a", 1, "b", 2, "c", 3).inOrder();
  }

  @Test
  public void put_existingKey_movesKeyToEnd() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);
    map.put("b", 2);
    map.put("c", 3);
    map.put("a", 4);

    assertThat(map).containsExactly("b", 2, "c", 3, "a", 4).inOrder();
  }

  @Test
  public void put_existingKey_returnsPreviousValue() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);

    assertThat(map.put("a", 2)).isEqualTo(1);
    assertThat(map.put("b", 3)).isNull();
  }

  @Test
  public void putAll_existingKeys_movesKeysToEnd() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);
    map.put("b", 2);
    map.put("c", 3);

    map.putAll(ImmutableMap.of("b", 4, "d", 5));

    assertThat(map).containsExactly("a", 1, "c", 3, "b", 4, "d", 5).inOrder();
  }

  @Test
  public void remove_thenPut_appendsKeyAtEnd() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);
    map.put("b", 2);

    assertThat(map.remove("a")).isEqualTo(1);
    map.put("a", 3);

    assertThat(map).containsExactly("b", 2, "a", 3).inOrder();
  }

  @Test
  public void equalsAndHashCode_matchEquivalentMap() {
    RecencyMap<String, Integer> map = new RecencyMap<>();
    map.put("a", 1);
    map.put("b", 2);
    map.put("a", 3);

    ImmutableMap<String, Integer> expected = ImmutableMap.of("b", 2, "a", 3);
    assertThat(map).isEqualTo(expected);
    assertThat(map.hashCode()).isEqualTo(expected.hashCode());
  }
}
