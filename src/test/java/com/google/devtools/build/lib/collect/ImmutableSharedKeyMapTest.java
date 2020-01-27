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

package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableSharedKeyMap}. */
@RunWith(JUnit4.class)
public final class ImmutableSharedKeyMapTest {

  @Test
  public void testBasicFunctionality() throws Exception {
    Object valueA = new Object();
    Object valueB = new Object();
    ImmutableSharedKeyMap<String, Object> map =
        ImmutableSharedKeyMap.<String, Object>builder().put("a", valueA).put("b", valueB).build();

    assertThat(map.get("a")).isSameInstanceAs(valueA);
    assertThat(map.get("b")).isSameInstanceAs(valueB);
    assertThat(map.get("c")).isNull();

    // Verify that we can find all items both by iteration and indexing
    ImmutableMap.Builder<String, Object> iterationCopy = ImmutableMap.builder();
    for (String key : map) {
      iterationCopy.put(key, map.get(key));
    }
    assertThat(iterationCopy.build()).isEqualTo(ImmutableMap.of("a", valueA, "b", valueB));

    ImmutableMap.Builder<String, Object> arrayIterationCopy = ImmutableMap.builder();
    for (int i = 0; i < map.size(); ++i) {
      arrayIterationCopy.put(map.keyAt(i), map.valueAt(i));
    }
    assertThat(arrayIterationCopy.build()).isEqualTo(ImmutableMap.of("a", valueA, "b", valueB));
  }

  @Test
  public void testEquality() throws Exception {
    ImmutableSharedKeyMap<String, Object> emptyMap =
        ImmutableSharedKeyMap.<String, Object>builder().build();

    Object valueA = new Object();
    Object valueB = new Object();

    ImmutableSharedKeyMap<String, Object> map =
        ImmutableSharedKeyMap.<String, Object>builder().put("a", valueA).put("b", valueB).build();

    // Two identically ordered maps are equal
    ImmutableSharedKeyMap<String, Object> exactCopy =
        ImmutableSharedKeyMap.<String, Object>builder().put("a", valueA).put("b", valueB).build();

    // The map is order sensitive, so different insertion orders aren't equal
    ImmutableSharedKeyMap<String, Object> oppositeOrderMap =
        ImmutableSharedKeyMap.<String, Object>builder().put("b", valueB).put("a", valueA).build();

    Object valueC = new Object();
    ImmutableSharedKeyMap<String, Object> biggerMap =
        ImmutableSharedKeyMap.<String, Object>builder()
            .put("a", valueA)
            .put("b", valueB)
            .put("c", valueC)
            .build();

    new EqualsTester()
        .addEqualityGroup(emptyMap)
        .addEqualityGroup(map, exactCopy)
        .addEqualityGroup(oppositeOrderMap)
        .addEqualityGroup(biggerMap)
        .testEquals();
  }

  @Test
  public void testMultipleIdenticalKeysThrowsException() throws Exception {
    Object valueA = new Object();
    Object valueB = new Object();
    Object valueC = new Object();
    ImmutableSharedKeyMap.Builder<String, Object> map =
        ImmutableSharedKeyMap.<String, Object>builder()
            .put("key", valueA)
            .put("key", valueB)
            .put("key", valueC);

    assertThrows(IllegalArgumentException.class, () -> map.build());
  }

  private static class SameHashCodeClass {
    @Override
    public int hashCode() {
      return 0;
    }
  }

  @Test
  public void testTwoKeysWithTheSameHashCode() throws Exception {
    SameHashCodeClass keyA = new SameHashCodeClass();
    SameHashCodeClass keyB = new SameHashCodeClass();
    Object valueA = new Object();
    Object valueB = new Object();
    ImmutableSharedKeyMap<SameHashCodeClass, Object> map =
        ImmutableSharedKeyMap.<SameHashCodeClass, Object>builder()
            .put(keyA, valueA)
            .put(keyB, valueB)
            .build();
    assertThat(map.get(keyA)).isSameInstanceAs(valueA);
    assertThat(map.get(keyB)).isSameInstanceAs(valueB);
  }
}
