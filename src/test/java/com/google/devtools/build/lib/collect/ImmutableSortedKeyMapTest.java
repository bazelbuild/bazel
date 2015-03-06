// Copyright 2014 Google Inc. All rights reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.common.collect.Maps;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyMap.Builder;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A test for {@link ImmutableSortedKeyListMultimap}. Started out as a blatant copy of
 * ImmutableListMapTest.
 */
@RunWith(JUnit4.class)
public class ImmutableSortedKeyMapTest {

  @Test
  public void emptyBuilder() {
    ImmutableSortedKeyMap<String, Integer> map
        = ImmutableSortedKeyMap.<String, Integer>builder().build();
    assertThat(map).isEmpty();
  }

  @Test
  public void singletonBuilder() {
    ImmutableSortedKeyMap<String, Integer> map = ImmutableSortedKeyMap.<String, Integer>builder()
        .put("one", 1)
        .build();
    assertMapEquals(map, "one", 1);
  }

  @Test
  public void builder() {
    ImmutableSortedKeyMap<String, Integer> map = ImmutableSortedKeyMap.<String, Integer>builder()
        .put("one", 1)
        .put("two", 2)
        .put("three", 3)
        .put("four", 4)
        .put("five", 5)
        .build();
    assertMapEquals(map,
        "five", 5, "four", 4, "one", 1, "three", 3, "two", 2);
  }

  @Test
  public void builderPutAllWithEmptyMap() {
    ImmutableSortedKeyMap<String, Integer> map = ImmutableSortedKeyMap.<String, Integer>builder()
        .putAll(Collections.<String, Integer>emptyMap())
        .build();
    assertThat(map).isEmpty();
  }

  @Test
  public void builderPutAll() {
    Map<String, Integer> toPut = new LinkedHashMap<>();
    toPut.put("one", 1);
    toPut.put("two", 2);
    toPut.put("three", 3);
    Map<String, Integer> moreToPut = new LinkedHashMap<>();
    moreToPut.put("four", 4);
    moreToPut.put("five", 5);

    ImmutableSortedKeyMap<String, Integer> map = ImmutableSortedKeyMap.<String, Integer>builder()
        .putAll(toPut)
        .putAll(moreToPut)
        .build();
    assertMapEquals(map,
        "five", 5, "four", 4, "one", 1, "three", 3, "two", 2);
  }

  @Test
  public void builderReuse() {
    ImmutableSortedKeyMap.Builder<String, Integer> builder =
        ImmutableSortedKeyMap.<String, Integer>builder();
    ImmutableSortedKeyMap<String, Integer> mapOne = builder
        .put("one", 1)
        .put("two", 2)
        .build();
    ImmutableSortedKeyMap<String, Integer> mapTwo = builder
        .put("three", 3)
        .put("four", 4)
        .build();

    assertMapEquals(mapOne, "one", 1, "two", 2);
    assertMapEquals(mapTwo, "four", 4, "one", 1, "three", 3, "two", 2);
  }

  @Test
  public void builderPutNullKey() {
    Builder<String, Integer> builder = new Builder<>();
    try {
      builder.put(null, 1);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void builderPutNullValue() {
    Builder<String, Integer> builder = new Builder<>();
    try {
      builder.put("one", null);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void builderPutNullKeyViaPutAll() {
    Builder<String, Integer> builder = new Builder<>();
    try {
      builder.putAll(Collections.<String, Integer>singletonMap(null, 1));
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void builderPutNullValueViaPutAll() {
    Builder<String, Integer> builder = new Builder<>();
    try {
      builder.putAll(Collections.<String, Integer>singletonMap("one", null));
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void of() {
    assertMapEquals(
        ImmutableSortedKeyMap.of("one", 1),
        "one", 1);
    assertMapEquals(
        ImmutableSortedKeyMap.of("one", 1, "two", 2),
        "one", 1, "two", 2);
  }

  @Test
  public void ofNullKey() {
    try {
      ImmutableSortedKeyMap.of((String) null, 1);
      fail();
    } catch (NullPointerException expected) {
    }

    try {
      ImmutableSortedKeyMap.of("one", 1, null, 2);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void ofNullValue() {
    try {
      ImmutableSortedKeyMap.of("one", null);
      fail();
    } catch (NullPointerException expected) {
    }

    try {
      ImmutableSortedKeyMap.of("one", 1, "two", null);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void copyOfEmptyMap() {
    ImmutableSortedKeyMap<String, Integer> copy
        = ImmutableSortedKeyMap.copyOf(Collections.<String, Integer>emptyMap());
    assertThat(copy).isEmpty();
    assertSame(copy, ImmutableSortedKeyMap.copyOf(copy));
  }

  @Test
  public void copyOfSingletonMap() {
    ImmutableSortedKeyMap<String, Integer> copy
        = ImmutableSortedKeyMap.copyOf(Collections.singletonMap("one", 1));
    assertMapEquals(copy, "one", 1);
    assertSame(copy, ImmutableSortedKeyMap.copyOf(copy));
  }

  @Test
  public void copyOf() {
    Map<String, Integer> original = new LinkedHashMap<>();
    original.put("one", 1);
    original.put("two", 2);
    original.put("three", 3);

    ImmutableSortedKeyMap<String, Integer> copy = ImmutableSortedKeyMap.copyOf(original);
    assertMapEquals(copy, "one", 1, "three", 3, "two", 2);
    assertSame(copy, ImmutableSortedKeyMap.copyOf(copy));
  }

  @Test
  public void nullGet() {
    ImmutableSortedKeyMap<String, Integer> map = ImmutableSortedKeyMap.of("one", 1);
    assertThat(map).doesNotContainKey(null);
  }

  @Test
  public void nullPointers() {
    NullPointerTester tester = new NullPointerTester();
    tester.testAllPublicStaticMethods(ImmutableSortedKeyMap.class);
    tester.testAllPublicInstanceMethods(
        new ImmutableSortedKeyMap.Builder<String, Object>());
    tester.testAllPublicInstanceMethods(ImmutableSortedKeyMap.<String, Integer>of());
    tester.testAllPublicInstanceMethods(ImmutableSortedKeyMap.of("one", 1));
    tester.testAllPublicInstanceMethods(
        ImmutableSortedKeyMap.of("one", 1, "two", 2));
  }

  private static <K, V> void assertMapEquals(Map<K, V> map,
      Object... alternatingKeysAndValues) {
    assertEquals(map.size(), alternatingKeysAndValues.length / 2);
    int i = 0;
    for (Entry<K, V> entry : map.entrySet()) {
      assertEquals(alternatingKeysAndValues[i++], entry.getKey());
      assertEquals(alternatingKeysAndValues[i++], entry.getValue());
    }
  }

  private static class IntHolder implements Serializable {
    public int value;

    public IntHolder(int value) {
      this.value = value;
    }

    @Override public boolean equals(Object o) {
      return (o instanceof IntHolder) && ((IntHolder) o).value == value;
    }

    @Override public int hashCode() {
      return value;
    }

    private static final long serialVersionUID = 5;
  }

  @Test
  public void mutableValues() {
    IntHolder holderA = new IntHolder(1);
    IntHolder holderB = new IntHolder(2);
    Map<String, IntHolder> map = ImmutableSortedKeyMap.of("a", holderA, "b", holderB);
    holderA.value = 3;
    assertThat(map.entrySet()).contains(Maps.immutableEntry("a", new IntHolder(3)));
    Map<String, Integer> intMap = ImmutableSortedKeyMap.of("a", 3, "b", 2);
    assertEquals(intMap.hashCode(), map.entrySet().hashCode());
    assertEquals(intMap.hashCode(), map.hashCode());
  }

  @Test
  public void toStringTest() {
    Map<String, Integer> map = ImmutableSortedKeyMap.of("a", 1, "b", 2);
    assertEquals("{a=1, b=2}", map.toString());
    map = ImmutableSortedKeyMap.of();
    assertEquals("{}", map.toString());
  }
}
