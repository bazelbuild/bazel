// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.testing.google.UnmodifiableCollectionTests;
import com.google.common.testing.EqualsTester;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link ImmutableSortedKeyListMultimap}. Started out as a copy of
 * ImmutableListMultimapTest.
 */
@RunWith(JUnit4.class)
public class ImmutableSortedKeyListMultimapTest {

  @Test
  public void builderPutAllIterable() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", Arrays.asList(1, 2, 3));
    builder.putAll("bar", Arrays.asList(4, 5));
    builder.putAll("foo", Arrays.asList(6, 7));
    Multimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 3, 6, 7), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5), multimap.get("bar"));
    assertEquals(7, multimap.size());
  }

  @Test
  public void builderPutAllVarargs() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.putAll("foo", 6, 7);
    Multimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 3, 6, 7), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5), multimap.get("bar"));
    assertEquals(7, multimap.size());
  }

  @Test
  public void builderPutAllMultimap() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put("foo", 1);
    toPut.put("bar", 4);
    toPut.put("foo", 2);
    toPut.put("foo", 3);
    Multimap<String, Integer> moreToPut = LinkedListMultimap.create();
    moreToPut.put("foo", 6);
    moreToPut.put("bar", 5);
    moreToPut.put("foo", 7);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll(toPut);
    builder.putAll(moreToPut);
    Multimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 3, 6, 7), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5), multimap.get("bar"));
    assertEquals(7, multimap.size());
  }

  @Test
  public void builderPutAllWithDuplicates() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.putAll("foo", 1, 6, 7);
    ImmutableSortedKeyListMultimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 3, 1, 6, 7), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5), multimap.get("bar"));
    assertEquals(8, multimap.size());
  }

  @Test
  public void builderPutWithDuplicates() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.put("foo", 1);
    ImmutableSortedKeyListMultimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 3, 1), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5), multimap.get("bar"));
    assertEquals(6, multimap.size());
  }

  @Test
  public void builderPutAllMultimapWithDuplicates() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put("foo", 1);
    toPut.put("bar", 4);
    toPut.put("foo", 2);
    toPut.put("foo", 1);
    toPut.put("bar", 5);
    Multimap<String, Integer> moreToPut = LinkedListMultimap.create();
    moreToPut.put("foo", 6);
    moreToPut.put("bar", 4);
    moreToPut.put("foo", 7);
    moreToPut.put("foo", 2);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll(toPut);
    builder.putAll(moreToPut);
    Multimap<String, Integer> multimap = builder.build();
    assertEquals(Arrays.asList(1, 2, 1, 6, 7, 2), multimap.get("foo"));
    assertEquals(Arrays.asList(4, 5, 4), multimap.get("bar"));
    assertEquals(9, multimap.size());
  }

  @Test
  public void builderPutNullKey() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put("foo", null);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    try {
      builder.put(null, 1);
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll(null, Arrays.asList(1, 2, 3));
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll(null, 1, 2, 3);
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll(toPut);
      fail();
    } catch (NullPointerException expected) {}
  }

  @Test
  public void builderPutNullValue() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put(null, 1);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    try {
      builder.put("foo", null);
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll("foo", Arrays.asList(1, null, 3));
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll("foo", 1, null, 3);
      fail();
    } catch (NullPointerException expected) {}
    try {
      builder.putAll(toPut);
      fail();
    } catch (NullPointerException expected) {}
  }

  @Test
  public void copyOf() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put("foo", 1);
    input.put("bar", 2);
    input.put("foo", 3);
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertEquals(multimap, input);
    assertEquals(input, multimap);
  }

  @Test
  public void copyOfWithDuplicates() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put("foo", 1);
    input.put("bar", 2);
    input.put("foo", 3);
    input.put("foo", 1);
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertEquals(multimap, input);
    assertEquals(input, multimap);
  }

  @Test
  public void copyOfEmpty() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertEquals(multimap, input);
    assertEquals(input, multimap);
  }

  @Test
  public void copyOfImmutableListMultimap() {
    Multimap<String, Integer> multimap = createMultimap();
    assertSame(multimap, ImmutableSortedKeyListMultimap.copyOf(multimap));
  }

  @Test
  public void copyOfNullKey() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put(null, 1);
    try {
      ImmutableSortedKeyListMultimap.copyOf(input);
      fail();
    } catch (NullPointerException expected) {}
  }

  @Test
  public void copyOfNullValue() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.putAll("foo", Arrays.asList(1, null, 3));
    try {
      ImmutableSortedKeyListMultimap.copyOf(input);
      fail();
    } catch (NullPointerException expected) {}
  }

  @Test
  public void emptyMultimapReads() {
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.of();
    assertFalse(multimap.containsKey("foo"));
    assertFalse(multimap.containsValue(1));
    assertFalse(multimap.containsEntry("foo", 1));
    assertTrue(multimap.entries().isEmpty());
    assertTrue(multimap.equals(ArrayListMultimap.create()));
    assertEquals(Collections.emptyList(), multimap.get("foo"));
    assertEquals(0, multimap.hashCode());
    assertTrue(multimap.isEmpty());
    assertEquals(HashMultiset.create(), multimap.keys());
    assertEquals(Collections.emptySet(), multimap.keySet());
    assertEquals(0, multimap.size());
    assertTrue(multimap.values().isEmpty());
    assertEquals("{}", multimap.toString());
  }

  @Test
  public void emptyMultimapWrites() {
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.of();
    UnmodifiableCollectionTests.assertMultimapIsUnmodifiable(
        multimap, "foo", 1);
  }

  private Multimap<String, Integer> createMultimap() {
    return ImmutableSortedKeyListMultimap.<String, Integer>builder()
        .put("foo", 1).put("bar", 2).put("foo", 3).build();
  }

  @Test
  public void multimapReads() {
    Multimap<String, Integer> multimap = createMultimap();
    assertTrue(multimap.containsKey("foo"));
    assertFalse(multimap.containsKey("cat"));
    assertTrue(multimap.containsValue(1));
    assertFalse(multimap.containsValue(5));
    assertTrue(multimap.containsEntry("foo", 1));
    assertFalse(multimap.containsEntry("cat", 1));
    assertFalse(multimap.containsEntry("foo", 5));
    assertFalse(multimap.entries().isEmpty());
    assertEquals(3, multimap.size());
    assertFalse(multimap.isEmpty());
    assertEquals("{bar=[2], foo=[1, 3]}", multimap.toString());
  }

  @Test
  public void multimapWrites() {
    Multimap<String, Integer> multimap = createMultimap();
    UnmodifiableCollectionTests.assertMultimapIsUnmodifiable(
        multimap, "bar", 2);
  }

  @Test
  public void multimapEquals() {
    Multimap<String, Integer> multimap = createMultimap();
    Multimap<String, Integer> arrayListMultimap
        = ArrayListMultimap.create();
    arrayListMultimap.putAll("foo", Arrays.asList(1, 3));
    arrayListMultimap.put("bar", 2);

    new EqualsTester()
        .addEqualityGroup(multimap, createMultimap(), arrayListMultimap,
            ImmutableSortedKeyListMultimap.<String, Integer>builder()
                .put("bar", 2).put("foo", 1).put("foo", 3).build())
        .addEqualityGroup(ImmutableSortedKeyListMultimap.<String, Integer>builder()
            .put("bar", 2).put("foo", 3).put("foo", 1).build())
        .addEqualityGroup(ImmutableSortedKeyListMultimap.<String, Integer>builder()
            .put("foo", 2).put("foo", 3).put("foo", 1).build())
        .addEqualityGroup(ImmutableSortedKeyListMultimap.<String, Integer>builder()
            .put("bar", 2).put("foo", 3).build())
        .testEquals();
  }

  @Test
  public void asMap() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", Arrays.asList(1, 2, 3));
    builder.putAll("bar", Arrays.asList(4, 5));
    Map<String, Collection<Integer>> map = builder.build().asMap();
    assertEquals(Arrays.asList(1, 2, 3), map.get("foo"));
    assertEquals(Arrays.asList(4, 5), map.get("bar"));
    assertEquals(2, map.size());
    assertTrue(map.containsKey("foo"));
    assertTrue(map.containsKey("bar"));
    assertFalse(map.containsKey("notfoo"));
  }

  @Test
  public void asMapEntries() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", Arrays.asList(1, 2, 3));
    builder.putAll("bar", Arrays.asList(4, 5));
    Set<Map.Entry<String, Collection<Integer>>> set = builder.build().asMap().entrySet();
    Set<Map.Entry<String, Collection<Integer>>> other =
        ImmutableSet.<Map.Entry<String, Collection<Integer>>>builder()
        .add(new SimpleImmutableEntry<String, Collection<Integer>>("foo", Arrays.asList(1, 2, 3)))
        .add(new SimpleImmutableEntry<String, Collection<Integer>>("bar", Arrays.asList(4, 5)))
        .build();
    assertEquals(other, set);
  }
}
