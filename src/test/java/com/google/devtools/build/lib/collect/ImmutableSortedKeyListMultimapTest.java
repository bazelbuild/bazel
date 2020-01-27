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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

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
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 3, 6, 7).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5).inOrder();
    assertThat(multimap).hasSize(7);
  }

  @Test
  public void builderPutAllVarargs() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.putAll("foo", 6, 7);
    Multimap<String, Integer> multimap = builder.build();
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 3, 6, 7).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5).inOrder();
    assertThat(multimap).hasSize(7);
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
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 3, 6, 7).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5).inOrder();
    assertThat(multimap).hasSize(7);
  }

  @Test
  public void builderPutAllWithDuplicates() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.putAll("foo", 1, 6, 7);
    ImmutableSortedKeyListMultimap<String, Integer> multimap = builder.build();
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 3, 1, 6, 7).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5).inOrder();
    assertThat(multimap).hasSize(8);
  }

  @Test
  public void builderPutWithDuplicates() {
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    builder.putAll("foo", 1, 2, 3);
    builder.putAll("bar", 4, 5);
    builder.put("foo", 1);
    ImmutableSortedKeyListMultimap<String, Integer> multimap = builder.build();
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 3, 1).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5).inOrder();
    assertThat(multimap).hasSize(6);
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
    assertThat(multimap).valuesForKey("foo").containsExactly(1, 2, 1, 6, 7, 2).inOrder();
    assertThat(multimap).valuesForKey("bar").containsExactly(4, 5, 4).inOrder();
    assertThat(multimap).hasSize(9);
  }

  @Test
  public void builderPutNullKey() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put("foo", null);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    assertThrows(NullPointerException.class, () -> builder.put(null, 1));
    assertThrows(NullPointerException.class, () -> builder.putAll(null, Arrays.asList(1, 2, 3)));
    assertThrows(NullPointerException.class, () -> builder.putAll(null, 1, 2, 3));
    assertThrows(NullPointerException.class, () -> builder.putAll(toPut));
  }

  @Test
  public void builderPutNullValue() {
    Multimap<String, Integer> toPut = LinkedListMultimap.create();
    toPut.put(null, 1);
    ImmutableSortedKeyListMultimap.Builder<String, Integer> builder
        = ImmutableSortedKeyListMultimap.builder();
    assertThrows(NullPointerException.class, () -> builder.put("foo", null));
    assertThrows(
        NullPointerException.class, () -> builder.putAll("foo", Arrays.asList(1, null, 3)));
    assertThrows(NullPointerException.class, () -> builder.putAll("foo", 1, null, 3));
    assertThrows(NullPointerException.class, () -> builder.putAll(toPut));
  }

  @Test
  public void copyOf() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put("foo", 1);
    input.put("bar", 2);
    input.put("foo", 3);
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertThat(input).isEqualTo(multimap);
    assertThat(multimap).isEqualTo(input);
  }

  @Test
  public void copyOfWithDuplicates() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put("foo", 1);
    input.put("bar", 2);
    input.put("foo", 3);
    input.put("foo", 1);
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertThat(input).isEqualTo(multimap);
    assertThat(multimap).isEqualTo(input);
  }

  @Test
  public void copyOfEmpty() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.copyOf(input);
    assertThat(input).isEqualTo(multimap);
    assertThat(multimap).isEqualTo(input);
  }

  @Test
  public void copyOfImmutableListMultimap() {
    Multimap<String, Integer> multimap = createMultimap();
    assertThat(ImmutableSortedKeyListMultimap.copyOf(multimap)).isSameInstanceAs(multimap);
  }

  @Test
  public void copyOfNullKey() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.put(null, 1);
    assertThrows(NullPointerException.class, () -> ImmutableSortedKeyListMultimap.copyOf(input));
  }

  @Test
  public void copyOfNullValue() {
    ListMultimap<String, Integer> input = ArrayListMultimap.create();
    input.putAll("foo", Arrays.asList(1, null, 3));
    assertThrows(NullPointerException.class, () -> ImmutableSortedKeyListMultimap.copyOf(input));
  }

  @Test
  public void emptyMultimapReads() {
    Multimap<String, Integer> multimap = ImmutableSortedKeyListMultimap.of();
    assertThat(multimap).doesNotContainKey("foo");
    assertThat(multimap.containsValue(1)).isFalse();
    assertThat(multimap).doesNotContainEntry("foo", 1);
    assertThat(multimap.entries()).isEmpty();
    assertThat(multimap.equals(ArrayListMultimap.create())).isTrue();
    assertThat(multimap).valuesForKey("foo").isEqualTo(Collections.emptyList());
    assertThat(multimap.hashCode()).isEqualTo(0);
    assertThat(multimap).isEmpty();
    assertThat(multimap.keys()).isEqualTo(HashMultiset.create());
    assertThat(multimap).isEmpty();
    assertThat(multimap).isEmpty();
    assertThat(multimap).isEmpty();
    assertThat(multimap.toString()).isEqualTo("{}");
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
    assertThat(multimap).containsKey("foo");
    assertThat(multimap).doesNotContainKey("cat");
    assertThat(multimap.containsValue(1)).isTrue();
    assertThat(multimap.containsValue(5)).isFalse();
    assertThat(multimap).containsEntry("foo", 1);
    assertThat(multimap).doesNotContainEntry("cat", 1);
    assertThat(multimap).doesNotContainEntry("foo", 5);
    assertThat(multimap.entries()).isNotEmpty();
    assertThat(multimap).hasSize(3);
    assertThat(multimap).isNotEmpty();
    assertThat(multimap.toString()).isEqualTo("{bar=[2], foo=[1, 3]}");
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
    assertThat(map).containsEntry("foo", Arrays.asList(1, 2, 3));
    assertThat(map).containsEntry("bar", Arrays.asList(4, 5));
    assertThat(map).hasSize(2);
    assertThat(map).containsKey("foo");
    assertThat(map).containsKey("bar");
    assertThat(map).doesNotContainKey("notfoo");
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
    assertThat(set).isEqualTo(other);
  }
}
