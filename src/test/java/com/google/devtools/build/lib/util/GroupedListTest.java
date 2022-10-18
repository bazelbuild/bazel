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
package com.google.devtools.build.lib.util;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link GroupedList}. */
@RunWith(TestParameterInjector.class)
public final class GroupedListTest {

  @Test
  public void singleGroup(@TestParameter({"0", "1", "2", "10"}) int size) {
    GroupedList<String> groupedList = new GroupedList<>();
    ImmutableList<String> elements =
        IntStream.range(0, size).mapToObj(String::valueOf).collect(toImmutableList());
    groupedList.appendGroup(elements);
    checkGroups(groupedList, size == 0 ? ImmutableList.of() : ImmutableList.of(elements));
  }

  @Test
  public void appendEmptyGroup_noOp() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of());
    assertThat(groupedList.isEmpty()).isTrue();
    groupedList.appendSingleton("a");
    groupedList.appendGroup(ImmutableList.of());
    groupedList.appendSingleton("b");
    checkGroups(groupedList, ImmutableList.of(ImmutableList.of("a"), ImmutableList.of("b")));
  }

  @Test
  public void identical_equal() {
    GroupedList<String> abc1 = new GroupedList<>();
    GroupedList<String> abc2 = new GroupedList<>();
    abc1.appendGroup(ImmutableList.of("a", "b", "c"));
    abc2.appendGroup(ImmutableList.of("a", "b", "c"));
    assertThat(abc1).isEqualTo(abc2);
  }

  @Test
  public void appendSingletonAndAppendGroupSizeOne_equal() {
    GroupedList<String> aSingleton = new GroupedList<>();
    GroupedList<String> aGroup = new GroupedList<>();
    aSingleton.appendSingleton("a");
    aGroup.appendGroup(ImmutableList.of("a"));
    assertThat(aSingleton).isEqualTo(aGroup);
  }

  @Test
  public void differentOrderWithinGroup_equal() {
    GroupedList<String> ab = new GroupedList<>();
    GroupedList<String> ba = new GroupedList<>();
    ab.appendGroup(ImmutableList.of("a", "b"));
    ba.appendGroup(ImmutableList.of("b", "a"));
    assertThat(ab).isEqualTo(ba);
  }

  @Test
  public void differentElements_notEqual() {
    GroupedList<String> abc = new GroupedList<>();
    GroupedList<String> xyz = new GroupedList<>();
    abc.appendGroup(ImmutableList.of("a", "b", "c"));
    xyz.appendGroup(ImmutableList.of("x", "y", "z"));
    assertThat(abc).isNotEqualTo(xyz);
  }

  @Test
  public void differentOrderOfGroups_notEqual() {
    GroupedList<String> ab = new GroupedList<>();
    GroupedList<String> ba = new GroupedList<>();
    ab.appendSingleton("a");
    ab.appendSingleton("b");
    ba.appendSingleton("b");
    ba.appendSingleton("a");
    assertThat(ab).isNotEqualTo(ba);
  }

  @Test
  public void differentGroupings_notEqual() {
    GroupedList<String> abGroup = new GroupedList<>();
    GroupedList<String> abSingletons = new GroupedList<>();
    abGroup.appendGroup(ImmutableList.of("a", "b"));
    abSingletons.appendSingleton("a");
    abSingletons.appendSingleton("b");
    assertThat(abGroup).isNotEqualTo(abSingletons);
  }

  @Test
  public void groups() {
    GroupedList<String> groupedList = new GroupedList<>();
    ImmutableList<ImmutableList<String>> groups =
        ImmutableList.of(
            ImmutableList.of("1"),
            ImmutableList.of("2a", "2b"),
            ImmutableList.of("3"),
            ImmutableList.of("4"),
            ImmutableList.of("5a", "5b", "5c"),
            ImmutableList.of("6a", "6b", "6c"));
    groups.forEach(groupedList::appendGroup);
    checkGroups(groupedList, groups);
  }

  @Test
  public void remove_groupsIntact() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendGroup(ImmutableList.of("2a", "2b", "2c"));
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    groupedList.remove(ImmutableSet.of("2c"));

    checkGroups(
        groupedList,
        ImmutableList.of(
            ImmutableList.of("1a", "1b"),
            ImmutableList.of("2a", "2b"),
            ImmutableList.of("3a", "3b")));
  }

  @Test
  public void remove_groupBecomesSingleton() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendGroup(ImmutableList.of("2a", "2b", "2c"));
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    groupedList.remove(ImmutableSet.of("2b", "2c"));

    checkGroups(
        groupedList,
        ImmutableList.of(
            ImmutableList.of("1a", "1b"), ImmutableList.of("2a"), ImmutableList.of("3a", "3b")));
  }

  @Test
  public void remove_groupBecomesEmpty() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendGroup(ImmutableList.of("2a", "2b", "2c"));
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    groupedList.remove(ImmutableSet.of("2a", "2b", "2c"));

    checkGroups(
        groupedList, ImmutableList.of(ImmutableList.of("1a", "1b"), ImmutableList.of("3a", "3b")));
  }

  @Test
  public void remove_singleton() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendSingleton("2");
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    groupedList.remove(ImmutableSet.of("2"));

    checkGroups(
        groupedList, ImmutableList.of(ImmutableList.of("1a", "1b"), ImmutableList.of("3a", "3b")));
  }

  @Test
  public void remove_wholeGroupedListBecomesEmpty() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendGroup(ImmutableList.of("2a", "2b", "2c"));
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    groupedList.remove(ImmutableSet.of("1a", "1b", "2a", "2b", "2c", "3a", "3b"));

    checkGroups(groupedList, ImmutableList.of());
  }

  @Test
  public void remove_elementNotPresent_throws() {
    GroupedList<String> groupedList = new GroupedList<>();
    groupedList.appendGroup(ImmutableList.of("1a", "1b"));
    groupedList.appendGroup(ImmutableList.of("2a", "2b", "2c"));
    groupedList.appendGroup(ImmutableList.of("3a", "3b"));

    assertThrows(RuntimeException.class, () -> groupedList.remove(ImmutableSet.of("2c", "2d")));
  }

  private static void checkGroups(
      GroupedList<String> groupedList, List<ImmutableList<String>> expectedGroups) {
    assertThat(groupedList.isEmpty()).isEqualTo(expectedGroups.isEmpty());
    assertThat(groupedList.listSize()).isEqualTo(expectedGroups.size());
    assertThat(groupedList).containsExactlyElementsIn(expectedGroups).inOrder();

    ImmutableList<String> expectedFlattened =
        ImmutableList.copyOf(Iterables.concat(expectedGroups));
    assertThat(groupedList.numElements()).isEqualTo(expectedFlattened.size());
    assertThat(groupedList.getAllElementsAsIterable())
        .containsExactlyElementsIn(expectedFlattened)
        .inOrder();
    assertThat(groupedList.toSet()).containsExactlyElementsIn(expectedFlattened).inOrder();

    checkCompression(groupedList);
  }

  private static void checkCompression(GroupedList<String> groupedList) {
    @GroupedList.Compressed Object compressed = groupedList.compress();
    assertThat(GroupedList.numElements(compressed)).isEqualTo(groupedList.numElements());
    assertThat(GroupedList.numGroups(compressed)).isEqualTo(groupedList.listSize());
    assertThat(GroupedList.compressedToIterable(compressed))
        .containsExactlyElementsIn(groupedList.getAllElementsAsIterable())
        .inOrder();
    assertThat(GroupedList.create(compressed)).containsExactlyElementsIn(groupedList).inOrder();
  }
}
