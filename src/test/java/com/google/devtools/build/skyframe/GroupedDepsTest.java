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
package com.google.devtools.build.skyframe;

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

/** Tests for {@link GroupedDeps}. */
@RunWith(TestParameterInjector.class)
public final class GroupedDepsTest {

  @TestParameter private boolean withHashSet;

  private GroupedDeps createEmpty() {
    if (withHashSet) {
      return new GroupedDeps.WithHashSet();
    }
    return new GroupedDeps();
  }

  private static SkyKey key(String arg) {
    return GraphTester.skyKey(arg);
  }

  @Test
  public void singleGroup(@TestParameter({"0", "1", "2", "10"}) int size) {
    GroupedDeps deps = createEmpty();
    ImmutableList<SkyKey> elements =
        IntStream.range(0, size).mapToObj(i -> key(String.valueOf(i))).collect(toImmutableList());
    deps.appendGroup(elements);
    checkGroups(deps, size == 0 ? ImmutableList.of() : ImmutableList.of(elements));
  }

  @Test
  public void appendEmptyGroup_noOp() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of());
    assertThat(deps.isEmpty()).isTrue();
    deps.appendSingleton(key("a"));
    deps.appendGroup(ImmutableList.of());
    deps.appendSingleton(key("b"));
    checkGroups(deps, ImmutableList.of(ImmutableList.of(key("a")), ImmutableList.of(key("b"))));
  }

  @Test
  public void identical_equal() {
    GroupedDeps abc1 = createEmpty();
    GroupedDeps abc2 = createEmpty();
    abc1.appendGroup(ImmutableList.of(key("a"), key("b"), key("c")));
    abc2.appendGroup(ImmutableList.of(key("a"), key("b"), key("c")));
    assertThat(abc1).isEqualTo(abc2);
  }

  @Test
  public void appendSingletonAndAppendGroupSizeOne_equal() {
    GroupedDeps aSingleton = createEmpty();
    GroupedDeps aGroup = createEmpty();
    aSingleton.appendSingleton(key("a"));
    aGroup.appendGroup(ImmutableList.of(key("a")));
    assertThat(aSingleton).isEqualTo(aGroup);
  }

  @Test
  public void differentOrderWithinGroup_equal() {
    GroupedDeps ab = createEmpty();
    GroupedDeps ba = createEmpty();
    ab.appendGroup(ImmutableList.of(key("a"), key("b")));
    ba.appendGroup(ImmutableList.of(key("b"), key("a")));
    assertThat(ab).isEqualTo(ba);
  }

  @Test
  public void differentElements_notEqual() {
    GroupedDeps abc = createEmpty();
    GroupedDeps xyz = createEmpty();
    abc.appendGroup(ImmutableList.of(key("a"), key("b"), key("c")));
    xyz.appendGroup(ImmutableList.of(key("x"), key("y"), key("z")));
    assertThat(abc).isNotEqualTo(xyz);
  }

  @Test
  public void differentOrderOfGroups_notEqual() {
    GroupedDeps ab = createEmpty();
    GroupedDeps ba = createEmpty();
    ab.appendSingleton(key("a"));
    ab.appendSingleton(key("b"));
    ba.appendSingleton(key("b"));
    ba.appendSingleton(key("a"));
    assertThat(ab).isNotEqualTo(ba);
  }

  @Test
  public void differentGroupings_notEqual() {
    GroupedDeps abGroup = createEmpty();
    GroupedDeps abSingletons = createEmpty();
    abGroup.appendGroup(ImmutableList.of(key("a"), key("b")));
    abSingletons.appendSingleton(key("a"));
    abSingletons.appendSingleton(key("b"));
    assertThat(abGroup).isNotEqualTo(abSingletons);
  }

  @Test
  public void groups() {
    GroupedDeps deps = createEmpty();
    ImmutableList<ImmutableList<SkyKey>> groups =
        ImmutableList.of(
            ImmutableList.of(key("1")),
            ImmutableList.of(key("2a"), key("2b")),
            ImmutableList.of(key("3")),
            ImmutableList.of(key("4")),
            ImmutableList.of(key("5a"), key("5b"), key("5c")),
            ImmutableList.of(key("6a"), key("6b"), key("6c")));
    groups.forEach(deps::appendGroup);
    checkGroups(deps, groups);
  }

  @Test
  public void remove_groupsIntact() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendGroup(ImmutableList.of(key("2a"), key("2b"), key("2c")));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    deps.remove(ImmutableSet.of(key("2c")));

    checkGroups(
        deps,
        ImmutableList.of(
            ImmutableList.of(key("1a"), key("1b")),
            ImmutableList.of(key("2a"), key("2b")),
            ImmutableList.of(key("3a"), key("3b"))));
  }

  @Test
  public void remove_groupBecomesSingleton() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendGroup(ImmutableList.of(key("2a"), key("2b"), key("2c")));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    deps.remove(ImmutableSet.of(key("2b"), key("2c")));

    checkGroups(
        deps,
        ImmutableList.of(
            ImmutableList.of(key("1a"), key("1b")),
            ImmutableList.of(key("2a")),
            ImmutableList.of(key("3a"), key("3b"))));
  }

  @Test
  public void remove_groupBecomesEmpty() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendGroup(ImmutableList.of(key("2a"), key("2b"), key("2c")));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    deps.remove(ImmutableSet.of(key("2a"), key("2b"), key("2c")));

    checkGroups(
        deps,
        ImmutableList.of(
            ImmutableList.of(key("1a"), key("1b")), ImmutableList.of(key("3a"), key("3b"))));
  }

  @Test
  public void remove_singleton() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendSingleton(key("2"));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    deps.remove(ImmutableSet.of(key("2")));

    checkGroups(
        deps,
        ImmutableList.of(
            ImmutableList.of(key("1a"), key("1b")), ImmutableList.of(key("3a"), key("3b"))));
  }

  @Test
  public void remove_wholeGroupedDepsBecomesEmpty() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendGroup(ImmutableList.of(key("2a"), key("2b"), key("2c")));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    deps.remove(
        ImmutableSet.of(
            key("1a"), key("1b"), key("2a"), key("2b"), key("2c"), key("3a"), key("3b")));

    checkGroups(deps, ImmutableList.of());
  }

  @Test
  public void remove_elementNotPresent_throws() {
    GroupedDeps deps = createEmpty();
    deps.appendGroup(ImmutableList.of(key("1a"), key("1b")));
    deps.appendGroup(ImmutableList.of(key("2a"), key("2b"), key("2c")));
    deps.appendGroup(ImmutableList.of(key("3a"), key("3b")));

    assertThrows(RuntimeException.class, () -> deps.remove(ImmutableSet.of(key("2c"), key("2d"))));
  }

  private static void checkGroups(GroupedDeps deps, List<ImmutableList<SkyKey>> expectedGroups) {
    assertThat(deps.isEmpty()).isEqualTo(expectedGroups.isEmpty());
    assertThat(deps.numGroups()).isEqualTo(expectedGroups.size());
    assertThat(deps).containsExactlyElementsIn(expectedGroups).inOrder();

    ImmutableList<SkyKey> expectedFlattened =
        ImmutableList.copyOf(Iterables.concat(expectedGroups));
    assertThat(deps.numElements()).isEqualTo(expectedFlattened.size());
    assertThat(deps.getAllElementsAsIterable())
        .containsExactlyElementsIn(expectedFlattened)
        .inOrder();
    if (deps instanceof GroupedDeps.WithHashSet) {
      assertThat(deps.toSet()).containsExactlyElementsIn(expectedFlattened);
    } else {
      assertThat(deps.toSet()).containsExactlyElementsIn(expectedFlattened).inOrder();
    }

    checkCompression(deps);
  }

  private static void checkCompression(GroupedDeps deps) {
    @GroupedDeps.Compressed Object compressed = deps.compress();
    assertThat(GroupedDeps.numElements(compressed)).isEqualTo(deps.numElements());
    assertThat(GroupedDeps.isEmpty(compressed)).isEqualTo(deps.isEmpty());
    assertThat(GroupedDeps.compressedToIterable(compressed))
        .containsExactlyElementsIn(deps.getAllElementsAsIterable())
        .inOrder();
    assertThat(GroupedDeps.decompress(compressed)).containsExactlyElementsIn(deps).inOrder();
    assertThat(GroupedDeps.decompress(compressed)).isNotInstanceOf(GroupedDeps.WithHashSet.class);
  }
}
