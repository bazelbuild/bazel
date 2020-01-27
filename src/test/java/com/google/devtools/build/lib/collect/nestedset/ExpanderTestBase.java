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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Base class for tests of {@link NestedSet} iteration behavior.
 *
 * <p>This class provides test cases for representative nested set structures; the expected results
 * must be provided by overriding the corresponding methods.
 */
public abstract class ExpanderTestBase {

  /**
   * Returns the type of the expander under test.
   */
  protected abstract Order expanderOrder();

  @Test
  public void simple() {
    NestedSet<String> s = prepareBuilder("c", "a", "b").build();

    assertThat(s.toList()).isEqualTo(simpleResult());
    assertSetContents(simpleResult(), s);
  }

  @Test
  public void simpleNoDuplicates() {
    NestedSet<String> s = prepareBuilder("c", "a", "a", "a", "b").build();

    assertThat(s.toList()).isEqualTo(simpleResult());
    assertSetContents(simpleResult(), s);
  }

  @Test
  public void nesting() {
    NestedSet<String> subset = prepareBuilder("c", "a", "e").build();
    NestedSet<String> s = prepareBuilder("b", "d").addTransitive(subset).build();

    assertSetContents(nestedResult(), s);
  }

  @Test
  public void builderReuse() {
    NestedSetBuilder<String> builder = prepareBuilder();
    assertSetContents(Collections.<String>emptyList(), builder.build());

    builder.add("b");
    assertSetContents(ImmutableList.of("b"), builder.build());

    builder.addAll(ImmutableList.of("d"));
    List<String> expected = prepareBuilder("b", "d").build().toList();
    assertSetContents(expected, builder.build());

    NestedSet<String> child = prepareBuilder("c", "a", "e").build();
    builder.addTransitive(child);
    assertSetContents(nestedResult(), builder.build());
  }

  @Test
  public void builderChaining() {
    NestedSet<String> s = prepareBuilder().add("b").addAll(ImmutableList.of("d"))
        .addTransitive(prepareBuilder("c", "a", "e").build()).build();
    assertSetContents(nestedResult(), s);
  }

  @Test
  public void addAllOrdering() {
    NestedSet<String> s1 = prepareBuilder().add("a").add("c").add("b").build();
    NestedSet<String> s2 = prepareBuilder().addAll(ImmutableList.of("a", "c", "b")).build();

    assertCollectionsEqual(s1.toList(), s2.toList());
  }

  @Test
  public void mixedAddAllOrdering() {
    NestedSet<String> s1 = prepareBuilder().add("a").add("b").add("c").add("d").build();
    NestedSet<String> s2 = prepareBuilder().add("a").addAll(ImmutableList.of("b", "c")).add("d")
        .build();

    assertCollectionsEqual(s1.toList(), s2.toList());
  }

  @Test
  public void transitiveDepsHandledSeparately() {
    NestedSet<String> subset = prepareBuilder("c", "a", "e").build();
    NestedSetBuilder<String> b = prepareBuilder();
    // The fact that we add the transitive subset between the add("b") and add("d") calls should
    // not change the result.
    b.add("b");
    b.addTransitive(subset);
    b.add("d");
    NestedSet<String> s = b.build();

    assertSetContents(nestedResult(), s);
  }

  @Test
  public void nestingNoDuplicates() {
    NestedSet<String> subset = prepareBuilder("c", "a", "e").build();
    NestedSet<String> s = prepareBuilder("b", "d", "e").addTransitive(subset).build();

    assertSetContents(nestedDuplicatesResult(), s);
  }

  @Test
  public void chain() {
    NestedSet<String> c = prepareBuilder("c").build();
    NestedSet<String> b = prepareBuilder("b").addTransitive(c).build();
    NestedSet<String> a = prepareBuilder("a").addTransitive(b).build();

    assertSetContents(chainResult(), a);
  }

  @Test
  public void diamond() {
    NestedSet<String> d = prepareBuilder("d").build();
    NestedSet<String> c = prepareBuilder("c").addTransitive(d).build();
    NestedSet<String> b = prepareBuilder("b").addTransitive(d).build();
    NestedSet<String> a = prepareBuilder("a").addTransitive(b).addTransitive(c).build();

    assertSetContents(diamondResult(), a);
  }

  @Test
  public void extendedDiamond() {
    NestedSet<String> d = prepareBuilder("d").build();
    NestedSet<String> e = prepareBuilder("e").build();
    NestedSet<String> b = prepareBuilder("b").addTransitive(d).addTransitive(e).build();
    NestedSet<String> c = prepareBuilder("c").addTransitive(e).addTransitive(d).build();
    NestedSet<String> a = prepareBuilder("a").addTransitive(b).addTransitive(c).build();
    assertSetContents(extendedDiamondResult(), a);
  }

  @Test
  public void extendedDiamondRightArm() {
    NestedSet<String> d = prepareBuilder("d").build();
    NestedSet<String> e = prepareBuilder("e").build();
    NestedSet<String> b = prepareBuilder("b").addTransitive(d).addTransitive(e).build();
    NestedSet<String> c2 = prepareBuilder("c2").addTransitive(e).addTransitive(d).build();
    NestedSet<String> c = prepareBuilder("c").addTransitive(c2).build();
    NestedSet<String> a = prepareBuilder("a").addTransitive(b).addTransitive(c).build();
    assertSetContents(extendedDiamondRightArmResult(), a);
  }

  @Test
  public void orderConflict() {
    NestedSet<String> child1 = prepareBuilder("a", "b").build();
    NestedSet<String> child2 = prepareBuilder("b", "a").build();
    NestedSet<String> parent = prepareBuilder().addTransitive(child1).addTransitive(child2).build();
    assertSetContents(orderConflictResult(), parent);
  }

  @Test
  public void orderConflictNested() {
    NestedSet<String> a = prepareBuilder("a").build();
    NestedSet<String> b = prepareBuilder("b").build();
    NestedSet<String> child1 = prepareBuilder().addTransitive(a).addTransitive(b).build();
    NestedSet<String> child2 = prepareBuilder().addTransitive(b).addTransitive(a).build();
    NestedSet<String> parent = prepareBuilder().addTransitive(child1).addTransitive(child2).build();
    assertSetContents(orderConflictResult(), parent);
  }

  @Test
  public void getOrderingEmpty() {
    NestedSet<String> s = prepareBuilder().build();
    assertThat(s.isEmpty()).isTrue();
    assertThat(s.getOrder()).isEqualTo(expanderOrder());
  }

  @Test
  public void getOrdering() {
    NestedSet<String> s = prepareBuilder("a", "b").build();
    assertThat(s.isEmpty()).isFalse();
    assertThat(s.getOrder()).isEqualTo(expanderOrder());
  }

  @Test
  public void nestingValidation() {
    for (Order ordering : Order.values()) {
      NestedSet<String> a = prepareBuilder("a", "b").build();
      NestedSetBuilder<String> b = new NestedSetBuilder<>(ordering);
      try {
        b.addTransitive(a);
        if (ordering != expanderOrder() && ordering != Order.STABLE_ORDER) {
          fail();  // An exception was expected.
        }
      } catch (IllegalArgumentException e) {
        if (ordering == expanderOrder() || ordering == Order.STABLE_ORDER) {
          fail();  // No exception was expected.
        }
      }
    }
  }

  private NestedSetBuilder<String> prepareBuilder(String... directMembers) {
    NestedSetBuilder<String> builder = new NestedSetBuilder<>(expanderOrder());
    builder.addAll(Lists.newArrayList(directMembers));
    return builder;
  }

  protected final void assertSetContents(List<String> expected, NestedSet<String> set) {
    assertThat(Lists.newArrayList(set.toList())).isEqualTo(expected);
    assertThat(Lists.newArrayList(set.toSet())).isEqualTo(expected);
  }

  protected final void assertCollectionsEqual(
      Collection<String> expected, Collection<String> actual) {
    assertThat(Lists.newArrayList(actual)).isEqualTo(Lists.newArrayList(expected));
  }

  /**
   * Returns the enumeration of the nested set {"c", "a", "b"} in the implementation's enumeration
   * order.
   *
   * @see #simple()
   * @see #simpleNoDuplicates()
   */
  protected List<String> simpleResult() {
    return ImmutableList.of("c", "a", "b");
  }

  /**
   * Returns the enumeration of the nested set {"b", "d", {"c", "a", "e"}} in the implementation's
   * enumeration order.
   *
   * @see #nesting()
   */
  protected abstract List<String> nestedResult();

  /**
   * Returns the enumeration of the nested set {"b", "d", "e", {"c", "a", "e"}} in the
   * implementation's enumeration order.
   *
   * @see #nestingNoDuplicates()
   */
  protected abstract List<String> nestedDuplicatesResult();

  /**
   * Returns the enumeration of nested set {"a", {"b", {"c"}}} in the implementation's enumeration
   * order.
   *
   * @see #chain()
   */
  protected abstract List<String> chainResult();

  /**
   * Returns the enumeration of the nested set {"a", {"b", D}, {"c", D}}, where D is {"d"}, in the
   * implementation's enumeration order.
   *
   * @see #diamond()
   */
  protected abstract List<String> diamondResult();

  /**
   * Returns the enumeration of the nested set {"a", {"b", E, D}, {"c", D, E}}, where D is {"d"} and
   * E is {"e"}, in the implementation's enumeration order.
   *
   * @see #extendedDiamond()
   */
  protected abstract List<String> extendedDiamondResult();

  /**
   * Returns the enumeration of the nested set {"a", {"b", E, D}, {"c", C2}}, where D is {"d"}, E is
   * {"e"} and C2 is {"c2", D, E}, in the implementation's enumeration order.
   *
   * @see #extendedDiamondRightArm()
   */
  protected abstract List<String> extendedDiamondRightArmResult();

  /**
   * Returns the enumeration of the nested set {{"a", "b"}, {"b", "a"}}.
   *
   * @see #orderConflict()
   * @see #orderConflictNested()
   */
  protected List<String> orderConflictResult() {
    return ImmutableList.of("a", "b");
  }
}
