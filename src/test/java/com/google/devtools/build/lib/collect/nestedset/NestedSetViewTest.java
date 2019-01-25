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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.collect.nestedset.NestedSetView}. */
@RunWith(JUnit4.class)
public class NestedSetViewTest {

  @Test
  public void testIdentifier() {
    NestedSet<String> inner = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder().addTransitive(inner).add("c").build();
    NestedSet<String> flat =
        NestedSetBuilder.<String>stableOrder().add("a").add("b").add("c").build();

    // The identifier should be independent of the view instance.
    assertThat(new NestedSetView<String>(inner).identifier())
        .isEqualTo(new NestedSetView<String>(inner).identifier());

    // Sets with different internal structure should have different identifiers
    assertThat(new NestedSetView<String>(flat).identifier())
        .isNotEqualTo(new NestedSetView<String>(outer).identifier());

    // Decomposing a set, the transitive sets should be correctly identified.
    Set<NestedSetView<String>> transitives = new NestedSetView<String>(outer).transitives();
    assertThat(transitives).hasSize(1);
    NestedSetView<String> extracted = transitives.iterator().next();
    assertThat(extracted.identifier()).isEqualTo(new NestedSetView<String>(inner).identifier());
  }

  @Test
  public void testDirects() {
    NestedSet<String> inner = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder()
            .add("c")
            .addTransitive(inner)
            .add("d")
            .add("e")
            .build();

    // The direct members should correctly be identified.
    assertThat(new NestedSetView<String>(outer).directs()).containsExactly("c", "d", "e");
  }

  @Test
  public void testTransitives() {
    // The inner sets have to have at least two elements, as NestedSet may decide to inline
    // singleton sets; however, we do not want to assert the inlining in the test.
    NestedSet<String> innerA = NestedSetBuilder.<String>stableOrder().add("a1").add("a2").build();
    NestedSet<String> innerB = NestedSetBuilder.<String>stableOrder().add("b1").add("b2").build();
    NestedSet<String> innerC = NestedSetBuilder.<String>stableOrder().add("c1").add("c2").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder()
            .add("x")
            .add("y")
            .addTransitive(innerA)
            .addTransitive(innerB)
            .addTransitive(innerC)
            .add("z")
            .build();

    // Decomposing the nested set, should give us the correct set of transitive members.
    ImmutableSet<Object> expected =
        ImmutableSet.of(
            new NestedSetView<String>(innerA).identifier(),
            new NestedSetView<String>(innerB).identifier(),
            new NestedSetView<String>(innerC).identifier());
    ImmutableSet.Builder<Object> found = new ImmutableSet.Builder<Object>();
    for (NestedSetView<String> transitive : new NestedSetView<String>(outer).transitives()) {
      found.add(transitive.identifier());
    }
    assertThat(found.build()).isEqualTo(expected);
  }

  /** Naively traverse a view and collect all elements reachable. */
  private static Set<String> contents(NestedSetView<String> view) {
    ImmutableSet.Builder<String> builder = new ImmutableSet.Builder<String>();
    builder.addAll(view.directs());
    for (NestedSetView<String> transitive : view.transitives()) {
      builder.addAll(contents(transitive));
    }
    return builder.build();
  }

  private static Set<Object> identifiers(Set<NestedSetView<String>> sets) {
    ImmutableSet.Builder<Object> builder = new ImmutableSet.Builder<Object>();
    for (NestedSetView<String> set : sets) {
      builder.add(set.identifier());
    }
    return builder.build();
  }

  @Test
  public void testContents() {
    // Verify that the elements reachable from view are the correct ones, regardless if singletons
    // are inlined or not. Also verify that sets with at least two elements are never inlined.
    NestedSet<String> singleA = NestedSetBuilder.<String>stableOrder().add("a").build();
    NestedSet<String> singleB = NestedSetBuilder.<String>stableOrder().add("b").build();
    NestedSet<String> multi = NestedSetBuilder.<String>stableOrder().add("c1").add("c2").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder()
            .add("x")
            .add("y")
            .addTransitive(multi)
            .addTransitive(singleA)
            .addTransitive(singleB)
            .add("z")
            .build();

    NestedSetView<String> view = new NestedSetView<String>(outer);
    assertThat(contents(view)).containsExactly("a", "b", "c1", "c2", "x", "y", "z");
    assertThat(identifiers(view.transitives()))
        .contains(new NestedSetView<String>(multi).identifier());
  }

  @Test
  public void testSplitFails() {
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSetView<String> v = new NestedSetView<>(a);
    MoreAsserts.assertThrows(
        IllegalArgumentException.class, () -> v.splitIfExceedsMaximumSize(-100));
    MoreAsserts.assertThrows(
        IllegalArgumentException.class, () -> v.splitIfExceedsMaximumSize(1));
  }

  @Test
  public void testSplitNoSplit() {
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSetView<String> v = new NestedSetView<>(a);
    assertThat(v.splitIfExceedsMaximumSize(2)).isSameAs(v);
    assertThat(v.splitIfExceedsMaximumSize(100)).isSameAs(v);
  }

  @Test
  public void testSplit() {
    NestedSet<String> a =
        NestedSetBuilder.<String>stableOrder()
            .addAll(Arrays.asList("a", "b", "c"))
            .build();
    NestedSetView<String> v = new NestedSetView<>(a);
    NestedSetView<String> s = v.splitIfExceedsMaximumSize(2);
    assertThat(s).isNotSameAs(v);
    assertThat(collectCheckSize(s, 2)).containsExactly("a", "b", "c");
  }

  @Test
  public void testRecursiveSplit() {
    NestedSet<String> a =
        NestedSetBuilder.<String>stableOrder()
            .addAll(Arrays.asList("a", "b", "c", "d", "e"))
            .build();
    NestedSetView<String> v = new NestedSetView<>(a);
    NestedSetView<String> s = v.splitIfExceedsMaximumSize(2);
    assertThat(s).isNotSameAs(v);
    assertThat(collectCheckSize(s, 2)).containsExactly("a", "b", "c", "d", "e");
  }

  private <T> List<T> collectCheckSize(NestedSetView<T> view, int maxSize) {
    return collectCheckSize(new ArrayList<>(), view, maxSize);
  }

  private <T> List<T> collectCheckSize(List<T> result, NestedSetView<T> view, int maxSize) {
    assertThat(view.directs().size()).isAtMost(maxSize);
    assertThat(view.transitives().size()).isAtMost(maxSize);
    for (NestedSetView<T> t : view.transitives()) {
      collectCheckSize(result, t, maxSize);
    }
    result.addAll(view.directs());
    return result;
  }
}
