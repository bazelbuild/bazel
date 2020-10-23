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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of NestedSet topology methods: toNode, getNonLeaves, getLeaves. */
@RunWith(JUnit4.class)
public class NestedSetTopologyTest {

  @Test
  public void testToNode() {
    NestedSet<String> inner = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder().addTransitive(inner).add("c").build();
    NestedSet<String> flat =
        NestedSetBuilder.<String>stableOrder().add("a").add("b").add("c").build();

    assertThat(inner.toNode()).isEqualTo(inner.toNode());

    // Sets with different internal structure should have different nodes
    assertThat(flat.toNode()).isNotEqualTo(outer.toNode());

    // Decomposing a set, the transitive sets should be correctly identified.
    List<NestedSet<String>> succs = outer.getNonLeaves();
    assertThat(succs).hasSize(1);
    NestedSet<String> succ0 = succs.get(0);
    assertThat(succ0.toNode()).isEqualTo(inner.toNode());
  }

  @Test
  public void testGetLeaves() {
    NestedSet<String> inner = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    NestedSet<String> outer =
        NestedSetBuilder.<String>stableOrder()
            .add("c")
            .addTransitive(inner)
            .add("d")
            .add("e")
            .build();

    // The direct members should correctly be identified.
    assertThat(outer.getLeaves()).containsExactly("c", "d", "e");
  }

  @Test
  public void testGetNonLeaves() {
    // The inner sets must have at least two elements, as NestedSet inlines singleton sets.
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

    // Decomposing the nested set should give us the correct list of transitive members.
    // Compare using strings as NestedSet.equals uses identity.
    assertThat(outer.getNonLeaves().toString())
        .isEqualTo(ImmutableList.of(innerA, innerB, innerC).toString());
  }

  /** Naively traverse a view and collect all elements reachable. */
  private static ImmutableSet<String> contents(NestedSet<String> set) {
    ImmutableSet.Builder<String> builder = new ImmutableSet.Builder<String>();
    builder.addAll(set.getLeaves());
    for (NestedSet<String> nonleaf : set.getNonLeaves()) {
      builder.addAll(contents(nonleaf));
    }
    return builder.build();
  }

  private static ImmutableSet<Object> nodes(Collection<NestedSet<String>> sets) {
    ImmutableSet.Builder<Object> builder = new ImmutableSet.Builder<Object>();
    for (NestedSet<String> set : sets) {
      builder.add(set.toNode());
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

    assertThat(contents(outer)).containsExactly("a", "b", "c1", "c2", "x", "y", "z");
    assertThat(nodes(outer.getNonLeaves())).contains(multi.toNode());
  }

  @Test
  public void testSplitFails() {
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    assertThrows(IllegalArgumentException.class, () -> a.splitIfExceedsMaximumSize(-100));
    assertThrows(IllegalArgumentException.class, () -> a.splitIfExceedsMaximumSize(1));
  }

  @Test
  public void testSplitNoSplit() {
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    assertThat(a.splitIfExceedsMaximumSize(2)).isSameInstanceAs(a);
    assertThat(a.splitIfExceedsMaximumSize(100)).isSameInstanceAs(a);
  }

  @Test
  public void testSplit() {
    NestedSet<String> a =
        NestedSetBuilder.<String>stableOrder()
            .addAll(Arrays.asList("a", "b", "c"))
            .build();
    NestedSet<String> v = a;
    NestedSet<String> s = v.splitIfExceedsMaximumSize(2);
    assertThat(s).isNotSameInstanceAs(v);
    assertThat(collectCheckSize(s, 2)).containsExactly("a", "b", "c");
  }

  @Test
  public void testRecursiveSplit() {
    NestedSet<String> a =
        NestedSetBuilder.<String>stableOrder()
            .addAll(Arrays.asList("a", "b", "c", "d", "e"))
            .build();
    NestedSet<String> v = a;
    NestedSet<String> s = v.splitIfExceedsMaximumSize(2);
    assertThat(s).isNotSameInstanceAs(v);
    assertThat(collectCheckSize(s, 2)).containsExactly("a", "b", "c", "d", "e");

    // Splitting may increment the graph depth, possibly more than once.
    assertThat(v.getApproxDepth()).isEqualTo(2);
    assertThat(s.getApproxDepth()).isEqualTo(4);
  }

  private static <T> List<T> collectCheckSize(NestedSet<T> set, int maxSize) {
    return collectCheckSize(new ArrayList<>(), set, maxSize);
  }

  private static <T> List<T> collectCheckSize(List<T> result, NestedSet<T> set, int maxSize) {
    assertThat(set.getLeaves().size()).isAtMost(maxSize);
    assertThat(set.getNonLeaves().size()).isAtMost(maxSize);
    for (NestedSet<T> nonleaf : set.getNonLeaves()) {
      collectCheckSize(result, nonleaf, maxSize);
    }
    result.addAll(set.getLeaves());
    return result;
  }
}
