// Copyright 2021 The Bazel Authors. All rights reserved.
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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link NestedSetVisitor}. */
@RunWith(JUnit4.class)
public final class NestedSetVisitorTest {

  @Test
  public void stableOrder() throws InterruptedException {
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>stableOrder()
            .addTransitive(NestedSetBuilder.<Integer>stableOrder().add(1).add(2).add(3).build())
            .add(4)
            .add(5)
            .add(6)
            .addTransitive(NestedSetBuilder.<Integer>stableOrder().add(7).add(8).add(9).build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    assertThat(visited).isEqualTo(set.toList());
  }

  @Test
  public void compileOrder() throws InterruptedException {
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>compileOrder()
            .addTransitive(NestedSetBuilder.<Integer>compileOrder().add(1).add(2).add(3).build())
            .add(4)
            .add(5)
            .add(6)
            .addTransitive(NestedSetBuilder.<Integer>compileOrder().add(7).add(8).add(9).build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    assertThat(visited).isEqualTo(set.toList());
  }

  @Test
  public void linkOrder() throws InterruptedException {
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>linkOrder()
            .addTransitive(NestedSetBuilder.<Integer>linkOrder().add(1).add(2).add(3).build())
            .add(4)
            .add(5)
            .add(6)
            .addTransitive(NestedSetBuilder.<Integer>linkOrder().add(7).add(8).add(9).build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    // #toList() for LINK_ORDER reverses the result list.
    assertThat(visited).isEqualTo(set.toList().reverse());
  }

  @Test
  public void naiveLinkOrder() throws InterruptedException {
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>naiveLinkOrder()
            .addTransitive(NestedSetBuilder.<Integer>naiveLinkOrder().add(1).add(2).add(3).build())
            .add(4)
            .add(5)
            .add(6)
            .addTransitive(NestedSetBuilder.<Integer>naiveLinkOrder().add(7).add(8).add(9).build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    assertThat(visited).isEqualTo(set.toList());
  }

  @Test
  public void mixedOrders() throws InterruptedException {
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>stableOrder()
            .addTransitive(
                NestedSetBuilder.<Integer>linkOrder()
                    .add(1)
                    .addTransitive(NestedSetBuilder.<Integer>linkOrder().add(2).add(3).build())
                    .addTransitive(NestedSetBuilder.<Integer>linkOrder().add(4).add(5).build())
                    .build())
            .add(6)
            .add(7)
            .add(8)
            .addTransitive(
                NestedSetBuilder.<Integer>naiveLinkOrder()
                    .addTransitive(NestedSetBuilder.<Integer>naiveLinkOrder().add(7).add(8).build())
                    .addTransitive(
                        NestedSetBuilder.<Integer>naiveLinkOrder().add(9).add(10).build())
                    .add(11)
                    .build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    assertThat(visited).isEqualTo(set.toList());
  }

  @Test
  public void duplicatesSkipped() throws InterruptedException {
    NestedSet<Integer> subset =
        NestedSetBuilder.<Integer>compileOrder().add(1).add(2).add(3).build();
    NestedSet<Integer> set =
        NestedSetBuilder.<Integer>compileOrder()
            .addTransitive(subset)
            .addTransitive(
                NestedSetBuilder.<Integer>compileOrder().add(4).addTransitive(subset).build())
            .add(5)
            .add(6)
            .add(7)
            .addTransitive(
                NestedSetBuilder.<Integer>compileOrder()
                    .add(8)
                    .add(9)
                    .addTransitive(subset)
                    .build())
            .build();

    List<Integer> visited = new ArrayList<>();
    new NestedSetVisitor<Integer>(visited::add, new HashSet<>()::add).visit(set);

    assertThat(visited).isEqualTo(set.toList());
    assertThat(visited).hasSize(9);
  }
}
