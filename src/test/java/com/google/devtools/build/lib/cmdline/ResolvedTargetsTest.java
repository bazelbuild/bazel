// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ResolvedTargets}. */
@RunWith(JUnit4.class)
public class ResolvedTargetsTest {

  private static ResolvedTargets.Builder<String> builderWith(List<String> targets) {
    ResolvedTargets.Builder<String> builder = ResolvedTargets.builder();
    builder.addAll(targets);
    return builder;
  }

  /** Checks that {@code filterOut} leaves what {@code filter(not(in(...)))} leaves, in order. */
  private static void assertFiltersOutLikeFilter(List<String> targets, List<String> toRemove) {
    ResolvedTargets<String> expected =
        builderWith(targets).filter(Predicates.not(Predicates.in(toRemove))).build();

    ResolvedTargets<String> actual = builderWith(targets).filterOut(toRemove).build();

    assertThat(actual.getTargets()).containsExactlyElementsIn(expected.getTargets()).inOrder();
    assertThat(actual.getFilteredTargets())
        .containsExactlyElementsIn(expected.getFilteredTargets())
        .inOrder();
  }

  @Test
  public void filterOut_oneTarget() {
    assertFiltersOutLikeFilter(ImmutableList.of("a", "b", "c"), ImmutableList.of("b"));
  }

  @Test
  public void filterOut_severalTargets_recordedInTheOrderOfTheTargets() {
    assertFiltersOutLikeFilter(
        ImmutableList.of("a", "b", "c", "d"), ImmutableList.of("d", "a", "c"));

    ResolvedTargets<String> result =
        builderWith(ImmutableList.of("a", "b", "c", "d"))
            .filterOut(ImmutableList.of("d", "a", "c"))
            .build();
    assertThat(result.getTargets()).containsExactly("b");
    assertThat(result.getFilteredTargets()).containsExactly("a", "c", "d").inOrder();
  }

  @Test
  public void filterOut_targetThatIsNotPresent_isNotRecorded() {
    assertFiltersOutLikeFilter(ImmutableList.of("a", "b"), ImmutableList.of("x"));
    assertFiltersOutLikeFilter(ImmutableList.of("a", "b"), ImmutableList.of("x", "b", "y"));

    ResolvedTargets<String> result =
        builderWith(ImmutableList.of("a", "b")).filterOut(ImmutableList.of("x")).build();
    assertThat(result.getFilteredTargets()).isEmpty();
  }

  @Test
  public void filterOut_everything() {
    assertFiltersOutLikeFilter(ImmutableList.of("a", "b"), ImmutableList.of("b", "a"));
  }

  @Test
  public void filterOut_nothing() {
    assertFiltersOutLikeFilter(ImmutableList.of("a", "b"), ImmutableList.of());
  }

  @Test
  public void filterOut_afterAnEarlierRemoval_keepsTheEarlierOnes() {
    ResolvedTargets<String> result =
        builderWith(ImmutableList.of("a", "b", "c"))
            .filterOut(ImmutableList.of("c"))
            .filterOut(ImmutableList.of("a"))
            .build();

    assertThat(result.getTargets()).containsExactly("b");
    assertThat(result.getFilteredTargets()).containsExactly("c", "a").inOrder();
  }
}
