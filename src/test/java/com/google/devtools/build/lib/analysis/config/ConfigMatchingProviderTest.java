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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ConfigMatchingProvider}, focused on refinement logic. */
@RunWith(JUnit4.class)
public class ConfigMatchingProviderTest {

  private static Label label(String labelStr) {
    return Label.parseCanonicalUnchecked(labelStr);
  }

  @Test
  public void refines_supersetConstraintValues() {
    // A with 2 constraint values refines B with 1 constraint value (standard superset).
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:x"), label("//cv:y")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:x")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isTrue();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_hierarchicalConstraintValues() {
    // A has glibc_2_42 (which refines glibc), B has glibc.
    // A should refine B via hierarchical constraint.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc/glibc:2_42")),
            ImmutableMap.of(label("//libc/glibc:2_42"), label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isTrue();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_hierarchicalWithSharedConstraint() {
    // A: {glibc_2_42, x86} where glibc_2_42 refines glibc.
    // B: {glibc, x86}
    // A refines B because glibc_2_42 hierarchically covers glibc and x86 matches directly.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc/glibc:2_42"), label("//cpu:x86")),
            ImmutableMap.of(label("//libc/glibc:2_42"), label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc:glibc"), label("//cpu:x86")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isTrue();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_hierarchicalDoesNotRefineUnrelated() {
    // glibc_2_42 refines glibc, but NOT musl.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc/glibc:2_42")),
            ImmutableMap.of(label("//libc/glibc:2_42"), label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc:musl")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isFalse();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_hierarchicalPlusExtraFlag() {
    // A has glibc_2_42 (refines glibc) AND a native flag.
    // B has glibc only.
    // A refines B (hierarchical constraint + additional flag).
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of("compilation_mode", "dbg"),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc/glibc:2_42")),
            ImmutableMap.of(label("//libc/glibc:2_42"), label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//libc:glibc")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isTrue();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_identicalConstraints_noRefinement() {
    // Same constraint values, same sizes. Neither refines the other.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:x")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:x")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isFalse();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_disjointConstraints_noRefinement() {
    // Completely different constraint values. Neither refines.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:x")),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(label("//cv:y")),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isFalse();
    assertThat(b.refines(a)).isFalse();
  }

  @Test
  public void refines_emptyConstraints_noRefinement() {
    // Both empty. Neither refines.
    ConfigMatchingProvider a =
        ConfigMatchingProvider.create(
            label("//test:a"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(),
            ConfigMatchingProvider.MatchResult.MATCH);
    ConfigMatchingProvider b =
        ConfigMatchingProvider.create(
            label("//test:b"),
            ImmutableMultimap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(),
            ConfigMatchingProvider.MatchResult.MATCH);
    assertThat(a.refines(b)).isFalse();
    assertThat(b.refines(a)).isFalse();
  }
}
