// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.OutputGroupProvider.determineOutputGroups;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Set;

/**
 * Tests for {@link OutputGroupProvider}.
 */
@RunWith(JUnit4.class)
public final class OutputGroupProviderTest {

  @Test
  public void testDetermineOutputGroupsOverridesDefaults() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("a", "b", "c"));
    assertThat(outputGroups).containsExactly("a", "b", "c");
  }

  @Test
  public void testDetermineOutputGroupsAddsToDefaults() throws Exception {
    Set<String> outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("+a"));
    assertThat(outputGroups).containsExactly("x", "y", "z", "a");
  }

  @Test
  public void testDetermineOutputGroupsRemovesFromDefaults() throws Exception {
    Set<String> outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-y"));
    assertThat(outputGroups).containsExactly("x", "z");
  }

  @Test
  public void testDetermineOutputGroupsMixedOverrideAdditionOverrides() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("a", "+b"));
    // The plain "a" causes the default output groups to be overridden.
    assertThat(outputGroups).containsExactly("a", "b");
  }

  @Test
  public void testDetermineOutputGroupsIgnoresUnknownGroup() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-foo"));
    // "foo" doesn't exist, but that shouldn't be a problem.
    assertThat(outputGroups).containsExactly("x", "y", "z");
  }

  @Test
  public void testDetermineOutputGroupsRemovesPreviouslyAddedGroup() throws Exception {
    Set<String> outputGroups;
    outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("+a", "-a"));
    assertThat(outputGroups).containsExactly("x", "y", "z");

    // Order matters here.
    outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-a", "+a"));
    assertThat(outputGroups).containsExactly("x", "y", "z", "a");
  }
}
