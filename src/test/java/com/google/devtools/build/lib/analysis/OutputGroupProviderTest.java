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
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.determineOutputGroups;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link OutputGroupInfo}.
 */
@RunWith(JUnit4.class)
public final class OutputGroupProviderTest {

  @Test
  public void testDetermineOutputGroupsOverridesDefaults() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("a", "b", "c"), false);
    assertThat(outputGroups).containsExactly("a", "b", "c");
  }

  @Test
  public void testDetermineOutputGroupsAddsToDefaults() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("+a"), false);
    assertThat(outputGroups).containsExactly("x", "y", "z", "a");
  }

  @Test
  public void testDetermineOutputGroupsRemovesFromDefaults() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-y"), false);
    assertThat(outputGroups).containsExactly("x", "z");
  }

  @Test
  public void testDetermineOutputGroupsMixedOverrideAdditionOverrides() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("a", "+b"), false);
    // The plain "a" causes the default output groups to be overridden.
    assertThat(outputGroups).containsExactly("a", "b");
  }

  @Test
  public void testDetermineOutputGroupsIgnoresUnknownGroup() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-foo"), false);
    // "foo" doesn't exist, but that shouldn't be a problem.
    assertThat(outputGroups).containsExactly("x", "y", "z");
  }

  @Test
  public void testDetermineOutputGroupsRemovesPreviouslyAddedGroup() throws Exception {
    Set<String> outputGroups;
    outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("+a", "-a"), false);
    assertThat(outputGroups).containsExactly("x", "y", "z");

    // Order matters here.
    outputGroups = determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-a", "+a"), false);
    assertThat(outputGroups).containsExactly("x", "y", "z", "a");
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroup() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList(), true);
    assertThat(outputGroups).containsExactly("x", "y", "z", OutputGroupInfo.VALIDATION);
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterOverride() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("foo"), true);
    assertThat(outputGroups).containsExactly("foo", OutputGroupInfo.VALIDATION);
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterAdd() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("+a"), true);
    assertThat(outputGroups).containsExactly("x", "y", "z", "a", OutputGroupInfo.VALIDATION);
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterRemove() throws Exception {
    Set<String> outputGroups =
        determineOutputGroups(ImmutableSet.of("x", "y", "z"), asList("-x"), true);
    assertThat(outputGroups).containsExactly("y", "z", OutputGroupInfo.VALIDATION);
  }
}
