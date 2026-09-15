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
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.DEFAULT;
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.determineOutputGroups;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.OutputGroupInfo.ValidationMode;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link OutputGroupInfo}. */
@RunWith(TestParameterInjector.class)
public final class OutputGroupProviderTest {

  @Test
  public void testDetermineOutputGroupsOverridesDefaults(@TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("a", "b", "c"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(shouldRunTests, "a", "b", "c"));
  }

  @Test
  public void testDetermineOutputGroupsAddsToDefaults(@TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("+a"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(shouldRunTests, "x", "y", "z", "a"));
  }

  @Test
  public void testDetermineOutputGroupsRemovesFromDefaults(@TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-y"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(outputGroupsWithDefaultIfRunningTests(shouldRunTests, "x", "z"));
  }

  @Test
  public void testDetermineOutputGroupsMixedOverrideAdditionOverrides(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("a", "+b"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    // The plain "a" causes the default output groups to be overridden.
    assertThat(outputGroups)
        .containsExactlyElementsIn(outputGroupsWithDefaultIfRunningTests(shouldRunTests, "a", "b"));
  }

  @Test
  public void testDetermineOutputGroupsIgnoresUnknownGroup(@TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-foo"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    // "foo" doesn't exist, but that shouldn't be a problem.
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(shouldRunTests, "x", "y", "z"));
  }

  @Test
  public void testDetermineOutputGroupsRemovesPreviouslyAddedGroup(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups;
    outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("+a", "-a"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(shouldRunTests, "x", "y", "z"));

    // Order matters here.
    outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-a", "+a"),
            ValidationMode.OFF,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(shouldRunTests, "x", "y", "z", "a"));
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroup(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList(),
            ValidationMode.OUTPUT_GROUP,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "x", "y", "z", OutputGroupInfo.VALIDATION));
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterOverride(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("foo"),
            ValidationMode.OUTPUT_GROUP,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "foo", OutputGroupInfo.VALIDATION));
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterAdd(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("+a"),
            ValidationMode.OUTPUT_GROUP,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "x", "y", "z", "a", OutputGroupInfo.VALIDATION));
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupAfterRemove(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-x"),
            ValidationMode.OUTPUT_GROUP,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "y", "z", OutputGroupInfo.VALIDATION));
  }

  @Test
  public void testDetermineOutputGroupsContainsValidationGroupDespiteRemove(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-" + OutputGroupInfo.VALIDATION),
            ValidationMode.OUTPUT_GROUP,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "x", "y", "z", OutputGroupInfo.VALIDATION));
  }

  @Test
  public void testDetermineOutputGroupsContainsTopLevelValidationGroup(
      @TestParameter boolean shouldRunTests) {
    Set<String> outputGroups =
        determineOutputGroups(
            ImmutableSet.of("x", "y", "z"),
            asList("-" + OutputGroupInfo.VALIDATION_TOP_LEVEL),
            ValidationMode.ASPECT,
            /*shouldRunTests=*/ shouldRunTests);
    assertThat(outputGroups)
        .containsExactlyElementsIn(
            outputGroupsWithDefaultIfRunningTests(
                shouldRunTests, "x", "y", "z", OutputGroupInfo.VALIDATION_TOP_LEVEL));
  }

  private static Iterable<String> outputGroupsWithDefaultIfRunningTests(
      boolean shouldRunTests, String... groups) {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    result.add(groups);
    if (shouldRunTests) {
      result.add(DEFAULT);
    }
    return result.build();
  }
}
