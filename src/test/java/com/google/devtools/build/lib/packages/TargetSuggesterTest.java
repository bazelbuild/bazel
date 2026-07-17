// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TargetSuggesterTest {

  @Test
  public void testRangeDoesntSuggestTarget() {
    String requestedTarget = "range";
    Set<String> packageTargets = new HashSet<>();
    packageTargets.add("target");

    ImmutableList<String> suggestedTargets =
        TargetSuggester.suggestedTargets(requestedTarget, packageTargets);
    assertThat(suggestedTargets).isEmpty();
  }

  @Test
  public void testMisspelledTargetRetrievesProperSuggestion() {
    String misspelledTarget = "AnrdiodTest";

    Set<String> packageTargets = new HashSet<>();
    packageTargets.add("AndroidTest");
    packageTargets.add("AndroidTest_deploy");
    packageTargets.add("AndroidTest_java");

    ImmutableList<String> suggestedTargets =
        TargetSuggester.suggestedTargets(misspelledTarget, packageTargets);
    assertThat(suggestedTargets).containsExactly("AndroidTest");
  }

  @Test
  public void testRetrieveMultipleTargets() {
    String misspelledTarget = "pixel_2_test";

    Set<String> packageTargets = new HashSet<>();
    packageTargets.add("pixel_5_test");
    packageTargets.add("pixel_6_test");
    packageTargets.add("android_2_test");

    ImmutableList<String> suggestedTargets =
        TargetSuggester.suggestedTargets(misspelledTarget, packageTargets);
    assertThat(suggestedTargets).containsExactly("pixel_5_test", "pixel_6_test");
  }

  @Test
  public void testOnlyClosestTargetIsReturned() {
    String misspelledTarget = "Pixel_5_test";

    Set<String> packageTargets = new HashSet<>();
    packageTargets.add("pixel_5_test");
    packageTargets.add("pixel_6_test");
    packageTargets.add("android_2_test");

    ImmutableList<String> suggestedTargets =
        TargetSuggester.suggestedTargets(misspelledTarget, packageTargets);
    assertThat(suggestedTargets).containsExactly("pixel_5_test");
  }

  @Test
  public void prettyPrintEmpty() {
    String empty = TargetSuggester.prettyPrintTargets(ImmutableList.of());
    assertThat(empty).isEmpty();
  }

  @Test
  public void prettyPrintSingleTarget_returnsSingleTarget() {
    ImmutableList<String> targets = ImmutableList.of("pixel_5_test");
    String targetString = TargetSuggester.prettyPrintTargets(targets);
    assertThat(targetString).isEqualTo(" (did you mean pixel_5_test?)");
  }

  @Test
  public void prettyPrintMultipleTargets_returnsMultipleTargets() {
    ImmutableList<String> targets = ImmutableList.of("pixel_5_test", "pixel_6_test");
    String targetString = TargetSuggester.prettyPrintTargets(targets);
    assertThat(targetString).isEqualTo(" (did you mean pixel_5_test, or pixel_6_test?)");
  }
}
