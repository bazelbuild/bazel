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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Encapsulates logic for gathering tests for {@code test_suite}'s {@code $implicit_tests}
 * attribute.
 *
 * <p>Usage is tightly coupled with the package loading process. Expected flow is roughly
 *
 * <ol>
 *   <li>{@link #getTestSuiteImplicitTestsRefForTags} for all relevant {@code test_suite}s
 *   <li>Then, after all all targets have been added to the package...
 *       <ol>
 *         <li>{@link #clearAccumulatedTests()}
 *         <li>{@link #processRule} for every rule in the package
 *         <li>{@link #sortTests()}
 *       </ol>
 *   <li>Repeat the previous step(s) as necessary, eg due to skyframe restarts from missing deps
 * </ol>
 */
class TestSuiteImplicitTestsAccumulator {

  private final Map<ImmutableSet<String>, ImplicitTestsAccumulator> testSuiteImplicitTests =
      new HashMap<>();

  /**
   * Returns a reference to the list of tests matching tags (or all tests if empty), to be populated
   * by {@link #processRule}.
   */
  List<Label> getTestSuiteImplicitTestsRefForTags(List<String> tags) {
    ImplicitTestsAccumulator accumulatorForTags =
        testSuiteImplicitTests.computeIfAbsent(
            ImmutableSet.copyOf(tags), ImplicitTestsAccumulator::new);
    return Collections.unmodifiableList(accumulatorForTags.tests);
  }

  /** Clears all accumulated tests. */
  void clearAccumulatedTests() {
    testSuiteImplicitTests.values().forEach(acc -> acc.tests.clear());
  }

  /**
   * Processes a rule from the package, adding it to the necessary {@code $implicit_test} values
   * returned by {@link #getTestSuiteImplicitTestsRefForTags}.
   */
  void processRule(Rule rule) {
    if (testSuiteImplicitTests.isEmpty()) {
      // No test suites requiring implicit test accumulation encountered.
      return;
    }

    if (TargetUtils.isTestRule(rule) && !TargetUtils.hasManualTag(rule)) {
      NonconfigurableAttributeMapper mapper = NonconfigurableAttributeMapper.of(rule);
      Set<String> testSuiteTags =
          ImmutableSet.<String>builder()
              .addAll(mapper.get("tags", Types.STRING_LIST))
              .add(mapper.get("size", Type.STRING))
              .build();
      for (ImplicitTestsAccumulator acc : testSuiteImplicitTests.values()) {
        if (TestTargetUtils.testMatchesFilters(testSuiteTags, acc.requiredTags, acc.excludedTags)) {
          acc.tests.add(rule.getLabel());
        }
      }
    }
  }

  /**
   * Sorts all of accumulated test lists returned by {@link #getTestSuiteImplicitTestsRefForTags}.
   */
  void sortTests() {
    testSuiteImplicitTests.values().forEach(acc -> Collections.sort(acc.tests));
  }

  private static class ImplicitTestsAccumulator {
    private final Collection<String> requiredTags;
    private final Collection<String> excludedTags;
    private final List<Label> tests = new ArrayList<>();

    private ImplicitTestsAccumulator(Set<String> testTags) {
      Pair<Collection<String>, Collection<String>> requiredAndExcludedTags =
          TestTargetUtils.sortTagsBySense(testTags);
      this.requiredTags = requiredAndExcludedTags.first;
      this.excludedTags = requiredAndExcludedTags.second;
    }
  }
}
