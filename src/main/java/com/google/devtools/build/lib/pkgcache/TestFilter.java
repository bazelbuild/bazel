// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.packages.TestTimeout;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Predicate that implements test filtering using the command-line options in {@link LoadingOptions}.
 * Implements {@link #hashCode} and {@link #equals} so it can be used as a Skyframe key.
 */
public final class TestFilter implements Predicate<Target> {
  /** Convert the options into a test filter. */
  public static TestFilter forOptions(
      LoadingOptions options, ExtendedEventHandler eventHandler, Set<String> ruleNames) {
    Predicate<Target> testFilter = Predicates.alwaysTrue();
    if (!options.testSizeFilterSet.isEmpty()) {
      testFilter = Predicates.and(testFilter,
          TestTargetUtils.testSizeFilter(options.testSizeFilterSet));
    }
    if (!options.testTimeoutFilterSet.isEmpty()) {
      testFilter = Predicates.and(testFilter,
          TestTargetUtils.testTimeoutFilter(options.testTimeoutFilterSet));
    }
    if (!options.testTagFilterList.isEmpty()) {
      testFilter = Predicates.and(testFilter,
          TargetUtils.tagFilter(options.testTagFilterList));
    }
    if (!options.testLangFilterList.isEmpty()) {
      testFilter = Predicates.and(testFilter,
          TestTargetUtils.testLangFilter(options.testLangFilterList, eventHandler, ruleNames));
    }
    return new TestFilter(options.testSizeFilterSet, options.testTimeoutFilterSet,
        options.testTagFilterList, options.testLangFilterList, testFilter);
  }

  private final Set<TestSize> testSizeFilterSet;
  private final Set<TestTimeout> testTimeoutFilterSet;
  private final List<String> testTagFilterList;
  private final List<String> testLangFilterList;
  private final Predicate<Target> impl;

  private TestFilter(Set<TestSize> testSizeFilterSet, Set<TestTimeout> testTimeoutFilterSet,
      List<String> testTagFilterList, List<String> testLangFilterList, Predicate<Target> impl) {
    this.testSizeFilterSet = testSizeFilterSet;
    this.testTimeoutFilterSet = testTimeoutFilterSet;
    this.testTagFilterList = testTagFilterList;
    this.testLangFilterList = testLangFilterList;
    this.impl = impl;
  }

  @Override
  public boolean apply(@Nullable Target input) {
    return impl.apply(input);
  }

  @Override
  public int hashCode() {
    return Objects.hash(testSizeFilterSet, testTimeoutFilterSet, testTagFilterList,
        testLangFilterList);
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof TestFilter)) {
      return false;
    }
    TestFilter f = (TestFilter) o;
    return f.testSizeFilterSet.equals(testSizeFilterSet)
        && f.testTimeoutFilterSet.equals(testTimeoutFilterSet)
        && f.testTagFilterList.equals(testTagFilterList)
        && f.testLangFilterList.equals(testLangFilterList);
  }
}
