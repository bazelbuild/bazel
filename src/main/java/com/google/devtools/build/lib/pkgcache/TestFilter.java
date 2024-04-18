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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Predicate that implements test filtering using the command-line options in {@link
 * LoadingOptions}. Implements {@link #hashCode} and {@link #equals} so it can be used as a Skyframe
 * key.
 */
@AutoCodec
public final class TestFilter implements com.google.common.base.Predicate<Target> {
  private static final Predicate<Target> ALWAYS_TRUE = (t) -> true;

  /** Convert the options into a test filter. */
  public static TestFilter forOptions(LoadingOptions options) {
    return new TestFilter(
        ImmutableSet.copyOf(options.testSizeFilterSet),
        ImmutableSet.copyOf(options.testTimeoutFilterSet),
        ImmutableList.copyOf(options.testTagFilterList),
        ImmutableList.copyOf(options.testLangFilterList));
  }

  private final ImmutableSet<TestSize> testSizeFilterSet;
  private final ImmutableSet<TestTimeout> testTimeoutFilterSet;
  private final ImmutableList<String> testTagFilterList;
  private final ImmutableList<String> testLangFilterList;
  private final Predicate<Target> impl;

  @VisibleForSerialization
  TestFilter(
      ImmutableSet<TestSize> testSizeFilterSet,
      ImmutableSet<TestTimeout> testTimeoutFilterSet,
      ImmutableList<String> testTagFilterList,
      ImmutableList<String> testLangFilterList) {
    this.testSizeFilterSet = testSizeFilterSet;
    this.testTimeoutFilterSet = testTimeoutFilterSet;
    this.testTagFilterList = testTagFilterList;
    this.testLangFilterList = testLangFilterList;
    Predicate<Target> testFilter = ALWAYS_TRUE;
    if (!testSizeFilterSet.isEmpty()) {
      testFilter = testFilter.and(testSizeFilter(testSizeFilterSet));
    }
    if (!testTimeoutFilterSet.isEmpty()) {
      testFilter = testFilter.and(testTimeoutFilter(testTimeoutFilterSet));
    }
    if (!testTagFilterList.isEmpty()) {
      testFilter = testFilter.and(TargetUtils.tagFilter(testTagFilterList));
    }
    if (!testLangFilterList.isEmpty()) {
      testFilter = testFilter.and(testLangFilter(testLangFilterList));
    }
    impl = testFilter;
  }

  @Override
  public boolean apply(@Nullable Target input) {
    return impl.test(input);
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
    if (!(o instanceof TestFilter f)) {
      return false;
    }
    return f.testSizeFilterSet.equals(testSizeFilterSet)
        && f.testTimeoutFilterSet.equals(testTimeoutFilterSet)
        && f.testTagFilterList.equals(testTagFilterList)
        && f.testLangFilterList.equals(testLangFilterList);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("testSizeFilterSet", testSizeFilterSet)
        .add("testTimeoutFilterSet", testTimeoutFilterSet)
        .add("testTagFilterList", testTagFilterList)
        .add("testLangFilterList", testLangFilterList)
        .toString();
  }

  /**
   * Returns a predicate to be used for test size filtering, i.e., that only accepts tests of the
   * given size.
   */
  @VisibleForTesting
  public static Predicate<Target> testSizeFilter(final Set<TestSize> allowedSizes) {
    return target ->
        target instanceof Rule && allowedSizes.contains(TestSize.getTestSize((Rule) target));
  }

  /**
   * Returns a predicate to be used for test timeout filtering, i.e., that only accepts tests of the
   * given timeout.
   */
  @VisibleForTesting
  public static Predicate<Target> testTimeoutFilter(final Set<TestTimeout> allowedTimeouts) {
    return target ->
        target instanceof Rule
            && allowedTimeouts.contains(TestTimeout.getTestTimeout((Rule) target));
  }

  /**
   * Returns a predicate to be used for test language filtering, i.e., that only accepts tests of
   * the specified languages.
   */
  private static Predicate<Target> testLangFilter(List<String> langFilterList) {
    final Set<String> requiredLangs = new HashSet<>();
    final Set<String> excludedLangs = new HashSet<>();

    for (String lang : langFilterList) {
      if (lang.startsWith("-")) {
        lang = lang.substring(1);
        excludedLangs.add(lang);
      } else {
        requiredLangs.add(lang);
      }
    }

    return rule -> {
      String ruleLang = TargetUtils.getRuleLanguage(rule);
      return (requiredLangs.isEmpty() || requiredLangs.contains(ruleLang))
          && !excludedLangs.contains(ruleLang);
    };
  }
}
