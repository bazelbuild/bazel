// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
import java.util.SortedSet;

/**
 * Helper class to heuristically compute an instrumentation filter from a list of tests to run.
 */
public final class InstrumentationFilterSupport {
  public static final String INSTRUMENTATION_FILTER_FLAG = "instrumentation_filter";

  /**
   * Method implements a heuristic used to set default value of the --instrumentation_filter option.
   * The following algorithm is used:
   *
   * <ul>
   *   <li>Identify all test targets on the command line.
   *   <li>Expand all test suites into the individual test targets.
   *   <li>Calculate list of package names containing all test targets above.
   *   <li>Replace all "javatests" directories in packages with "java". Similarly, replace
   *       "test/java/" with "main/java". Also, strip trailing "/internal", "/public", and "/tests"
   *       from packages. (See {@link #getInstrumentedPrefix}.)
   *   <li>Set --instrumentation_filter default value to instrument everything in those packages.
   * </ul>
   */
  @VisibleForTesting
  public static String computeInstrumentationFilter(
      EventHandler eventHandler, Collection<Target> testTargets) {
    SortedSet<String> packageFilters = Sets.newTreeSet();
    collectInstrumentedPackages(testTargets, packageFilters);
    optimizeFilterSet(packageFilters);

    String instrumentationFilter =
        Joiner.on("[/:],^//")
            .appendTo(new StringBuilder("^//"), packageFilters)
            .append("[/:]")
            .toString();
    // Fix up if one of the test targets is a top-level target. "//foo[/:]" matches everything
    // under //foo and subpackages, but "//[/:]" only matches targets directly under the top-level
    // package.
    if (instrumentationFilter.equals("^//[/:]")) {
      instrumentationFilter = "^//";
    }
    if (!packageFilters.isEmpty()) {
      eventHandler.handle(
          Event.info("Using default value for --instrumentation_filter: \""
              + instrumentationFilter + "\"."));
      eventHandler.handle(Event.info("Override the above default with --"
          + INSTRUMENTATION_FILTER_FLAG));
    }
    return instrumentationFilter;
  }

  private static void collectInstrumentedPackages(
      Collection<Target> targets, Set<String> packageFilters) {
    for (Target target : targets) {
      // Add package-based filters for every test target.
      packageFilters.add(getInstrumentedPrefix(target.getLabel().getPackageName()));
      if (TargetUtils.isTestSuiteRule(target)) {
        AttributeMap attributes = NonconfigurableAttributeMapper.of((Rule) target);
        // We don't need to handle $implicit_tests attribute since we already added
        // test_suite package to the set.
        for (Label label : attributes.get("tests", BuildType.LABEL_LIST)) {
          // Add package-based filters for all tests in the test suite.
          packageFilters.add(getInstrumentedPrefix(label.getPackageName()));
        }
      }
    }
  }

  /**
   * Returns prefix string that should be instrumented for a given package. Input string should
   * be formatted like the output of Label.getPackageName().
   * Generally, package name will be used as such string with two modifications.
   * - "javatests/ directories will be substituted with "java/", since we do
   * not want to instrument java test code. "java/" directories in "test/" will
   * be replaced by the same in "main/".
   * - "/internal", "/public", and "tests/" package suffix will be dropped, since usually we would
   * want to instrument code in the parent package as well
   */
  @VisibleForTesting
  public static String getInstrumentedPrefix(String packageName) {
    if (packageName.endsWith("/internal")) {
      packageName = packageName.substring(0, packageName.length() - "/internal".length());
    } else if (packageName.endsWith("/public")) {
      packageName = packageName.substring(0, packageName.length() - "/public".length());
    } else if (packageName.endsWith("/tests")) {
      packageName = packageName.substring(0, packageName.length() - "/tests".length());
    }
    return packageName
        .replaceFirst("(?<=^|/)javatests/", "java/")
        .replaceFirst("(?<=^|/)test/java/", "main/java/");
  }

  private static void optimizeFilterSet(SortedSet<String> packageFilters) {
    Iterator<String> iterator = packageFilters.iterator();
    if (iterator.hasNext()) {
      // Optimize away nested filters.
      iterator = packageFilters.iterator();
      String prev = iterator.next();
      while (iterator.hasNext()) {
        String current = iterator.next();
        if (current.startsWith(prev)) {
          iterator.remove();
        } else {
          prev = current;
        }
      }
    }
  }
}
