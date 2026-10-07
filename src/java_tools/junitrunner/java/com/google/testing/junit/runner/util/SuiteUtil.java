// Copyright 2024 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Utility methods for inspecting and manipulating test suites. */
@SuppressWarnings({"JdkImmutableCollections", "JdkCollectors"})
public final class SuiteUtil {

  private SuiteUtil() {}

  /**
   * Parses the given test suite property string into a list of class names, trimming whitespace,
   * dropping empty entries, and preserving encounter order while eliminating duplicates.
   *
   * @param property comma-separated list of test class names
   * @return deduplicated list of class names, or an empty list if property is null or blank
   */
  public static List<String> parseSuiteClassNames(@Nullable String property) {
    if (property == null || property.trim().isEmpty()) {
      return Collections.emptyList();
    }
    return Collections.unmodifiableList(
        Arrays.stream(property.split(","))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .distinct()
            .collect(Collectors.toList()));
  }

  /**
   * Returns a canonical name for the top-level suite.
   *
   * <p>If a single suite class is provided, its canonical name is used. Otherwise, the {@code
   * TEST_TARGET} environment variable is used if set, falling back to the alphabetically minimum
   * package name, or {@code "TestSuite"}.
   */
  public static String getTopLevelSuiteName(List<Class<?>> suites) {
    return getTopLevelSuiteName(suites, System.getenv("TEST_TARGET"));
  }

  static String getTopLevelSuiteName(List<Class<?>> suites, @Nullable String testTarget) {
    if (suites.size() == 1) {
      return suites.get(0).getCanonicalName();
    }
    // TODO: Consider supporting an explicit suite name or attribute/property for reporting.
    if (testTarget != null && !testTarget.isEmpty()) {
      return testTarget;
    }
    return suites.stream()
        .map(SuiteUtil::getPackageName)
        .filter(p -> !p.isEmpty())
        .min(String::compareTo)
        .orElse("TestSuite");
  }

  private static String getPackageName(Class<?> clazz) {
    Package pkg = clazz.getPackage();
    if (pkg != null) {
      return pkg.getName();
    }
    String name = clazz.getName();
    int lastDot = name.lastIndexOf('.');
    return lastDot != -1 ? name.substring(0, lastDot) : "";
  }
}
