// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.common.base.Strings;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.regex.Pattern;

/**
 * Filter that filters out test cases that either matches or does not match a specified regular
 * expression.
 */
public final class RegExTestCaseFilter extends Filter {
  private static final String TEST_NAME_FORMAT = "%s#%s";

  private final Pattern pattern;
  private final boolean isNegated;

  /**
   * Returns a filter that evaluates to {@code true} if the test case description matches
   * specified regular expression. Otherwise, returns {@code false}.
   */
  public static RegExTestCaseFilter include(String regularExpression) {
    return new RegExTestCaseFilter(regularExpression, false);
  }

 /**
   * Returns a filter that evaluates to {@code false} if the test case description matches
   * specified regular expression. Otherwise, returns {@code true}.
   */
  public static RegExTestCaseFilter exclude(String regularExpression) {
    return new RegExTestCaseFilter(regularExpression, true);
  }

  private RegExTestCaseFilter(String regularExpression, boolean isNegated) {
    this.isNegated = isNegated;
    this.pattern = Pattern.compile(regularExpression);
  }

  @Override
  public boolean shouldRun(Description description) {
    if (description.isSuite()) {
      return true;
    }

    boolean match = pattern.matcher(formatDescriptionName(description)).find();
    return isNegated ? !match : match;
  }

  @Override
  public String describe() {
    return String.format("%sRegEx[%s]", isNegated ? "NOT " : "", pattern.toString());
  }

  private static String formatDescriptionName(Description description) {
    String methodName = Strings.nullToEmpty(description.getMethodName());
    String className = Strings.nullToEmpty(description.getClassName());
    if (methodName.trim().isEmpty() || className.trim().isEmpty()) {
      return description.getDisplayName();
    }
    return String.format(TEST_NAME_FORMAT, className, methodName);
  }
}
