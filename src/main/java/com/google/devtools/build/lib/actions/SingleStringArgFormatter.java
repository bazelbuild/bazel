// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

/**
 * Implementation of a formatter that supports only a single '%s'
 *
 * <p>This implementation is used in command line item expansions that use formatting. We use a
 * custom implementation to improve performance and avoid GC.
 */
public class SingleStringArgFormatter {

  /**
   * Returns true if the format string contains a single '%s'.
   *
   * <p>%% escape sequences are supported.
   */
  public static boolean isValid(String formatStr) {
    return onlyOccurrence(formatStr, formatStr.length()) != -1;
  }

  /**
   * Returns the equivalent result of <code>String.format(formatStr, subject)</code>, under the
   * assumption that the format string contains a single %s.
   *
   * <p>Use {@link #isValid} to validate the format string.
   *
   * @throws IllegalArgumentException if the format string is invalid.
   */
  public static String format(String formatStr, String subject) {
    int n = formatStr.length();
    int idx = onlyOccurrence(formatStr, n);
    if (idx < 0) {
      throw new IllegalArgumentException(
          "Expected format string with single '%s', found: " + formatStr);
    }
    return new StringBuilder(n + subject.length() - 2)
        .append(formatStr, 0, idx)
        .append(subject)
        .append(formatStr, idx + 2, n)
        .toString();
  }

  /*
   * Returns the index of the only occurrence of %s. Skips any %%. Returns -1 if {@code formatStr}
   * is malformed.
   */
  private static int onlyOccurrence(String formatStr, int n) {
    int idx = nextOccurrence(formatStr, n, 0);
    if (idx < 0) {
      return -1;
    }
    // Only one occurrence please
    if (nextOccurrence(formatStr, n, idx + 2) != -1) {
      return -1;
    }
    return idx;
  }

  /**
   * Returns next occurrence of %s. Skips any %%.
   *
   * @return
   *     <li>[0-n]: Index of next %s
   *     <li>-1: No %s found until end of string
   *     <li>-2: Illegal sequence found, eg. %f
   */
  private static int nextOccurrence(String formatStr, int n, int idx) {
    while (idx < n) {
      idx = formatStr.indexOf('%', idx);
      if (idx == -1) {
        break;
      }
      if ((idx + 1) < n) {
        char c = formatStr.charAt(idx + 1);
        if (c == 's') {
          return idx;
        }
        if (c != '%') {
          // Illegal sequence found
          return -2;
        }
        idx += 2;
      } else {
        // Terminating '%' found, illegal
        return -2;
      }
    }
    return -1;
  }
}
