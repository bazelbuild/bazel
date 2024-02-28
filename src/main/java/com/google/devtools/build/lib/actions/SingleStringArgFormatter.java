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

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;

/**
 * Implementation of a formatter that supports only a single '%s'
 *
 * <p>This implementation is used in command line item expansions that use formatting. We use a
 * custom implementation to improve performance and avoid GC.
 */
public final class SingleStringArgFormatter {

  private SingleStringArgFormatter() {}

  /**
   * Returns true if the format string contains a single '%s'.
   *
   * <p>%% escape sequences are supported.
   */
  public static boolean isValid(String formatStr) {
    return onlyOccurrence(formatStr, formatStr.length()) != -1;
  }

  /**
   * Calculates the format specifier's contribution to the length of a string created by calling
   * {@link #format}, without actually applying any formatting.
   *
   * <p>For a typical format specifier with no escape characters, returns {@code formatStr.length()
   * - 2}, since the {@code %s} gets replaced during formatting. The result may differ if the format
   * specifier contains escape characters.
   *
   * <p>For all valid format specifiers, the following holds:
   *
   * <pre>{@code
   * format(formatStr, subject).length() == formatSpecifierLength(formatStr) + subject.length()
   * }</pre>
   *
   * @throws IllegalArgumentException if the format string is invalid.
   */
  public static int formattedLength(String formatStr) {
    return getLengthAndMaybeFormat(formatStr, "", null);
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
    StringBuilder sb = new StringBuilder(formatStr.length() + subject.length() - 2);
    getLengthAndMaybeFormat(formatStr, subject, sb);
    return sb.toString();
  }

  @CanIgnoreReturnValue
  private static int getLengthAndMaybeFormat(
      String formatStr, String subject, @Nullable StringBuilder sb) {
    int n = formatStr.length();
    int idx0 = 0;
    int subjects = 0;
    int length = 0;

    while (idx0 < n) {
      int idx = formatStr.indexOf('%', idx0);
      if (idx == -1) {
        break;
      }
      if ((idx + 1) < n) {
        char c = formatStr.charAt(idx + 1);
        if (c == 's') {
          if (sb != null) {
            sb.append(formatStr, idx0, idx).append(subject);
          }
          length += idx - idx0 + subject.length();
          subjects++;
        } else if (c == '%') {
          if (sb != null) {
            sb.append(formatStr, idx0, idx + 1);
          }
          length += idx + 1 - idx0;
        } else {
          // Illegal sequence found
          throw new IllegalArgumentException(
              "Expected format string with single '%s', found: " + formatStr);
        }
        idx0 = idx + 2;
      } else {
        // Terminating '%' found, illegal
        throw new IllegalArgumentException(
            "Expected format string with single '%s', found: " + formatStr);
      }
    }
    if (subjects != 1) {
      throw new IllegalArgumentException(
          "Expected format string with single '%s', found: " + formatStr);
    }
    if (sb != null) {
      sb.append(formatStr, idx0, n);
    }
    length += n - idx0;
    return length;
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
