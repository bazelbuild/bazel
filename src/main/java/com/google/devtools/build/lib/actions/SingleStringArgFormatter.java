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
public final class SingleStringArgFormatter {

  private SingleStringArgFormatter() {}

  /**
   * Returns true if the format string is a valid single-arg formatter.
   *
   * <p>Requirements are:
   *
   * <ul>
   *   <li>Contains exactly one '%s'.
   *   <li>Each occurrence of '%' is either '%s' or '%%' (escape sequence).
   * </ul>
   */
  public static boolean isValid(String formatStr) {
    return formattedLengthOrInvalid(formatStr) != -1;
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
    int length = formattedLengthOrInvalid(formatStr);
    if (length == -1) {
      throw invalidFormatString(formatStr);
    }
    return length;
  }

  /** Returns the formatted length or {@code -1} if invalid. */
  private static int formattedLengthOrInvalid(String formatStr) {
    int length = 0;
    int n = formatStr.length();
    int idx = 0;
    boolean found = false;

    while (idx < n) {
      int next = formatStr.indexOf('%', idx);
      if (next == -1) {
        length += n - idx;
        break;
      }
      if (next == n - 1) {
        return -1; // Terminating '%'.
      }
      switch (formatStr.charAt(next + 1)) {
        case 's' -> {
          if (found) {
            return -1; // Multiple '%s'.
          }
          length += next - idx;
          found = true;
        }
        case '%' -> length += next + 1 - idx;
        default -> {
          return -1; // Illegal sequence.
        }
      }
      idx = next + 2;
    }

    return found ? length : -1;
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
    int n = formatStr.length();
    int idx = 0;
    boolean found = false;

    while (idx < n) {
      int next = formatStr.indexOf('%', idx);
      if (next == -1) {
        sb.append(formatStr, idx, n);
        break;
      }
      if (next == n - 1) {
        throw invalidFormatString(formatStr); // Terminating '%'.
      }
      switch (formatStr.charAt(next + 1)) {
        case 's' -> {
          if (found) {
            throw invalidFormatString(formatStr); // Multiple '%s'.
          }
          sb.append(formatStr, idx, next).append(subject);
          found = true;
        }
        case '%' -> sb.append(formatStr, idx, next + 1);
        default -> throw invalidFormatString(formatStr); // Illegal sequence.
      }
      idx = next + 2;
    }

    if (!found) {
      throw invalidFormatString(formatStr); // No '%s'.
    }
    return sb.toString();
  }

  private static IllegalArgumentException invalidFormatString(String formatStr) {
    return new IllegalArgumentException(
        "Expected format string with single '%s', found: " + formatStr);
  }
}
