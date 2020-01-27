// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.escape.CharEscaperBuilder;
import com.google.common.escape.Escaper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Various utility methods operating on strings.
 */
public class StringUtilities {

  private static final Joiner NEWLINE_JOINER = Joiner.on('\n');

  private static final Escaper KEY_ESCAPER = new CharEscaperBuilder()
      .addEscape('!', "!!")
      .addEscape('<', "!<")
      .addEscape('>', "!>")
      .toEscaper();

  private static final Escaper CONTROL_CHAR_ESCAPER = new CharEscaperBuilder()
      .addEscape('\r', "\\r")
      .addEscapes(new char[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, /*13=\r*/
          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 127}, "<?>")
      .toEscaper();

  /**
   * Java doesn't have multiline string literals, so having to join a bunch
   * of lines is a very common problem. So, here's a static method that we
   * can static import in such situations.
   */
  public static String joinLines(String... lines) {
    return NEWLINE_JOINER.join(lines);
  }

  /**
   * A corollary to {@link #joinLines(String[])} for collections.
   */
  public static String joinLines(Collection<String> lines) {
    return NEWLINE_JOINER.join(lines);
  }

  /**
   * combineKeys(x1, ..., xn):
   *   Computes a string that encodes the sequence
   *   x1, ..., xn.  Distinct sequences map to distinct strings.
   *
   *   The encoding is intended to be vaguely human-readable.
   */
  public static String combineKeys(Iterable<String> parts) {
    final StringBuilder buf = new StringBuilder(128);
    for (String part : parts) {
      // We enclose each part in angle brackets to separate them.  Some
      // trickiness is required to ensure that the result is unique (distinct
      // sequences map to distinct strings): we escape any angle bracket
      // characters in the parts by preceding them with an escape character
      // (we use "!") and we also need to escape any escape characters.
      buf.append('<');
      buf.append(KEY_ESCAPER.escape(part));
      buf.append('>');
    }
    return buf.toString();
  }

  /**
   * combineKeys(x1, ..., xn):
   *   Computes a string that encodes the sequence
   *   x1, ..., xn.  Distinct sequences map to distinct strings.
   *
   *   The encoding is intended to be vaguely human-readable.
   */
  public static String combineKeys(String... parts) {
    return combineKeys(ImmutableList.copyOf(parts));
  }

  /**
   * Replaces all occurrences of 'literal' in 'input' with 'replacement'.
   * Like {@link String#replaceAll(String, String)} but for literal Strings
   * instead of regular expression patterns.
   *
   * @param input the input String
   * @param literal the literal String to replace in 'input'.
   * @param replacement the replacement String to replace 'literal' in 'input'.
   * @return the 'input' String with all occurrences of 'literal' replaced with
   *        'replacement'.
   */
  public static String replaceAllLiteral(String input, String literal,
                                         String replacement) {
    int literalLength = literal.length();
    if (literalLength == 0) {
      return input;
    }
    StringBuilder result = new StringBuilder(
        input.length() + replacement.length());
    int start = 0;
    int index = 0;

    while ((index = input.indexOf(literal, start)) >= 0) {
      result.append(input, start, index);
      result.append(replacement);
      start = index + literalLength;
    }
    result.append(input.substring(start));
    return result.toString();
  }

  /**
   * Creates a simple key-value table of the form
   *
   * <pre>
   * key: some value
   * another key: some other value
   * yet another key: and so on ...
   * </pre>
   *
   * The return value will not include a final {@code "\n"}.
   */
  public static String layoutTable(Map<String, String> data) {
    List<String> tableLines = new ArrayList<>();
    for (Map.Entry<String, String> entry : data.entrySet()) {
      tableLines.add(entry.getKey() + ": " + entry.getValue());
    }
    return NEWLINE_JOINER.join(tableLines);
  }

  /**
   * Returns an easy-to-read string approximation of a number of bytes,
   * e.g. "21MB".  Note, these are IEEE units, i.e. decimal not binary powers.
   */
  public static String prettyPrintBytes(long bytes) {
    if (bytes < 1E4) {  // up to 10KB
      return bytes + "B";
    } else if (bytes < 1E7) {  // up to 10MB
      return ((int) (bytes / 1E3)) + "KB";
    } else if (bytes < 1E11) {  // up to 100GB
      return ((int) (bytes / 1E6)) + "MB";
    } else {
      return ((int) (bytes / 1E9)) + "GB";
    }
  }

  /**
   * Returns true if 'source' contains 'target' as a sub-array.
   */
  public static boolean containsSubarray(char[] source, char[] target) {
    if (target.length > source.length) {
      return false;
    }
    for (int i = 0; i < source.length - target.length + 1; i++) {
      boolean matches = true;
      for (int j = 0; j < target.length; j++) {
        if (source[i + j] != target[j]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        return true;
      }
    }
    return false;
  }

  /**
   * Replace control characters with visible strings.
   * @return the sanitized string.
   */
  public static String sanitizeControlChars(String message) {
    return CONTROL_CHAR_ESCAPER.escape(message);
  }

  /**
   * Converts a Java style function name to a Python style function name the following way:
   * every upper case character gets replaced with an underscore and its lower case counterpart.
   * <p>E.g. fooBar --> foo_bar 
   */
  public static String toPythonStyleFunctionName(String javaStyleFunctionName) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < javaStyleFunctionName.length(); i++) {
      char c = javaStyleFunctionName.charAt(i);
      if (Character.isUpperCase(c)) {
        sb.append('_').append(Character.toLowerCase(c));
      } else {
        sb.append(c);
      }
    }
    return sb.toString();
  }
}
