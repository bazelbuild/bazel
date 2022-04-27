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
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Iterator;

/**
 * Various utility methods operating on strings.
 */
public class StringUtil {
  /**
   * Creates a comma-separated list of words as in English.
   *
   * <p>Example: ["a", "b", "c"] -&gt; "a, b or c".
   */
  public static String joinEnglishList(Iterable<?> choices) {
    return joinEnglishList(choices, "or", "");
  }

  /**
   * Creates a comma-separated list of words as in English with the given last-separator.
   *
   * <p>Example with lastSeparator="then": ["a", "b", "c"] -&gt; "a, b then c".
   */
  public static String joinEnglishList(Iterable<?> choices, String lastSeparator) {
    return joinEnglishList(choices, lastSeparator, "");
  }

  /**
   * Creates a comma-separated list of words as in English with the given last-separator and quotes.
   *
   * <p>Example with lastSeparator="then", quote="'": ["a", "b", "c"] -&gt; "'a', 'b' then 'c'".
   */
  public static String joinEnglishList(Iterable<?> choices, String lastSeparator, String quote) {
    StringBuilder buf = new StringBuilder();
    for (Iterator<?> ii = choices.iterator(); ii.hasNext(); ) {
      Object choice = ii.next();
      if (buf.length() > 0) {
        buf.append(ii.hasNext() ? "," : " " + lastSeparator);
        buf.append(" ");
      }
      buf.append(quote).append(choice).append(quote);
    }
    return buf.length() == 0 ? "nothing" : buf.toString();
  }

  /**
   * Lists items up to a given limit, then prints how many were omitted.
   */
  public static StringBuilder listItemsWithLimit(StringBuilder appendTo, int limit,
      Collection<?> items) {
    Preconditions.checkState(limit > 0);
    Joiner.on(", ").appendTo(appendTo, Iterables.limit(items, limit));
    if (items.size() > limit) {
      appendTo.append(" ...(omitting ")
          .append(items.size() - limit)
          .append(" more item(s))");
    }
    return appendTo;
  }

  /**
   * Returns the ordinal representation of the number.
   */
  public static String ordinal(int number) {
    switch (number) {
      case 1:
        return "1st";
      case 2:
        return "2nd";
      case 3:
        return "3rd";
      default:
        return number + "th";
    }
  }

 /**
   * Decodes UTF-8 stored in a String, treating each input character as a byte.
   *
   * Several Bazel subsystems, including Starlark, store bytes in `String`
   * values where each `char` stores one `byte` in its lower 8 bits. This
   * function converts its input to a `[]byte`, then decodes that byte array
   * as UTF-8.
   *
   * Using U+2049 (EXCLAMATION QUESTION MARK) as an example:
   *
   *   "\u2049".getBytes(UTF_8) == [0xE2, 0x81, 0x89]
   *
   *   decodeBytestringUtf8("\u00E2\u0081\u0089") == "\u2049"
   *
   * The return value is suitable for passing to Protobuf string fields or
   * printing to the terminal.
   */
  public static String decodeBytestringUtf8(String bytestringUtf8) {
    if (bytestringUtf8.chars().allMatch(c -> c < 128)) {
      return bytestringUtf8;
    }
    final byte[] utf8 = bytestringUtf8.getBytes(StandardCharsets.ISO_8859_1);
    return new String(utf8, StandardCharsets.UTF_8);
  }

  /**
   * Encodes a String to UTF-8, then converts those UTF-8 bytes to a String
   * by zero-extending each `byte` into a `char`.
   *
   * Using U+2049 (EXCLAMATION QUESTION MARK) as an example:
   *
   *   "\u2049".getBytes(UTF_8) == [0xE2, 0x81, 0x89]
   *
   *   encodeBytestringUtf8("\u2049") == "\u00E2\u0081\u0089"
   *
   * See {@link #decodeBytestringUtf8} for motivation.
   */
  public static String encodeBytestringUtf8(String unicode) {
    if (unicode.chars().allMatch(c -> c < 128)) {
      return unicode;
    }
    final byte[] utf8 = unicode.getBytes(StandardCharsets.UTF_8);
    return new String(utf8, StandardCharsets.ISO_8859_1);
  }
}
