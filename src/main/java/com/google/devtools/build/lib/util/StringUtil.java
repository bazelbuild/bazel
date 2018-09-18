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
import java.util.Collection;
import java.util.Iterator;
import javax.annotation.Nullable;

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
   * Appends a prefix and a suffix to each of the Strings.
   */
  public static Iterable<String> append(Iterable<String> values, final String prefix,
      final String suffix) {
    return Iterables.transform(values, input -> prefix + input + suffix);
  }

  /**
   * Indents the specified string by the given number of characters.
   *
   * <p>The beginning of the string before the first newline is not indented.
   */
  public static String indent(String input, int depth) {
    StringBuilder prefix = new StringBuilder();
    prefix.append("\n");
    for (int i = 0; i < depth; i++) {
      prefix.append(" ");
    }

    return input.replace("\n", prefix);
  }

  /**
   * Strips a suffix from a string. If the string does not end with the suffix, returns null.
   */
  public static String stripSuffix(String input, String suffix) {
    return input.endsWith(suffix)
        ? input.substring(0, input.length() - suffix.length())
        : null;
  }

  /**
   * Capitalizes the first character of a string.
   */
  public static String capitalize(String input) {
    if (input.isEmpty()) {
      return input;
    }

    char first = input.charAt(0);
    char capitalized = Character.toUpperCase(first);
    return first == capitalized ? input : capitalized + input.substring(1);
  }

  /** Convert empty string to null. */
  @Nullable
  public static String emptyToNull(@Nullable String input) {
    if (input == null || input.isEmpty()) {
      return null;
    }
    return input;
  }
}
