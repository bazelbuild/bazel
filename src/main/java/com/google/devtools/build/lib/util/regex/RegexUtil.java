// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.regex;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A utility class for regex-related operations. */
public final class RegexUtil {
  // If a regex pattern matches this pattern, it matches like a literal string after unescaping
  // dots. Both `\` and `.` are disallowed unless they appear as `\.`.
  private static final Pattern LITERAL_PATTERN_WITH_DOT_UNESCAPED =
      Pattern.compile("(?:[^\\[\\](){}^$|*+?.\\\\]|\\\\\\.)*");

  // If a regex matches this pattern, it matches on the suffix of a string. We need to rule out
  // possessive and reluctant quantifiers, which are the only possible characters after the ".*"
  // prefix that make it unsafe to drop.
  private static final Pattern EXTRACT_SUFFIX_MATCH =
      Pattern.compile("(?:\\^|\\\\A)*\\.\\*(?<suffix>[^?+].*)?");

  /**
   * Returns a {@link Predicate} that matches the input string (in internal encoding according to
   * {@link com.google.devtools.build.lib.util.StringEncoding}) against the regex pattern with the
   * same semantics as {@link Matcher#matches()}, but more optimized and as if the pattern was
   * compiled with {@link Pattern#DOTALL}.
   */
  public static Predicate<String> asOptimizedMatchingPredicate(Pattern regexPattern) {
    String pattern = regexPattern.pattern();
    if (pattern.contains("|") || pattern.contains("\\Q")) {
      // Alternations make it so that appending "$" may not force a match to the end. While we could
      // wrap the pattern in "(?:" + pattern + ")$", that would no longer be amenable to the regex
      // engine's optimization pass exploited below - it only applies when the first node is a
      // literal, not a group.
      // Unmatched \Q...\E sequences can have the same effect.
      return s -> regexPattern.matcher(internalToUnicode(s)).matches();
    }
    // If the pattern matches a literal string, optimize to an equals call.
    if (LITERAL_PATTERN_WITH_DOT_UNESCAPED.matcher(pattern).matches()) {
      return unicodeToInternal(pattern.replace("\\.", "."))::equals;
    }
    // Recognize a pattern that starts with ".*" and drop it so that the engine sees the next part
    // of the pattern as the beginning and potentially optimizes the literal search.
    // We don't apply the same transformation to the end of the pattern since it is harder to
    // determine if a ".*" suffix is safe to drop due to backslash escapes.
    Matcher suffixMatch = EXTRACT_SUFFIX_MATCH.matcher(pattern);
    if (!suffixMatch.matches()) {
      return s -> regexPattern.matcher(internalToUnicode(s)).matches();
    }
    String suffixPattern = suffixMatch.group("suffix");
    // A null suffixPattern implies the regex is equivalent to ".*", which matches any input string.
    if (suffixPattern == null) {
      return s -> true;
    }
    // If the pattern matches a literal suffix, optimize to a string suffix
    // match, which is by far the fastest way to match.
    if (LITERAL_PATTERN_WITH_DOT_UNESCAPED.matcher(suffixPattern).matches()) {
      String literalSuffix = unicodeToInternal(suffixPattern.replace("\\.", "."));
      return s -> s.endsWith(literalSuffix);
    }
    // Turn the "match" pattern into an equivalent "find" pattern, since these are the only ones
    // that benefit from the Boyer-Moore optimization in the Java regex engine.
    // https://github.com/openjdk/jdk/blob/50dced88ff1aed23bb4c8fe9e4a08e6cc200b897/src/java.base/share/classes/java/util/regex/Pattern.java#L1959-L1969
    Pattern compiled = Pattern.compile(suffixPattern + "\\z", Pattern.DOTALL);
    return s -> compiled.matcher(internalToUnicode(s)).find();
  }

  private RegexUtil() {}
}
