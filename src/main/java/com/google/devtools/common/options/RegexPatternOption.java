// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Option class wrapping a {@link Pattern class}. We wrap the {@link Pattern} class instance since
 * it uses reference equality, which breaks the assumption of {@link Converter} that {@code
 * converter.convert(sameString).equals(converter.convert(sameString)}.
 *
 * <p>Please note that the equality implementation is based solely on the input regex, therefore
 * patterns expressing the same intent with different regular expressions (e.g. {@code "a"} and
 * {@code "[a]"} will not be treated as equal.
 */
@AutoValue
public abstract class RegexPatternOption {
  static RegexPatternOption create(Pattern regexPattern) {
    return new AutoValue_RegexPatternOption(
        Preconditions.checkNotNull(regexPattern), optimize(regexPattern));
  }

  /** The original regex pattern. */
  public abstract Pattern regexPattern();

  /**
   * A potentially optimized {@link Predicate} that matches the entire input string against the
   * regex pattern.
   */
  public abstract Predicate<String> matcher();

  // If a regex pattern matches this pattern, it matches like a literal string after unescaping
  // dots.
  private static final Pattern LITERAL_PATTERN_WITH_DOT_UNESCAPED =
      Pattern.compile("(?:[^\\[\\](){}^$|*+?.\\\\]|\\\\\\.)*");

  // If a regex matches this pattern, it matches on the suffix of a string. We need to rule out
  // possessive and reluctant quantifiers, which are the only possible characters after the ".*"
  // prefix that make it unsafe to drop.
  private static final Pattern EXTRACT_SUFFIX_MATCH =
      Pattern.compile("\\^?\\.\\*(?<suffix>[^?+].*)?");

  /**
   * Returns a {@link Predicate} that matches the input string against the regex pattern with the
   * same semantics as {@link Matcher#matches()}, but as if the pattern was compiled with {@link
   * Pattern#DOTALL} mode (i.e., the dot character matches all characters, including line
   * terminators).
   */
  private static Predicate<String> optimize(Pattern regexPattern) {
    String pattern = regexPattern.pattern();
    if (pattern.contains("|")) {
      // Alternations make it so that appending "$" may not force a match to the end.
      return s -> regexPattern.matcher(s).matches();
    }
    // Recognize a pattern that starts with ".*" and drop it so that the engine sees the next part
    // of the pattern as the beginning and potentially optimizes the literal search.
    // We don't apply the same transformation to the end of the pattern since it is harder to
    // determine if a ".*" suffix is safe to drop due to backslash escapes.
    Matcher suffixMatch = EXTRACT_SUFFIX_MATCH.matcher(pattern);
    if (!suffixMatch.matches()) {
      return s -> regexPattern.matcher(s).matches();
    }
    String suffixPattern = suffixMatch.group("suffix");
    // A null suffixPattern implies the regex is equivalent to ".*", which matches any input string.
    if (suffixPattern == null) {
      return s -> true;
    }
    // If the pattern matches a literal suffix, optimize to a string suffix
    // match, which is by far the fastest way to match.
    if (LITERAL_PATTERN_WITH_DOT_UNESCAPED.matcher(suffixPattern).matches()) {
      String literalSuffix = suffixPattern.replace("\\.", ".");
      return s -> s.endsWith(literalSuffix);
    }
    // Turn the "match" pattern into an equivalent "find" pattern, since these are the only ones
    // that benefit from the Boyer-Moore optimization in the Java regex engine.
    // https://github.com/openjdk/jdk/blob/50dced88ff1aed23bb4c8fe9e4a08e6cc200b897/src/java.base/share/classes/java/util/regex/Pattern.java#L1959-L1969
    // Guard against the surprising edge case where $ can match right before a single final line
    // terminator (only \n due to Pattern.UNIX_LINES).
    Pattern compiled =
        Pattern.compile(suffixPattern + "$(?!\n)", Pattern.DOTALL | Pattern.UNIX_LINES);
    return s -> compiled.matcher(s).find();
  }

  @Override
  public final boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof RegexPatternOption)) {
      return false;
    }

    RegexPatternOption otherOption = (RegexPatternOption) other;
    return otherOption.regexPattern().pattern().equals(regexPattern().pattern());
  }

  @Override
  public final int hashCode() {
    return regexPattern().pattern().hashCode();
  }
}
