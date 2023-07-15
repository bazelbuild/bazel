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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/**
 * Handles options that specify list of included/excluded regex expressions. Validates whether
 * string is included in that filter.
 *
 * <p>String is considered to be included into the filter if it does not match any of the excluded
 * regex expressions and if it matches at least one included regex expression.
 */
@Immutable
public final class RegexFilter implements Predicate<String> {
  // Null inclusion or exclusion pattern means those patterns are not used.
  @Nullable private final Pattern inclusionPattern;
  @Nullable private final Pattern exclusionPattern;
  private final int hashCode;

  @Nullable private final String originalInput;

  /**
   * Converts from a comma-separated list of regex expressions with optional -/+ prefix into the
   * RegexFilter. Commas prefixed with backslash are considered to be part of regex definition and
   * not a delimiter between separate regex expressions.
   *
   * <p>Order of expressions is not important. Empty entries are ignored. '-' marks an excluded
   * expression.
   */
  public static class RegexFilterConverter extends Converter.Contextless<RegexFilter> {

    @Override
    public RegexFilter convert(String input) throws OptionsParsingException {
      List<String> inclusionList = new ArrayList<>();
      List<String> exclusionList = new ArrayList<>();

      for (String piece : input.split("(?<!\\\\),")) { // Split on ',' but not on '\,'
        piece = piece.replace("\\,", ",");
        boolean isExcluded = piece.startsWith("-");
        if (isExcluded || piece.startsWith("+")) {
          piece = piece.substring(1);
        }
        if (piece.length() > 0) {
          (isExcluded ? exclusionList : inclusionList).add(piece);
        }
      }

      try {
        return new RegexFilter(inclusionList, exclusionList, input);
      } catch (PatternSyntaxException e) {
        throw new OptionsParsingException(
            "Failed to build valid regular expression: " + e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of regex expressions with prefix '-' specifying"
          + " excluded paths";
    }
  }

  /**
   * Constructor taking regexes directly.
   *
   * <p>Null {@code inclusionPattern} or {@code exclusionPattern} means that inclusion or exclusion
   * matching will not be applied, respectively.
   */
  private RegexFilter(
      @Nullable Pattern inclusionPattern,
      @Nullable Pattern exclusionPattern,
      @Nullable String originalInput) {
    this.inclusionPattern = inclusionPattern;
    this.exclusionPattern = exclusionPattern;
    this.originalInput = originalInput;
    this.hashCode =
        Objects.hash(
            inclusionPattern == null ? null : inclusionPattern.pattern(),
            exclusionPattern == null ? null : exclusionPattern.pattern());
  }

  private RegexFilter(List<String> inclusions, List<String> exclusions, String originalInput) {
    this(takeUnionOfRegexes(inclusions), takeUnionOfRegexes(exclusions), originalInput);
  }

  /** Creates new RegexFilter using provided inclusion and exclusion path lists. */
  public RegexFilter(List<String> inclusions, List<String> exclusions) {
    this(inclusions, exclusions, /* originalInput= */ null);
  }

  /**
   * Converts a list of regex expressions into a single regex representing its union or null when
   * the list is empty.
   */
  @Nullable
  private static Pattern takeUnionOfRegexes(List<String> regexList) {
    if (regexList.isEmpty()) {
      return null;
    }
    TreeSet<String> deduped = new TreeSet<>(regexList);
    // Wraps each individual regex into an independent group, then combines them using '|' and
    // wraps the result in a non-capturing group.
    return Pattern.compile("(?:(?>" + Joiner.on(")|(?>").join(deduped) + "))");
  }

  /**
   * @return true iff given string is included (it does not match exclusion pattern (if any) and
   *     matches inclusionPatter (if any)).
   */
  public boolean isIncluded(String value) {
    if (exclusionPattern != null && exclusionPattern.matcher(value).find()) {
      return false;
    }
    if (inclusionPattern == null) {
      return true;
    }
    return inclusionPattern.matcher(value).find();
  }

  @Override
  public boolean test(String value) {
    return isIncluded(value);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    if (inclusionPattern != null) {
      builder.append(inclusionPattern.pattern().replace(",", "\\,"));
      if (exclusionPattern != null) {
        builder.append(",");
      }
    }
    if (exclusionPattern != null) {
      builder.append("-");
      builder.append(exclusionPattern.pattern().replace(",", "\\,"));
    }
    return builder.toString();
  }

  /**
   * RegexFilter doesn't serialize cleanly: {@code
   * !RegexFilter.convert(".*").toString().equals(".")}.
   *
   * <p>This method provides the ability to reproduce the original input string.
   */
  @Nullable
  public String toOriginalString() {
    return originalInput;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof RegexFilter)) {
      return false;
    }

    RegexFilter otherFilter = (RegexFilter) other;
    if (this.exclusionPattern == null ^ otherFilter.exclusionPattern == null) {
      return false;
    }
    if (this.inclusionPattern == null ^ otherFilter.inclusionPattern == null) {
      return false;
    }
    if (this.exclusionPattern != null && !this.exclusionPattern.pattern().equals(
        otherFilter.exclusionPattern.pattern())) {
      return false;
    }
    if (this.inclusionPattern != null && !this.inclusionPattern.pattern().equals(
        otherFilter.inclusionPattern.pattern())) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }
}
