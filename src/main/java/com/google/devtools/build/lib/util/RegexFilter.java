// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Handles options that specify list of included/excluded regex expressions.
 * Validates whether string is included in that filter.
 *
 * String is considered to be included into the filter if it does not match
 * any of the excluded regex expressions and if it matches at least one
 * included regex expression.
 */
public class RegexFilter implements Serializable {
  private final Pattern inclusionPattern;
  private final Pattern exclusionPattern;
  private final int hashCode;

  /**
   * Converts from a colon-separated list of regex expressions with optional
   * -/+ prefix into the RegexFilter. Colons prefixed with backslash are
   * considered to be part of regex definition and not a delimiter between
   * separate regex expressions.
   *
   * Order of expressions is not important. Empty entries are ignored.
   * '-' marks an excluded expression.
   */
  public static class RegexFilterConverter
      implements Converter<RegexFilter> {

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
        return new RegexFilter(inclusionList, exclusionList);
      } catch (PatternSyntaxException e) {
        throw new OptionsParsingException("Failed to build valid regular expression: "
            + e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of regex expressions with prefix '-' specifying"
          + " excluded paths";
    }

  }

  /**
   * Creates new RegexFilter using provided inclusion and exclusion path lists.
   */
  public RegexFilter(List<String> inclusions, List<String> exclusions) {
    inclusionPattern = convertRegexListToPattern(inclusions);
    exclusionPattern = convertRegexListToPattern(exclusions);
    hashCode = Objects.hash(inclusions, exclusions);
  }

  /**
   * Converts list of regex expressions into one compiled regex expression.
   */
  private static Pattern convertRegexListToPattern(List<String> regexList) {
    if (regexList.isEmpty()) {
      return null;
    }
    // Wrap each individual regex in the independent group, combine them using '|' and
    // wrap in the non-capturing group.
    return Pattern.compile("(?:(?>" + Joiner.on(")|(?>").join(regexList) + "))");
  }

  /**
   * @return true iff given string is included (it is does not match exclusion
   *         pattern (if any) and matches inclusionPatter (if any).
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
