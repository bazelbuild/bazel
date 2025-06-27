// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.outputfilter;

import com.google.common.collect.ImmutableMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Represents a single, parsed output suppression rule. */
final class SuppressionRule {
  private static final Pattern KEYWORD_PATTERN = Pattern.compile("(\\w+):(\\S+)");

  private final ImmutableMap<String, String> keywords;
  private final Pattern pattern;
  private final int expectedCount;

  private SuppressionRule(
      ImmutableMap<String, String> keywords, Pattern pattern, int expectedCount) {
    this.keywords = keywords;
    this.pattern = pattern;
    this.expectedCount = expectedCount;
  }

  static SuppressionRule create(String ruleString) throws IllegalArgumentException {
    Matcher matcher = KEYWORD_PATTERN.matcher(ruleString);
    ImmutableMap.Builder<String, String> keywords = ImmutableMap.builder();
    int lastMatchEnd = 0;
    while (matcher.find()) {
      keywords.put(matcher.group(1), matcher.group(2));
      lastMatchEnd = matcher.end();
    }

    String regex = ruleString.substring(lastMatchEnd).trim();
    if (regex.isEmpty()) {
      throw new IllegalArgumentException("Suppression rule must have a pattern to match.");
    }

    ImmutableMap<String, String> parsedKeywords = keywords.build();
    int expectedCount = -1;
    if (parsedKeywords.containsKey("count")) {
      expectedCount = Integer.parseInt(parsedKeywords.get("count"));
    }

    return new SuppressionRule(parsedKeywords, Pattern.compile(regex), expectedCount);
  }

  Pattern getPattern() {
    return pattern;
  }

  int getExpectedCount() {
    return expectedCount;
  }

  boolean hasKeyword(String keyword) {
    return keywords.containsKey(keyword);
  }

  String getKeywordValue(String keyword) {
    return keywords.get(keyword);
  }
}
