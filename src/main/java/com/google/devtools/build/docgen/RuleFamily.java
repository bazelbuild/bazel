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

package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.common.escape.CharEscaperBuilder;
import com.google.common.escape.Escaper;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;
import java.util.Locale;

/**
 * Helper class for representing a rule family in the rule summary table template.
 *
 * <p>The rules are separated into categories by rule class: binary, library, test, flag, and other.
 */
@Immutable
public class RuleFamily {
  private static final Escaper FAMILY_NAME_ESCAPER = new CharEscaperBuilder()
      .addEscape('+', "p")
      .addEscapes(new char[] {'[', ']', '(', ')'}, "")
      .addEscapes(new char[] {' ', '/'}, "-")
      .toEscaper();

  private final String summary;
  private final String name;
  private final String id;

  private final ImmutableList<RuleDocumentation> binaryRules;
  private final ImmutableList<RuleDocumentation> libraryRules;
  private final ImmutableList<RuleDocumentation> testRules;
  private final ImmutableList<RuleDocumentation> otherRules;

  private final ImmutableList<RuleDocumentation> rules;
  private final ImmutableList<RuleDocumentation> flags;

  RuleFamily(ListMultimap<RuleType, RuleDocumentation> ruleTypeMap, String name, String summary) {
    this.name = name;
    this.id = normalize(name);
    this.summary = summary;
    this.binaryRules = ImmutableList.copyOf(ruleTypeMap.get(RuleType.BINARY));
    this.libraryRules = ImmutableList.copyOf(ruleTypeMap.get(RuleType.LIBRARY));
    this.testRules = ImmutableList.copyOf(ruleTypeMap.get(RuleType.TEST));
    this.otherRules = ImmutableList.copyOf(ruleTypeMap.get(RuleType.OTHER));

    rules =
        ImmutableList.<RuleDocumentation>builder()
            .addAll(binaryRules)
            .addAll(libraryRules)
            .addAll(testRules)
            .addAll(otherRules)
            .build();
    this.flags = ImmutableList.copyOf(ruleTypeMap.get(RuleType.FLAG));
  }

  /*
   * Returns a "normalized" version of the input string. Used to convert rule family names into
   * strings that are more friendly as file names. For example, "C / C++" is converted to
   * "c-cpp".
   */
  static String normalize(String s) {
    return FAMILY_NAME_ESCAPER.escape(s.toLowerCase(Locale.ROOT)).replaceAll("[-]+", "-");
  }

  public int size() {
    return rules.size();
  }

  public String getName() {
    return name;
  }

  public String getSummary() {
    return summary;
  }

  public String getId() {
    return id;
  }

  public List<RuleDocumentation> getBinaryRules() {
    return binaryRules;
  }

  public List<RuleDocumentation> getLibraryRules() {
    return libraryRules;
  }

  public List<RuleDocumentation> getTestRules() {
    return testRules;
  }

  public List<RuleDocumentation> getOtherRules() {
    return otherRules;
  }

  public List<RuleDocumentation> getRules() {
    return rules;
  }

  public List<RuleDocumentation> getFlags() {
    return flags;
  }
}
