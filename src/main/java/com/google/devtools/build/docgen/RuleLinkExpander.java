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

import com.google.common.collect.ImmutableSet;

import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;

/**
 * Helper class used for expanding link references in rule and attribute documentation.
 *
 * <p>See {@link com.google.devtools.build.docgen.DocgenConsts.BLAZE_RULE_LINK} for the regex used
 * to match link references.
 */
class RuleLinkExpander {
  private static final String EXAMPLES_SUFFIX = "_examples";
  private static final String ARGS_SUFFIX = "_args";

  private static final Set<String> STATIC_PAGES = ImmutableSet.<String>of(
      "common-definitions",
      "make-variables",
      "predefined-python-variables");

  private final Map<String, String> ruleIndex;

  RuleLinkExpander(Map<String, String> ruleIndex) {
    this.ruleIndex = ruleIndex;
  }

  private void appendRuleLink(Matcher matcher, StringBuffer sb, String ruleName, String ref) {
    String ruleFamily = ruleIndex.get(ruleName);
    String link = ruleFamily + ".html#" + ref;
    matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
  }

  /**
   * Expands all rule references in the input HTML documentation.
   *
   * @param htmlDoc The input HTML documentation with ${link foo.bar} references.
   * @return The HTML documentation with all link references expanded.
   */
  public String expand(String htmlDoc) throws IllegalArgumentException {
    Matcher matcher = DocgenConsts.BLAZE_RULE_LINK.matcher(htmlDoc);
    StringBuffer sb = new StringBuffer(htmlDoc.length());
    while (matcher.find()) {
      // The first capture group matches the entire reference, e.g. "cc_binary.deps".
      String ref = matcher.group(1);
      // The second capture group only matches the rule name, e.g. "cc_binary" in "cc_binary.deps".
      String name = matcher.group(2);

      // The name in the reference is the name of a rule. Get the rule family for the rule and
      // replace the reference with a link with the form of rule-family.html#rule.attribute. For
      // example, ${link cc_library.deps} expands to c-cpp.html#cc_library.deps.
      if (ruleIndex.containsKey(name)) {
        appendRuleLink(matcher, sb, name, ref);
        continue;
      }

      // The name is referencing the examples or arguments of a rule (e.g. "cc_library_args" or
      // "cc_library_examples"). Strip the suffix and then try matching the name to a rule family.
      if (name.endsWith(EXAMPLES_SUFFIX) || name.endsWith(ARGS_SUFFIX)) {
        String ruleName = name.substring(
            0, name.indexOf(name.endsWith(EXAMPLES_SUFFIX) ? EXAMPLES_SUFFIX : ARGS_SUFFIX));
        if (ruleIndex.containsKey(ruleName)) {
          appendRuleLink(matcher, sb, ruleName, ref);
          continue;
        }
      }

      // The name is not the name of a rule but is the name of a static page, such as
      // common-definitions. Generate a link to that page, and append the page heading if
      // specified. For example, ${link common-definitions.label-expansion} expands to
      // common-definitions.html#label-expansion.
      if (STATIC_PAGES.contains(name)) {
        String link = name + ".html";
        // The fourth capture group matches the attribute name, or page heading, e.g.
        // "label-expansion" in "common-definitions.label-expansion".
        String pageHeading = matcher.group(4);
        if (pageHeading != null) {
          link = link + "#" + pageHeading;
        }
        matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
        continue;
      }

      // If the reference does not match any rule or static page, throw an exception.
      throw new IllegalArgumentException(
          "link tag does not match any rule or BE page: " + matcher.group());
    }
    matcher.appendTail(sb);
    return sb.toString();
  }
}
