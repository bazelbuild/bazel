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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;

/**
 * Helper class used for expanding link references in rule and attribute documentation.
 *
 * <p>See {@link com.google.devtools.build.docgen.DocgenConsts.BLAZE_RULE_LINK} for the regex used
 * to match link references.
 */
public class RuleLinkExpander {
  private static final String EXAMPLES_SUFFIX = "_examples";
  private static final String ARGS_SUFFIX = "_args";
  private static final String IMPLICIT_OUTPUTS_SUFFIX = "_implicit_outputs";
  private static final String FUNCTIONS_PAGE = "functions";

  private static final ImmutableSet<String> STATIC_PAGES =
      ImmutableSet.<String>of(
          "common-definitions", "make-variables", "predefined-python-variables");
  private static final ImmutableMap<String, String> FUNCTIONS =
      ImmutableMap.<String, String>builder()
          .put("load", FUNCTIONS_PAGE)
          .put("package", FUNCTIONS_PAGE)
          .put("package_group", FUNCTIONS_PAGE)
          .put("description", FUNCTIONS_PAGE)
          .put("distribs", FUNCTIONS_PAGE)
          .put("licenses", FUNCTIONS_PAGE)
          .put("exports_files", FUNCTIONS_PAGE)
          .put("glob", FUNCTIONS_PAGE)
          .put("select", FUNCTIONS_PAGE)
          .put("workspace", FUNCTIONS_PAGE)
          .build();

  private final String productName;
  private final Map<String, String> ruleIndex = new HashMap<>();
  private final boolean singlePage;

  RuleLinkExpander(String productName, Map<String, String> ruleIndex, boolean singlePage) {
    this.productName = productName;
    this.ruleIndex.putAll(ruleIndex);
    this.ruleIndex.putAll(FUNCTIONS);
    this.singlePage = singlePage;
  }

  RuleLinkExpander(String productName, boolean singlePage) {
    this.productName = productName;
    this.ruleIndex.putAll(FUNCTIONS);
    this.singlePage = singlePage;
  }

  public void addIndex(Map<String, String> ruleIndex) {
    this.ruleIndex.putAll(ruleIndex);
  }

  private void appendRuleLink(Matcher matcher, StringBuffer sb, String ruleName, String ref) {
    String ruleFamily = ruleIndex.get(ruleName);
    String link = singlePage
        ?  "#" + ref
        : ruleFamily + ".html#" + ref;
    matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
  }

  /*
   * Match and replace all ${link rule.attribute} references.
   */
  private String expandRuleLinks(String htmlDoc) throws IllegalArgumentException {
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

      // The name is referencing the examples, arguments, or implicit outputs of a rule (e.g.
      // "cc_library_args", "cc_library_examples", or "java_binary_implicit_outputs"). Strip the
      // suffix and then try matching the name to a rule family.
      if (name.endsWith(EXAMPLES_SUFFIX)
          || name.endsWith(ARGS_SUFFIX)
          || name.endsWith(IMPLICIT_OUTPUTS_SUFFIX)) {
        int endIndex;
        if (name.endsWith(EXAMPLES_SUFFIX)) {
          endIndex = name.indexOf(EXAMPLES_SUFFIX);
        } else if (name.endsWith(ARGS_SUFFIX)) {
          endIndex = name.indexOf(ARGS_SUFFIX);
        } else {
          endIndex = name.indexOf(IMPLICIT_OUTPUTS_SUFFIX);
        }
        String ruleName = name.substring(0, endIndex);
        if (ruleIndex.containsKey(ruleName)) {
          appendRuleLink(matcher, sb, ruleName, ref);
          continue;
        }
      }

      // The name is not the name of a rule but is the name of a static page, such as
      // common-definitions. Generate a link to that page.
      if (STATIC_PAGES.contains(name)) {
        String link = singlePage
            ? "#" + name
            : name + ".html";
        // For referencing headings on a static page, use the following syntax:
        // ${link static_page_name#heading_name}, example: ${link make-variables#gendir}
        String pageHeading = matcher.group(4);
        if (pageHeading != null) {
          throw new IllegalArgumentException(
              "Invalid link syntax for BE page: " + matcher.group()
              + "\nUse ${link static-page#heading} syntax instead.");
        }
        matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
        continue;
      }

      // If the reference does not match any rule or static page, throw an exception.
      throw new IllegalArgumentException(
          "Rule family " + name + " in link tag does not match any rule or BE page: "
          + matcher.group());
    }
    matcher.appendTail(sb);
    return sb.toString();
  }

  /*
   * Match and replace all ${link rule#heading} references.
   */
  private String expandRuleHeadingLinks(String htmlDoc) throws IllegalArgumentException {
    Matcher matcher = DocgenConsts.BLAZE_RULE_HEADING_LINK.matcher(htmlDoc);
    StringBuffer sb = new StringBuffer(htmlDoc.length());
    while (matcher.find()) {
      // The second capture group only matches the rule name, e.g. "cc_library" in
      // "cc_library#some_heading"
      String name = matcher.group(2);
      // The third capture group only matches the heading, e.g. "some_heading" in
      // "cc_library#some_heading"
      String heading = matcher.group(3);

      // The name in the reference is the name of a rule. Get the rule family for the rule and
      // replace the reference with the link in the form of rule-family.html#heading. Examples of
      // this include custom <a name="heading"> tags in the description or examples for the rule.
      if (ruleIndex.containsKey(name)) {
        String ruleFamily = ruleIndex.get(name);
        String link = singlePage
            ? "#" + heading
            : ruleFamily + ".html#" + heading;
        matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
        continue;
      }

      // The name is of a static page, such as common.definitions. Generate a link to that page, and
      // append the page heading. For example, ${link common-definitions#label-expansion} expands to
      // common-definitions.html#label-expansion.
      if (STATIC_PAGES.contains(name)) {
        String link = singlePage
            ? "#" + heading
            : name + ".html#" + heading;
        matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
        continue;
      }

      // Links to the user manual are handled specially. Meh.
      if ("user-manual".equals(name)) {
        String link = productName.toLowerCase(Locale.US) + "-" + name + ".html#" + heading;
        matcher.appendReplacement(sb, Matcher.quoteReplacement(link));
        continue;
      }

      // If the reference does not match any rule or static page, throw an exception.
      throw new IllegalArgumentException(
          "Rule family " + name + " in link tag does not match any rule or BE page: "
          + matcher.group());
    }
    matcher.appendTail(sb);
    return sb.toString();
  }

  /**
   * Expands all rule references in the input HTML documentation.
   *
   * @param htmlDoc The input HTML documentation with ${link foo.bar} references.
   * @return The HTML documentation with all link references expanded.
   */
  public String expand(String htmlDoc) throws IllegalArgumentException {
    String expanded = expandRuleLinks(htmlDoc);
    return expandRuleHeadingLinks(expanded);
  }

  /**
   * Expands the rule reference.
   *
   * <p>This method is used to expand references in the BE velocity templates.
   *
   * @param ref The rule reference to expand.
   * @return The expanded rule reference.
   */
  public String expandRef(String ref) throws IllegalArgumentException {
    return expand("${link " + ref + "}");
  }
}
