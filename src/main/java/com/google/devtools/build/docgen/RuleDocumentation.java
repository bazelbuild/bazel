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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * A class representing the documentation of a rule along with some meta-data. The sole ruleName
 * field is used as a key for comparison, equals and hashcode.
 *
 * <p> The class contains meta information about the rule:
 * <ul>
 * <li> Rule type: categorizes the rule based on it's general (language independent) purpose,
 * see {@link RuleType}.
 * <li> Rule family: categorizes the rule based on language.
 * </ul>
 *
 * <p> The class also contains physical information about the documentation,
 * such as declaring file name and the first line of the raw documentation. This can be useful for
 * proper error signaling during documentation processing.
 */
public class RuleDocumentation implements Comparable<RuleDocumentation> {
  private final String ruleName;
  private final RuleType ruleType;
  private final String ruleFamily;
  private final String familySummary;
  private final String htmlDocumentation;
  // Store these information for error messages
  private final int startLineCount;
  private final String fileName;
  private final ImmutableSet<String> flags;

  private final Map<String, String> docVariables = new HashMap<>();
  // Only one attribute per attributeName is allowed
  private final Set<RuleDocumentationAttribute> attributes = new TreeSet<>();

  private RuleLinkExpander linkExpander;

  /**
   * Name of the page documenting common build rule terms and concepts.
   */
  static final String COMMON_DEFINITIONS_PAGE = "common-definitions.html";

  /**
   * Creates a RuleDocumentation from the rule's name, type, family and raw html documentation
   * (meaning without expanding the variables in the doc).
   */
  RuleDocumentation(
      String ruleName,
      String ruleType,
      String ruleFamily,
      String htmlDocumentation,
      int startLineCount,
      String fileName,
      ImmutableSet<String> flags,
      String familySummary)
      throws BuildEncyclopediaDocException {
    Preconditions.checkNotNull(ruleName);
    this.ruleName = ruleName;
    if (flags.contains(DocgenConsts.FLAG_GENERIC_RULE)) {
      this.ruleType = RuleType.OTHER;
    } else {
      try {
        this.ruleType = RuleType.valueOf(ruleType);
      } catch (IllegalArgumentException e) {
        throw new BuildEncyclopediaDocException(
            fileName, startLineCount, "Invalid rule type " + ruleType);
      }
    }
    this.ruleFamily = ruleFamily;
    this.htmlDocumentation = htmlDocumentation;
    this.startLineCount = startLineCount;
    this.fileName = fileName;
    this.flags = flags;
    this.familySummary = familySummary;
  }

  RuleDocumentation(
      String ruleName,
      String ruleType,
      String ruleFamily,
      String htmlDocumentation,
      int startLineCount,
      String fileName,
      ImmutableSet<String> flags)
      throws BuildEncyclopediaDocException {
    this(
        ruleName,
        ruleType,
        ruleFamily,
        htmlDocumentation,
        startLineCount,
        fileName,
        flags,
        "");
  }

  /**
   * Returns the name of the rule.
   */
  public String getRuleName() {
    return ruleName;
  }

  /**
   * Returns the type of the rule
   */
  RuleType getRuleType() {
    return ruleType;
  }

  /**
   * Returns the family of the rule. The family is usually the corresponding programming language,
   * except for rules independent of language, such as genrule. E.g. the family of the java_library
   * rule is 'JAVA', the family of genrule is 'GENERAL'.
   */
  String getRuleFamily() {
    return ruleFamily;
  }

  /**
   * Return the contribution of this rule to the summary for the rule family. Usually, the "main"
   * rule in a family provides the summary, but all contributions are accumulated.
   */
  String getFamilySummary() {
    return familySummary;
  }

  /**
   * Returns a "normalized" version of the input string. Used to convert rule family names into
   * strings that are more friendly as file names. For example, "C / C++" is converted to
   * "c-cpp".
   */
  @VisibleForTesting
  static String normalize(String s) {
    return s.toLowerCase()
        .replace("+", "p")
        .replaceAll("[()]", "")
        .replaceAll("[\\s/]", "-")
        .replaceAll("[-]+", "-");
  }

  /**
   * Returns the number of first line of the rule documentation in its declaration file.
   */
  int getStartLineCount() {
    return startLineCount;
  }

  /**
   * Returns true if this rule documentation has the parameter flag.
   */
  boolean hasFlag(String flag) {
    return flags.contains(flag);
  }

  /**
   * Returns true if this rule applies to a specific programming language (e.g. java_library),
   * returns false if it is a generic action (e.g. genrule, filegroup).
   *
   * <p>A rule is considered to be specific to a programming language by default. Generic rules
   * have to be marked with the flag GENERIC_RULE in their #BLAZE_RULE definition.
   */
  boolean isLanguageSpecific() {
    return !flags.contains(DocgenConsts.FLAG_GENERIC_RULE);
  }

  /**
   * Adds a variable name - value pair to the documentation to be substituted.
   */
  void addDocVariable(String varName, String value) {
    docVariables.put(varName, value);
  }

  /**
   * Adds a rule documentation attribute to this rule documentation.
   */
  void addAttribute(RuleDocumentationAttribute attribute) {
    attributes.add(attribute);
  }

  /**
   * Returns the rule's set of RuleDocumentationAttributes.
   */
  public Set<RuleDocumentationAttribute> getAttributes() {
    return attributes;
  }

  /**
   * Sets the {@link RuleLinkExpander} to be used to expand links in the HTML documentation for
   * both this RuleDocumentation and all {@link RuleDocumentationAttribute}s associated with this
   * rule.
   */
  public void setRuleLinkExpander(RuleLinkExpander linkExpander) {
    this.linkExpander = linkExpander;
    for (RuleDocumentationAttribute attribute : attributes) {
      attribute.setRuleLinkExpander(linkExpander);
    }
  }

  /**
   * Returns the html documentation in the exact format it should be written into the Build
   * Encyclopedia (expanding variables).
   */
  public String getHtmlDocumentation() throws BuildEncyclopediaDocException {
    String expandedDoc = htmlDocumentation;
    // Substituting variables
    for (Map.Entry<String, String> docVariable : docVariables.entrySet()) {
      expandedDoc = expandedDoc.replace("${" + docVariable.getKey() + "}",
          expandBuiltInVariables(docVariable.getKey(), docVariable.getValue()));
    }
    if (linkExpander != null) {
      try {
        expandedDoc = linkExpander.expand(expandedDoc);
      } catch (IllegalArgumentException e) {
        throw new BuildEncyclopediaDocException(fileName, startLineCount, e.getMessage());
      }
    }
    return expandedDoc;
  }

  /**
   * Returns the documentation of the rule in a form which is printable on the command line.
   */
  String getCommandLineDocumentation() {
    return "\n" + DocgenConsts.toCommandLineFormat(htmlDocumentation);
  }

  /**
   * Returns a string containing any extra documentation for the name attribute for this
   * rule.
   */
  public String getNameExtraHtmlDoc() throws BuildEncyclopediaDocException {
    String expandedDoc = docVariables.containsKey(DocgenConsts.VAR_NAME)
        ? docVariables.get(DocgenConsts.VAR_NAME)
        : "";
    if (linkExpander != null) {
      try {
        expandedDoc = linkExpander.expand(expandedDoc);
      } catch (IllegalArgumentException e) {
        throw new BuildEncyclopediaDocException(fileName, startLineCount, e.getMessage());
      }
    }
    return expandedDoc;
  }

  /**
   * Returns whether this rule is deprecated.
   */
  public boolean isDeprecated() {
    return hasFlag(DocgenConsts.FLAG_DEPRECATED);
  }

  /**
   * Returns a string containing the attribute signature for this rule with HTML links
   * to the attributes.
   */
  public String getAttributeSignature() {
    StringBuilder sb = new StringBuilder();
    sb.append(String.format("%s(<a href=\"#%s.name\">name</a>, ", ruleName, ruleName));
    int i = 0;
    for (RuleDocumentationAttribute attributeDoc : attributes) {
      String attrName = attributeDoc.getAttributeName();
      // Generate the link for the attribute documentation
      if (attributeDoc.isCommonType()) {
        sb.append(String.format("<a href=\"%s#%s.%s\">%s</a>",
            COMMON_DEFINITIONS_PAGE,
            attributeDoc.getGeneratedInRule(ruleName).toLowerCase(),
            attrName,
            attrName));
      } else {
        sb.append(String.format("<a href=\"#%s.%s\">%s</a>",
            attributeDoc.getGeneratedInRule(ruleName).toLowerCase(),
            attrName,
            attrName));
      }
      if (i < attributes.size() - 1) {
        sb.append(", ");
      } else {
        sb.append(")");
      }
      i++;
    }
    return sb.toString();
  }

  private String expandBuiltInVariables(String key, String value) {
    // Some built in BLAZE variables need special handling, e.g. adding headers
    switch (key) {
      case DocgenConsts.VAR_IMPLICIT_OUTPUTS:
        return String.format("<h4 id=\"%s_implicit_outputs\">Implicit output targets</h4>\n%s",
            ruleName.toLowerCase(), value);
      default:
        return value;
    }
  }

  /**
   * Returns a set of examples based on markups which can be used as BUILD file
   * contents for testing.
   */
  Set<String> extractExamples() throws BuildEncyclopediaDocException {
    String[] lines = htmlDocumentation.split(DocgenConsts.LS);
    Set<String> examples = new HashSet<>();
    StringBuilder sb = null;
    boolean inExampleCode = false;
    int lineCount = 0;
    for (String line : lines) {
      if (!inExampleCode) {
        if (DocgenConsts.BLAZE_RULE_EXAMPLE_START.matcher(line).matches()) {
          inExampleCode = true;
          sb = new StringBuilder();
        } else if (DocgenConsts.BLAZE_RULE_EXAMPLE_END.matcher(line).matches()) {
          throw new BuildEncyclopediaDocException(fileName, startLineCount + lineCount,
              "No matching start example tag (#BLAZE_RULE.EXAMPLE) for end example tag.");
        }
      } else {
        if (DocgenConsts.BLAZE_RULE_EXAMPLE_END.matcher(line).matches()) {
          inExampleCode = false;
          examples.add(sb.toString());
          sb = null;
        } else if (DocgenConsts.BLAZE_RULE_EXAMPLE_START.matcher(line).matches()) {
          throw new BuildEncyclopediaDocException(fileName, startLineCount + lineCount,
              "No start example tags (#BLAZE_RULE.EXAMPLE) in a row.");
        } else {
          sb.append(line + DocgenConsts.LS);
        }
      }
      lineCount++;
    }
    return examples;
  }

  /**
   * Creates a BuildEncyclopediaDocException with the file containing this rule doc and
   * the number of the first line (where the rule doc is defined). Can be used to create
   * general BuildEncyclopediaDocExceptions about this rule.
   */
  BuildEncyclopediaDocException createException(String msg) {
    return new BuildEncyclopediaDocException(fileName, startLineCount, msg);
  }

  @Override
  public int hashCode() {
    return ruleName.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RuleDocumentation)) {
      return false;
    }
    return ruleName.equals(((RuleDocumentation) obj).ruleName);
  }

  private int getTypePriority() {
    switch (ruleType) {
      case BINARY:
        return 1;
      case LIBRARY:
        return 2;
      case TEST:
        return 3;
      case OTHER:
        return 4;
    }
    throw new IllegalArgumentException("Illegal value of ruleType: " + ruleType);
  }

  @Override
  public int compareTo(RuleDocumentation o) {
    if (this.getTypePriority() < o.getTypePriority()) {
      return -1;
    } else if (this.getTypePriority() > o.getTypePriority()) {
      return 1;
    } else {
      return this.ruleName.compareTo(o.ruleName);
    }
  }

  @Override
  public String toString() {
    return String.format("%s (TYPE = %s, FAMILY = %s)", ruleName, ruleType, ruleFamily);
  }
}
