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
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * A class representing the documentation of a rule along with some meta-data. The sole ruleName
 * field is used as a key for comparison, equals and hashcode.
 *
 * <p>The class contains meta information about the rule:
 *
 * <ul>
 *   <li>Rule type: categorizes the rule based on it's general (language independent) purpose, such
 *       as "binary" or "library"; see {@link RuleType}.
 *   <li>Rule family: categorizes the rule based on language, such as "Java" or "C / C++".
 * </ul>
 *
 * <p>For generating error messages, the class also stores the location where raw documentation was
 * retrieved.
 */
public class RuleDocumentation implements Comparable<RuleDocumentation> {
  private final String ruleName;
  private final RuleType ruleType;
  private final String ruleFamily;
  private final String familySummary;
  private final String htmlDocumentation;
  private final String location; // for error messages
  private final String sourceUrl; // for linking in rendered docs
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
      String location,
      String sourceUrl,
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
        throw new BuildEncyclopediaDocException(location, "Invalid rule type " + ruleType);
      }
    }
    this.ruleFamily = ruleFamily;
    this.htmlDocumentation = htmlDocumentation;
    this.location = location;
    this.sourceUrl = sourceUrl;
    this.flags = flags;
    this.familySummary = familySummary;
  }

  RuleDocumentation(
      String ruleName,
      String ruleType,
      String ruleFamily,
      String htmlDocumentation,
      String fileName,
      int line,
      String sourceUrl,
      ImmutableSet<String> flags,
      String familySummary)
      throws BuildEncyclopediaDocException {
    this(
        ruleName,
        ruleType,
        ruleFamily,
        htmlDocumentation,
        BuildEncyclopediaDocException.formatLocation(fileName, line),
        sourceUrl,
        flags,
        familySummary);
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
    return Ascii.toLowerCase(s)
        .replace('+', 'p')
        .replaceAll("[()]", "")
        .replaceAll("[\\s/]", "-")
        .replaceAll("[-]+", "-");
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

  /** Adds multiple rule documentation attributes to this rule documentation. */
  void addAttributes(Collection<RuleDocumentationAttribute> attribute) {
    attributes.addAll(attribute);
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
        throw new BuildEncyclopediaDocException(location, e.getMessage());
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
        throw new BuildEncyclopediaDocException(location, e.getMessage());
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

  /** Returns the URL of the rule's source file in its source code repository. */
  public String getSourceUrl() {
    return sourceUrl;
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
        sb.append(
            String.format(
                "<a href=\"%s#%s.%s\">%s</a>",
                COMMON_DEFINITIONS_PAGE,
                Ascii.toLowerCase(attributeDoc.getGeneratedInRule(ruleName)),
                attrName,
                attrName));
      } else {
        sb.append(
            String.format(
                "<a href=\"#%s.%s\">%s</a>",
                Ascii.toLowerCase(attributeDoc.getGeneratedInRule(ruleName)), attrName, attrName));
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
    return switch (key) {
      case DocgenConsts.VAR_IMPLICIT_OUTPUTS ->
          String.format(
              "<h4 id=\"%s_implicit_outputs\">Implicit output targets</h4>\n%s",
              Ascii.toLowerCase(ruleName), value);
      default -> value;
    };
  }

  /**
   * Creates a BuildEncyclopediaDocException with the file containing this rule doc and
   * the number of the first line (where the rule doc is defined). Can be used to create
   * general BuildEncyclopediaDocExceptions about this rule.
   */
  BuildEncyclopediaDocException createException(String msg) {
    return new BuildEncyclopediaDocException(location, msg);
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
    return switch (ruleType) {
      case BINARY -> 1;
      case LIBRARY -> 2;
      case TEST -> 3;
      case OTHER -> 4;
    };
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
