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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.packages.Type;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

/**
 * A class storing a rule attribute documentation along with some meta information. The class
 * provides functionality to compute the ancestry level of this attribute's generator rule
 * definition class compared to other rule definition classes.
 *
 * <p>Warning, two RuleDocumentationAttribute objects are equal based on only the attributeName.
 */
public class RuleDocumentationAttribute
    implements Comparable<RuleDocumentationAttribute>, Cloneable {

  private static final ImmutableMap<Type<?>, String> TYPE_DESC =
      ImmutableMap.<Type<?>, String>builder()
          .put(Type.BOOLEAN, "Boolean")
          .put(Type.INTEGER, "Integer")
          .put(Type.INTEGER_LIST, "List of integers")
          .put(Type.STRING, "String")
          .put(Type.STRING_DICT, "Dictionary: String -> String")
          .put(Type.STRING_LIST, "List of strings")
          .put(BuildType.TRISTATE, "Integer")
          .put(BuildType.LABEL, "<a href=\"../build-ref.html#labels\">Label</a>")
          .put(BuildType.LABEL_LIST, "List of <a href=\"../build-ref.html#labels\">labels</a>")
          .put(
              BuildType.LABEL_DICT_UNARY,
              "Dictionary mapping strings to <a href=\"../build-ref.html#labels\">labels</a>")
          .put(BuildType.LICENSE, "Licence type")
          .put(BuildType.NODEP_LABEL, "<a href=\"../build-ref.html#name\">Name</a>")
          .put(BuildType.NODEP_LABEL_LIST, "List of <a href=\"../build-ref.html#name\">names</a>")
          .put(BuildType.OUTPUT, "<a href=\"../build-ref.html#filename\">Filename</a>")
          .put(
              BuildType.OUTPUT_LIST, "List of <a href=\"../build-ref.html#filename\">filenames</a>")
          .build();

  private final Class<? extends RuleDefinition> definitionClass;
  private final String attributeName;
  private final String htmlDocumentation;
  private final String commonType;
  // Used to expand rule link references in the attribute documentation.
  private RuleLinkExpander linkExpander;
  private int startLineCnt;
  private String fileName;
  private Set<String> flags;
  private Attribute attribute;


  /**
   * Creates common RuleDocumentationAttribute such as deps or data.
   * These attribute docs have no definitionClass or htmlDocumentation (it's in the BE header).
   */
  static RuleDocumentationAttribute create(
      String attributeName, String commonType, String htmlDocumentation) {
    RuleDocumentationAttribute docAttribute = new RuleDocumentationAttribute(
        null, attributeName, htmlDocumentation, 0, "", ImmutableSet.<String>of(), commonType);
    return docAttribute;
  }

  /**
   * Creates a RuleDocumentationAttribute with all the necessary fields for explicitly
   * defined rule attributes.
   */
  static RuleDocumentationAttribute create(Class<? extends RuleDefinition> definitionClass,
      String attributeName, String htmlDocumentation, int startLineCnt, String fileName,
      Set<String> flags) {
    return new RuleDocumentationAttribute(definitionClass, attributeName, htmlDocumentation,
        startLineCnt, fileName, flags, null);
  }

  private RuleDocumentationAttribute(Class<? extends RuleDefinition> definitionClass,
      String attributeName, String htmlDocumentation, int startLineCnt, String fileName,
      Set<String> flags, String commonType) {
    Preconditions.checkNotNull(attributeName, "AttributeName must not be null.");
    this.definitionClass = definitionClass;
    this.attributeName = attributeName;
    this.htmlDocumentation = htmlDocumentation;
    this.startLineCnt = startLineCnt;
    this.flags = flags;
    this.commonType = commonType;
    this.fileName = fileName;
  }

  @Override
  protected Object clone() throws CloneNotSupportedException {
    return super.clone();
  }

  /**
   * Sets the Attribute object that this documents.
   */
  void setAttribute(Attribute attribute) {
    this.attribute = attribute;
  }

  /**
   * Returns the name of the rule attribute.
   */
  public String getAttributeName() {
    return attributeName;
  }

  /** Returns the file name where the rule attribute is defined. */
  public String getFileName() {
    return fileName;
  }

  /**
   * Returns whether this attribute is marked as deprecated.
   */
  public boolean isDeprecated() {
    return hasFlag(DocgenConsts.FLAG_DEPRECATED);
  }

  /**
   * Sets the {@link RuleLinkExpander} to be used to expand links in the HTML documentation.
   */
  public void setRuleLinkExpander(RuleLinkExpander linkExpander) {
    this.linkExpander = linkExpander;
  }

  /**
   * Returns the html documentation of the rule attribute.
   */
  public String getHtmlDocumentation() throws BuildEncyclopediaDocException {
    String expandedHtmlDoc = htmlDocumentation;
    if (linkExpander != null) {
      try {
        expandedHtmlDoc = linkExpander.expand(expandedHtmlDoc);
      } catch (IllegalArgumentException e) {
        throw new BuildEncyclopediaDocException(fileName, startLineCnt, e.getMessage());
      }
    }
    return expandedHtmlDoc;
  }

  /** Returns whether the param is required or optional. */
  public boolean isMandatory() {
    if (attribute == null) {
      return false;
    }
    return attribute.isMandatory();
  }

  private String getDefaultValue() {
    if (attribute == null) {
      return "";
    }
    String prefix = "; default is ";
    Object value = attribute.getDefaultValueUnchecked();
    if (value instanceof Boolean) {
      return prefix + ((Boolean) value ? "True" : "False");
    } else if (value instanceof Integer) {
      return prefix + String.valueOf(value);
    } else if (value instanceof String && !((String) value).isEmpty()) {
      return prefix + "\"" + value + "\"";
    } else if (value instanceof TriState) {
      switch((TriState) value) {
        case AUTO:
          return prefix + "-1";
        case NO:
          return prefix + "0";
        case YES:
          return prefix + "1";
      }
    } else if (value instanceof Label) {
      return prefix + "<code>" + value + "</code>";
    }
    return "";
  }

  /**
   * Returns a string containing the synopsis for this attribute.
   */
  public String getSynopsis() {
    if (attribute == null) {
      return "";
    }
    StringBuilder sb =
        new StringBuilder()
            .append(TYPE_DESC.get(attribute.getType()))
            .append("; ")
            .append(attribute.isMandatory() ? "required" : "optional")
            .append(
                !attribute.isConfigurable()
                    ? String.format(
                        "; <a href=\"%s#configurable-attributes\">nonconfigurable</a>",
                        RuleDocumentation.COMMON_DEFINITIONS_PAGE)
                    : "");
    if (!attribute.isMandatory()) {
      sb.append(getDefaultValue());
    }
    return sb.toString();
  }

  /**
   * Returns the number of first line of the attribute documentation in its declaration file.
   */
  int getStartLineCnt() {
    return startLineCnt;
  }

  /**
   * Returns true if the attribute doc is of a common attribute type.
   */
  public boolean isCommonType() {
    return commonType != null;
  }

  /**
   * Returns the common attribute type if this attribute doc is of a common type
   * otherwise actualRule.
   */
  String getGeneratedInRule(String actualRule) {
    return isCommonType() ? commonType : actualRule;
  }

  /**
   * Returns true if this attribute documentation has the parameter flag.
   */
  boolean hasFlag(String flag) {
    return flags.contains(flag);
  }

  /**
   * Returns the length of a shortest path from usingClass to the definitionClass of this
   * RuleDocumentationAttribute in the rule definition ancestry graph. Returns -1
   * if definitionClass is not the ancestor (transitively) of usingClass.
   */
  int getDefinitionClassAncestryLevel(Class<? extends RuleDefinition> usingClass,
      ConfiguredRuleClassProvider ruleClassProvider) {
    if (usingClass.equals(definitionClass)) {
      return 0;
    }
    // Storing nodes (rule class definitions) with the length of the shortest path from usingClass
    Map<Class<? extends RuleDefinition>, Integer> visited = new HashMap<>();
    LinkedList<Class<? extends RuleDefinition>> toVisit = new LinkedList<>();
    visited.put(usingClass, 0);
    toVisit.add(usingClass);
    // Searching the shortest path from usingClass to this.definitionClass using BFS
    do {
      Class<? extends RuleDefinition> ancestor = toVisit.removeFirst();
      visitAncestor(ancestor, visited, toVisit, ruleClassProvider);
      if (ancestor.equals(definitionClass)) {
        return visited.get(ancestor);
      }
    } while (!toVisit.isEmpty());
    return -1;
  }

  private void visitAncestor(
      Class<? extends RuleDefinition> usingClass,
      Map<Class<? extends RuleDefinition>, Integer> visited,
      LinkedList<Class<? extends RuleDefinition>> toVisit,
      ConfiguredRuleClassProvider ruleClassProvider) {
    RuleDefinition instance = getRuleDefinition(usingClass, ruleClassProvider);
    for (Class<? extends RuleDefinition> ancestor : instance.getMetadata().ancestors()) {
      if (!visited.containsKey(ancestor)) {
        toVisit.addLast(ancestor);
        visited.put(ancestor, visited.get(usingClass) + 1);
      }
    }
  }

  private RuleDefinition getRuleDefinition(Class<? extends RuleDefinition> usingClass,
      ConfiguredRuleClassProvider ruleClassProvider) {
    if (ruleClassProvider == null) {
      try {
        return usingClass.getConstructor().newInstance();
      } catch (ReflectiveOperationException | IllegalArgumentException e) {
        throw new IllegalStateException(e);
      }
    }
    return ruleClassProvider.getRuleClassDefinition(usingClass.getName());
  }

  private int getAttributeOrderingPriority(RuleDocumentationAttribute attribute) {
    if (DocgenConsts.ATTRIBUTE_ORDERING.containsKey(attribute.attributeName)) {
      return DocgenConsts.ATTRIBUTE_ORDERING.get(attribute.attributeName);
    } else {
      return 0;
    }
  }

  @Override
  public int compareTo(RuleDocumentationAttribute o) {
    int thisPriority = getAttributeOrderingPriority(this);
    int otherPriority = getAttributeOrderingPriority(o);
    if (thisPriority > otherPriority) {
      return 1;
    } else if (thisPriority < otherPriority) {
      return -1;
    } else {
      return this.attributeName.compareTo(o.attributeName);
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RuleDocumentationAttribute)) {
      return false;
    }
    return attributeName.equals(((RuleDocumentationAttribute) obj).attributeName);
  }

  @Override
  public int hashCode() {
    return attributeName.hashCode();
  }
}
