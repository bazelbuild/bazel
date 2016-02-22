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
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.syntax.Type;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

/**
 * A class storing a rule attribute documentation along with some meta information.
 * The class provides functionality to compute the ancestry level of this attribute's
 * generator rule definition class compared to other rule definition classes.
 *
 * <p>Warning, two RuleDocumentationAttribute objects are equal based on only the attributeName.
 */
public class RuleDocumentationAttribute implements Comparable<RuleDocumentationAttribute> {

  private static final Map<Type<?>, String> TYPE_DESC = ImmutableMap.<Type<?>, String>builder()
      .put(Type.BOOLEAN, "Boolean")
      .put(Type.INTEGER, "Integer")
      .put(Type.INTEGER_LIST, "List of integers")
      .put(Type.STRING, "String")
      .put(Type.STRING_LIST, "List of strings")
      .put(BuildType.TRISTATE, "Integer")
      .put(BuildType.LABEL, "<a href=\"../build-ref.html#labels\">Label</a>")
      .put(BuildType.LABEL_LIST, "List of <a href=\"../build-ref.html#labels\">labels</a>")
      .put(BuildType.LABEL_DICT_UNARY,
          "Dictionary mapping strings to <a href=\"../build-ref.html#labels\">labels</a>")
      .put(BuildType.LABEL_LIST_DICT,
          "Dictionary mapping strings to lists of <a href=\"../build-ref.html#labels\">labels</a>")
      .put(BuildType.NODEP_LABEL, "<a href=\"../build-ref.html#name\">Name</a>")
      .put(BuildType.NODEP_LABEL_LIST, "List of <a href=\"../build-ref.html#name\">names</a>")
      .put(BuildType.OUTPUT, "<a href=\"../build-ref.html#filename\">Filename</a>")
      .put(BuildType.OUTPUT_LIST, "List of <a href=\"../build-ref.html#filename\">filenames</a>")
      .build();

  private final Class<? extends RuleDefinition> definitionClass;
  private final String attributeName;
  private final String htmlDocumentation;
  private final String commonType;
  private int startLineCnt;
  private Set<String> flags;
  private Attribute attribute;

  /**
   * Creates common RuleDocumentationAttribute such as deps or data.
   * These attribute docs have no definitionClass or htmlDocumentation (it's in the BE header).
   */
  static RuleDocumentationAttribute create(
      String attributeName, String commonType, String htmlDocumentation) {
    RuleDocumentationAttribute docAttribute = new RuleDocumentationAttribute(
        null, attributeName, htmlDocumentation, 0, ImmutableSet.<String>of(), commonType);
    return docAttribute;
  }

  /**
   * Creates a RuleDocumentationAttribute with all the necessary fields for explicitly
   * defined rule attributes.
   */
  static RuleDocumentationAttribute create(Class<? extends RuleDefinition> definitionClass,
      String attributeName, String htmlDocumentation, int startLineCnt, Set<String> flags) {
    return new RuleDocumentationAttribute(definitionClass, attributeName, htmlDocumentation,
        startLineCnt, flags, null);
  }

  private RuleDocumentationAttribute(Class<? extends RuleDefinition> definitionClass,
      String attributeName, String htmlDocumentation, int startLineCnt, Set<String> flags,
      String commonType) {
    Preconditions.checkNotNull(attributeName, "AttributeName must not be null.");
    this.definitionClass = definitionClass;
    this.attributeName = attributeName;
    this.htmlDocumentation = htmlDocumentation;
    this.startLineCnt = startLineCnt;
    this.flags = flags;
    this.commonType = commonType;
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

  /**
   * Returns whether this attribute is marked as deprecated.
   */
  public boolean isDeprecated() {
    return hasFlag(DocgenConsts.FLAG_DEPRECATED);
  }

  /**
   * Returns the raw html documentation of the rule attribute.
   */
  public String getHtmlDocumentation() {
    return htmlDocumentation;
  }

  private String getDefaultValue() {
    if (attribute == null) {
      return "";
    }
    String prefix = "; default is ";
    Object value = attribute.getDefaultValueForTesting();
    if (value instanceof Boolean) {
      return prefix + ((Boolean) value ? "1" : "0");
    } else if (value instanceof Integer) {
      return prefix + String.valueOf(value);
    } else if (value instanceof String && !(((String) value).isEmpty())) {
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
    StringBuilder sb = new StringBuilder()
        .append(TYPE_DESC.get(attribute.getType()))
        .append("; " + (attribute.isMandatory() ? "required" : "optional"))
        .append(getDefaultValue());
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
  int getDefinitionClassAncestryLevel(Class<? extends RuleDefinition> usingClass) {
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
      visitAncestor(ancestor, visited, toVisit);
      if (ancestor.equals(definitionClass)) {
        return visited.get(ancestor);
      }
    } while (!toVisit.isEmpty());
    return -1;
  }

  private void visitAncestor(
      Class<? extends RuleDefinition> usingClass,
      Map<Class<? extends RuleDefinition>, Integer> visited,
      LinkedList<Class<? extends RuleDefinition>> toVisit) {
    RuleDefinition instance;
    try {
      instance = usingClass.newInstance();
    } catch (IllegalAccessException | InstantiationException e) {
      throw new IllegalStateException(e);
    }
    for (Class<? extends RuleDefinition> ancestor : instance.getMetadata().ancestors()) {
      if (!visited.containsKey(ancestor)) {
        toVisit.addLast(ancestor);
        visited.put(ancestor, visited.get(usingClass) + 1);
      }
    }
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
