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
package com.google.devtools.build.docgen;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;

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
class RuleDocumentationAttribute implements Comparable<RuleDocumentationAttribute> {

  private final Class<? extends RuleDefinition> definitionClass;
  private final String attributeName;
  private final String htmlDocumentation;
  private final String commonType;
  private int startLineCnt;
  private Set<String> flags;

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
   * Returns the name of the rule attribute.
   */
  String getAttributeName() {
    return attributeName;
  }

  /**
   * Returns the raw html documentation of the rule attribute.
   */
  String getHtmlDocumentation() {
    return htmlDocumentation;
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
  boolean isCommonType() {
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
   * RuleDocumentationAttribute in the Google3RuleDefinition ancestry graph. Returns -1
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
    BlazeRule ann = usingClass.getAnnotation(BlazeRule.class);
    if (ann != null) {
      for (Class<? extends RuleDefinition> ancestor : ann.ancestors()) {
        if (!visited.containsKey(ancestor)) {
          toVisit.addLast(ancestor);
          visited.put(ancestor, visited.get(usingClass) + 1);
        }
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
