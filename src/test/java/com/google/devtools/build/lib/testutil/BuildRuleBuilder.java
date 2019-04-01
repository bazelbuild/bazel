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

package com.google.devtools.build.lib.testutil;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.packages.RuleClass;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility for quickly creating BUILD file rules for use in tests.
 *
 * <p>The use case for this class is writing BUILD files where simple
 * readability for the sake of rules' relationship to the test framework
 * is more important than detailed semantics and layout.
 *
 * <p>The behavior provided by this class is not meant to be exhaustive,
 * but should handle a majority of simple cases.
 *
 * <p>Example:
 *
 * <pre>
 *   String text = new BuildRuleBuilder("java_library", "MyRule")
        .setSources("First.java", "Second.java", "Third.java")
        .setDeps(":library", "//java/com/google/common/collect")
        .setResources("schema/myschema.xsd")
        .build();
 * </pre>
 *
 */
public class BuildRuleBuilder {
  protected final RuleClass ruleClass;
  protected final String ruleName;
  private Multimap<String, String> multiValueAttributes;
  private Map<String, Object> singleValueAttributes;
  protected Map<String, RuleClass> ruleClassMap;

  /**
   * Create a new instance.
   *
   * @param ruleClass the rule class of the new rule
   * @param ruleName the name of the new rule.
   */
  public BuildRuleBuilder(String ruleClass, String ruleName) {
    this(ruleClass, ruleName, getDefaultRuleClassMap());
  }

  public BuildRuleBuilder(String ruleClass, String ruleName, Map<String, RuleClass> ruleClassMap) {
    this.ruleClass = ruleClassMap.get(ruleClass);
    this.ruleName = ruleName;
    this.multiValueAttributes = LinkedHashMultimap.create();
    this.singleValueAttributes = new HashMap<>();
    this.ruleClassMap = ruleClassMap;
  }

  protected static Map<String, RuleClass> getDefaultRuleClassMap() {
    return TestRuleClassProvider.getRuleClassProvider().getRuleClassMap();
  }

  /** Sets the value of a single valued attribute */
  public BuildRuleBuilder setSingleValueAttribute(String attrName, Object value) {
    Preconditions.checkState(
        !singleValueAttributes.containsKey(attrName), "attribute '%s' already set", attrName);
    singleValueAttributes.put(attrName, value);
    return this;
  }

  /**
   * Sets the value of a list type attribute
   */
  public BuildRuleBuilder addMultiValueAttributes(String attrName, String... value) {
    multiValueAttributes.putAll(attrName, Lists.newArrayList(value));
    return this;
  }

  /**
   * Generate the rule
   *
   * @return a string representation of the rule.
   */
  public String build() {
    StringBuilder sb = new StringBuilder();
    sb.append(ruleClass.getName()).append("(");
    printNormal(sb, "name", ruleName);
    for (Map.Entry<String, Collection<String>> entry : multiValueAttributes.asMap().entrySet()) {
      printArray(sb, entry.getKey(), entry.getValue());
    }
    for (Map.Entry<String, Object> entry : singleValueAttributes.entrySet()) {
      printNormal(sb, entry.getKey(), entry.getValue());
    }
    sb.append(")\n");
    return sb.toString();
  }

  private void printArray(StringBuilder sb, String attr, Collection<String> values) {
    if (values == null || values.isEmpty()) {
      return;
    }
    sb.append("      ").append(attr).append(" = ");
    printList(sb, values);
    sb.append(",");
    sb.append("\n");
  }

  private void printNormal(StringBuilder sb, String attr, Object value) {
    if (value == null) {
      return;
    }
    sb.append("      ").append(attr).append(" = ");
    if (value instanceof Integer) {
      sb.append(value);
    } else {
      sb.append("'").append(value).append("'");
    }
    sb.append(",");
    sb.append("\n");
  }

  /**
   * Turns iterable of {a b c} into string "['a', 'b', 'c']", appends to
   * supplied StringBuilder.
   */
  private void printList(StringBuilder sb, Collection<String> elements) {
    sb.append("[");
    Joiner.on(",").appendTo(sb,
        Iterables.transform(elements, new Function<String, String>() {
          @Override
          public String apply(String from) {
            return "'" + from + "'";
          }
        }));
    sb.append("]");
  }

  /**
   * Returns the transitive closure of file names need to be generated in order
   * for this rule to build.
   */
  public Collection<String> getFilesToGenerate() {
    return ImmutableList.of();
  }

  /**
   * Returns the transitive closure of BuildRuleBuilders need to be generated in order
   * for this rule to build.
   */
  public Collection<BuildRuleBuilder> getRulesToGenerate() {
    return ImmutableList.of();
  }

  /**
   * Returns a {@link Dependency} of this {@link BuildRuleBuilder} using attrName.
   */
  public Dependency dependsVia(String attrName) {
    return new Dependency(this, attrName);
  }

  /**
   * Representing a {@link BuildRuleBuilder} depending on an other rule via a certain attribute.
   */
  public class Dependency {
    private BuildRuleBuilder buildRuleBuilder;
    private String attrName;

    private Dependency(BuildRuleBuilder buildRuleBuilder, String attrName) {
      this.buildRuleBuilder = buildRuleBuilder;
      this.attrName = attrName;
    }

    /**
     * Returns this {@link BuildRuleBuilder} with a new dependency on otherRule.
     */
    public BuildRuleBuilder on(BuildRuleBuilder otherRule) {
      buildRuleBuilder.addMultiValueAttributes(attrName, otherRule.ruleName);
      return buildRuleBuilder;
    }
  }
}
