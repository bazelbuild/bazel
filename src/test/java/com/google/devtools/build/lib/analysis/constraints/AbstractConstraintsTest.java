// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Common functionality for tests for the constraint enforcement system.
 */
public abstract class AbstractConstraintsTest extends BuildViewTestCase {
  /**
   * Creates an environment group on the scratch filesystem consisting of the specified
   * environments and specified defaults, set via a builder-style interface. The package name
   * is the same as the group name.
   */
  protected class EnvironmentGroupMaker {
    private final String name;
    private Set<String> environments;
    private Set<String> defaults;
    private final Multimap<String, String> fulfillsMap = HashMultimap.create();

    public EnvironmentGroupMaker(String name) {
      this.name = name;
    }

    public EnvironmentGroupMaker setEnvironments(String... environments) {
      this.environments = ImmutableSet.copyOf(environments);
      return this;
    }

    public EnvironmentGroupMaker setDefaults(String... environments) {
      this.defaults = ImmutableSet.copyOf(environments);
      return this;
    }

    /**
     * Declares that env1 fulfills env2.
     */
    public EnvironmentGroupMaker setFulfills(String env1, String env2) {
      fulfillsMap.put(env1, env2);
      return this;
    }

    protected final void make() throws Exception {
      StringBuilder builder = new StringBuilder();
      for (String env : environments) {
        builder.append("environment(name = '" + env + "',\n")
            .append(getAttrDef("fulfills", fulfillsMap.get(env).toArray(new String[0])))
            .append(")\n");
      }
      String envGroupName = name.contains("/") ? name.substring(name.lastIndexOf("/") + 1) : name;
      builder.append("environment_group(\n")
          .append("    name = '" + envGroupName + "',\n")
          .append(getAttrDef("environments", environments.toArray(new String[0])) + ",\n")
          .append(getAttrDef("defaults", defaults.toArray(new String[0])) + ",\n")
          .append(")");
      scratch.file("" + name + "/BUILD", builder.toString());
    }
  }

  /**
   * Returns a rule definition of the given name, type and custom attribute settings.
   */
  protected static String getRuleDef(String ruleType, String ruleName, String... customAttributes) {
    String ruleDef =
        ruleType + "(\n"
        + "    name = '" + ruleName + "',\n"
        + "    srcs = ['" + ruleName + ".sh'],\n";
    for (String customAttribute : customAttributes) {
      ruleDef += "    " + customAttribute + ",\n";
    }
    ruleDef += ")\n";
    return ruleDef;
  }

  /**
   * Given the inputs, returns the string "attrName = [':label1', ':label2', etc.]"
   */
  protected static String getAttrDef(String attrName, String... labels) {
    String attrDef = "    " + attrName + " = [";
    for (String label : labels) {
      attrDef += "'" + label + "', ";
    }
    attrDef += "]";
    return attrDef;
  }

  /**
   * The core constraint semantics check that if rule A depends on rule B, B must support all of
   * A's environments. To model this in the tests below, we construct two rules: a "depending"
   * rule (i.e. A) that depends on a "dependency" rule (i.e. B). Each test can construct its
   * own instance of these rules with its own environments specifications by calling this method
   * and {@link #getDependencyRule} with appropriate environment settings passed in as custom
   * attributes.
   *
   * <p>This method constructs and returns the depending rule (i.e. A).
   */
  protected static String getDependingRule(String... customAttributes) {
    List<String> attrsAsList = Lists.newArrayList(customAttributes);
    attrsAsList.add(getAttrDef("deps", "dep"));
    return getRuleDef("sh_library", "main", attrsAsList.toArray(new String[0]));
  }

  /**
   * Returns the rule that {@link #getDependingRule} depends on. This rule must support every
   * environment supported by the other one for their constraint relationship to be considered
   * valid.
   */
  protected static String getDependencyRule(String... customAttributes) {
    return getRuleDef("sh_library", "dep", customAttributes);
  }

  /**
   * Returns the attribute definition that constrains a rule to the given environments. Inputs
   * are expected to be package-relative labels (e.g. {@code "foo_env"}).
   */
  protected static String constrainedTo(String... environments) {
    return getAttrDef("restricted_to", environments);
  }

  /**
   * Returns the attribute definition that designates a rule compatible with the given environments.
   */
  protected static String compatibleWith(String... environments) {
    return getAttrDef("compatible_with", environments);
  }

  /**
   * Returns the environments supported by a rule.
   */
  protected Collection<Label> supportedEnvironments(String ruleName, String ruleDef)
      throws Exception {
    return (new RuleContextConstraintSemantics())
        .getSupportedEnvironments(
            getRuleContext(scratchConfiguredTarget("hello", ruleName, ruleDef)))
        .getEnvironments();
  }
}
