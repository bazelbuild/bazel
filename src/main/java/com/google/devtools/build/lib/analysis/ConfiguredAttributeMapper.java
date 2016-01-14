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
package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuildType.Selector;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@link AttributeMap} implementation that binds a rule's attribute as follows:
 *
 * <ol>
 *   <li>If the attribute is selectable (i.e. its BUILD declaration is of the form
 *   "attr = { config1: "value1", "config2: "value2", ... }", returns the subset of values
 *   chosen by the current configuration in accordance with Bazel's documented policy on
 *   configurable attribute selection.
 *   <li>If the attribute is not selectable (i.e. its value is static), returns that value with
 *   no additional processing.
 * </ol>
 *
 * <p>Example usage:
 * <pre>
 *   Label fooLabel = ConfiguredAttributeMapper.of(ruleConfiguredTarget).get("foo", Type.LABEL);
 * </pre>
 */
public class ConfiguredAttributeMapper extends AbstractAttributeMapper {

  private final Map<Label, ConfigMatchingProvider> configConditions;
  private Rule rule;

  private ConfiguredAttributeMapper(Rule rule, Set<ConfigMatchingProvider> configConditions) {
    super(Preconditions.checkNotNull(rule).getPackage(), rule.getRuleClassObject(), rule.getLabel(),
        rule.getAttributeContainer());
    ImmutableMap.Builder<Label, ConfigMatchingProvider> builder = ImmutableMap.builder();
    for (ConfigMatchingProvider configCondition : configConditions) {
      builder.put(configCondition.label(), configCondition);
    }
    this.configConditions = builder.build();
    this.rule = rule;
  }

  /**
   * "Do-it-all" constructor that just needs a {@link RuleConfiguredTarget}.
   */
  public static ConfiguredAttributeMapper of(RuleConfiguredTarget ct) {
    return new ConfiguredAttributeMapper(ct.getTarget(), ct.getConfigConditions());
  }

  /**
   * "Manual" constructor that requires the caller to pass the set of configurability conditions
   * that trigger this rule's configurable attributes.
   *
   * <p>If you don't know how to do this, you really want to use one of the "do-it-all"
   * constructors.
   */
  @VisibleForTesting
  public static ConfiguredAttributeMapper of(
      Rule rule, Set<ConfigMatchingProvider> configConditions) {
    return new ConfiguredAttributeMapper(rule, configConditions);
  }

  /**
   * Checks that all attributes can be mapped to their configured values. This is
   * useful for checking that the configuration space in a configured attribute doesn't
   * contain unresolvable contradictions.
   *
   * @throws EvalException if any attribute's value can't be resolved under this mapper
   */
  public void validateAttributes() throws EvalException {
    for (String attrName : getAttributeNames()) {
      getAndValidate(attrName, getAttributeType(attrName));
    }
  }

  /**
   * Variation of {@link #get} that throws an informative exception if the attribute
   * can't be resolved due to intrinsic contradictions in the configuration.
   */
  private <T> T getAndValidate(String attributeName, Type<T> type) throws EvalException  {
    SelectorList<T> selectorList = getSelectorList(attributeName, type);
    if (selectorList == null) {
      // This is a normal attribute.
      return super.get(attributeName, type);
    }

    List<T> resolvedList = new ArrayList<>();
    for (Selector<T> selector : selectorList.getSelectors()) {
      resolvedList.add(resolveSelector(attributeName, selector));
    }
    return resolvedList.size() == 1 ? resolvedList.get(0) : type.concat(resolvedList);
  }

  private <T> T resolveSelector(String attributeName, Selector<T> selector)
      throws EvalException {
    ConfigMatchingProvider matchingCondition = null;
    Set<Label> conditionLabels = new LinkedHashSet<>();
    T matchingValue = null;

    // Find the matching condition and record its value (checking for duplicates).
    for (Map.Entry<Label, T> entry : selector.getEntries().entrySet()) {
      Label selectorKey = entry.getKey();
      if (BuildType.Selector.isReservedLabel(selectorKey)) {
        continue;
      }

      ConfigMatchingProvider curCondition = Verify.verifyNotNull(configConditions.get(
          rule.getLabel().resolveRepositoryRelative(selectorKey)));
      conditionLabels.add(curCondition.label());

      if (curCondition.matches()) {
        if (matchingCondition == null || curCondition.refines(matchingCondition)) {
          // A match is valid if either this is the *only* condition that matches or this is a
          // more "precise" specification of another matching condition (in which case we choose
          // the most precise one).
          matchingCondition = curCondition;
          matchingValue = entry.getValue();
        } else if (matchingCondition.refines(curCondition)) {
          // The originally matching conditions is more precise, so keep that one.
        } else {
          throw new EvalException(rule.getAttributeLocation(attributeName),
              "Both " + matchingCondition.label() + " and " + curCondition.label()
              + " match configurable attribute \"" + attributeName + "\" in " + getLabel()
              + ". Multiple matches are not allowed unless one is a specialization of the other");
        }
      }
    }

    // If nothing matched, choose the default condition.
    if (matchingCondition == null) {
      if (!selector.hasDefault()) {
        throw new EvalException(rule.getAttributeLocation(attributeName),
            "Configurable attribute \"" + attributeName + "\" doesn't match this "
            + "configuration (would a default condition help?).\nConditions checked:\n "
            + Joiner.on("\n ").join(conditionLabels));
      }
      matchingValue = selector.getDefault();
    }

    return matchingValue;
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    try {
      return getAndValidate(attributeName, type);
    } catch (EvalException e) {
      // Callers that reach this branch should explicitly validate the attribute through an
      // appropriate call and handle the exception directly. This method assumes
      // pre-validated attributes.
      throw new IllegalStateException(
          "lookup failed on attribute " + attributeName + ": " + e.getMessage());
    }
  }
}
