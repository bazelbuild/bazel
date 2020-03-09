// Copyright 2017 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.Whitelist;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.syntax.Starlark;
import java.util.List;
import java.util.Optional;

/**
 * The implementation of the config_feature_flag rule for defining custom flags for Android rules.
 */
public class ConfigFeatureFlag implements RuleConfiguredTargetFactory {
  /** The name of the policy that is used to restrict access to the config_feature_flag rule. */
  private static final String WHITELIST_NAME = "config_feature_flag";

  /** The label of the policy that is used to restrict access to the config_feature_flag rule. */
  private static final String WHITELIST_LABEL =
      "//tools/whitelists/config_feature_flag:config_feature_flag";

  /** Constructs a definition for the attribute used to restrict access to config_feature_flag. */
  public static Attribute.Builder<Label> getWhitelistAttribute(RuleDefinitionEnvironment env) {
    return Whitelist.getAttributeFromWhitelistName(WHITELIST_NAME)
        .value(env.getToolsLabel(WHITELIST_LABEL));
  }

  /**
   * Constructs a definition for the attribute used to restrict access to config_feature_flag. The
   * whitelist will only be reached if the given {@code attributeToInspect} has a value explicitly
   * specified. It must be non-configurable.
   */
  public static Attribute.Builder<Label> getWhitelistAttribute(
      RuleDefinitionEnvironment env, String attributeToInspect) {
    final Label label = env.getToolsLabel(WHITELIST_LABEL);
    return Whitelist.getAttributeFromWhitelistName(WHITELIST_NAME)
        .value(
            new ComputedDefault() {
              @Override
              public Label getDefault(AttributeMap rule) {
                return rule.isAttributeValueExplicitlySpecified(attributeToInspect) ? label : null;
              }
            });
  }

  /**
   * Returns whether config_feature_flag and related features are available to the current rule.
   *
   * <p>The current rule must have an attribute defined on it created with {@link
   * #getWhitelistAttribute}.
   */
  public static boolean isAvailable(RuleContext ruleContext) {
    return Whitelist.isAvailable(ruleContext, WHITELIST_NAME);
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    if (!ConfigFeatureFlag.isAvailable(ruleContext)) {
      throw ruleContext.throwWithRuleError(
          String.format(
              "the %s rule is not available in package '%s'",
              ruleContext.getRuleClassNameForLogging(),
              ruleContext.getLabel().getPackageIdentifier()));
    }

    List<String> specifiedValues = ruleContext.attributes().get("allowed_values", STRING_LIST);
    ImmutableSet<String> values = ImmutableSet.copyOf(specifiedValues);
    Predicate<String> isValidValue = Predicates.in(values);
    if (values.size() != specifiedValues.size()) {
      ImmutableMultiset<String> groupedValues = ImmutableMultiset.copyOf(specifiedValues);
      ImmutableList.Builder<String> duplicates = new ImmutableList.Builder<String>();
      for (Multiset.Entry<String> value : groupedValues.entrySet()) {
        if (value.getCount() > 1) {
          duplicates.add(value.getElement());
        }
      }
      ruleContext.attributeError(
          "allowed_values",
          "cannot contain duplicates, but contained multiple of "
              + Starlark.repr(duplicates.build()));
    }

    Optional<String> defaultValue =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("default_value")
            ? Optional.of(ruleContext.attributes().get("default_value", STRING))
            : Optional.empty();
    if (defaultValue.isPresent() && !isValidValue.apply(defaultValue.get())) {
      ruleContext.attributeError(
          "default_value",
          "must be one of "
              + Starlark.repr(values.asList())
              + ", but was "
              + Starlark.repr(defaultValue.get()));
    }

    if (ruleContext.hasErrors()) {
      // Don't bother validating the value if the flag was already incorrectly specified without
      // looking at the value.
      return null;
    }

    Optional<String> configuredValue =
        ruleContext
            .getFragment(ConfigFeatureFlagConfiguration.class)
            .getFeatureFlagValue(ruleContext.getOwner());

    if (configuredValue.isPresent() && !isValidValue.apply(configuredValue.get())) {
      // TODO(b/140635901): When configurationError is available, use that instead.
      ruleContext.ruleError(
          "value must be one of "
              + Starlark.repr(values.asList())
              + ", but was "
              + Starlark.repr(configuredValue.get()));
      return null;
    }

    if (!configuredValue.isPresent() && !defaultValue.isPresent()) {
      // TODO(b/140635901): When configurationError is available, use that instead.
      ruleContext.ruleError("flag has no default and must be set, but was not set");
      return null;
    }

    String value = configuredValue.orElseGet(defaultValue::get);

    ConfigFeatureFlagProvider provider = ConfigFeatureFlagProvider.create(value, isValidValue);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(provider)
        .build();
  }
}
