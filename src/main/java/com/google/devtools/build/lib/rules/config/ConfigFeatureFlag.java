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
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;

/**
 * The implementation of the config_feature_flag rule for defining custom flags for Android rules.
 */
public class ConfigFeatureFlag implements RuleConfiguredTargetFactory {
  /**
   * The name of the policy that is used to restrict access to the config_feature_flag rule and
   * attribute-triggered access to the feature flags setter transition.
   */
  public static final String ALLOWLIST_NAME = "config_feature_flag";

  /** The label of the policy for ALLOWLIST_NAME. */
  private static final String ALLOWLIST_LABEL =
      "//tools/allowlists/config_feature_flag:config_feature_flag";

  /** Constructs a definition for the attribute used to restrict access to config_feature_flag. */
  public static Attribute.Builder<Label> getAllowlistAttribute(RuleDefinitionEnvironment env) {
    return Allowlist.getAttributeFromAllowlistName(ALLOWLIST_NAME)
        .value(env.getToolsLabel(ALLOWLIST_LABEL));
  }

  /**
   * Constructs a definition for the attribute used to restrict access to config_feature_flag. The
   * allowlist will only be reached if the given {@code attributeToInspect} has a value explicitly
   * specified.
   */
  public static Attribute.Builder<Label> getAllowlistAttribute(
      RuleDefinitionEnvironment env, String attributeToInspect) {
    final Label label = env.getToolsLabel(ALLOWLIST_LABEL);
    return Allowlist.getAttributeFromAllowlistName(ALLOWLIST_NAME)
        .value(
            /**
             * Critically, get is never actually called on attributeToInspect and thus it is not
             * necessary to declare whether it is configurable, for this context.
             */
            new ComputedDefault() {
              @Nullable
              @Override
              public Label getDefault(AttributeMap rule) {
                return rule.isAttributeValueExplicitlySpecified(attributeToInspect) ? label : null;
              }
            });
  }

  /**
   * The name of the policy that is used to restrict access to rule definitions attaching the
   * feature flag setting transition.
   *
   * <p>Defined here for consistency with ALLOWLIST_NAME policy.
   */
  public static final String SETTER_ALLOWLIST_NAME = "config_feature_flag_setter";

  /** The label of the policy for SETTER_ALLOWLIST_NAME. */
  private static final String SETTER_ALLOWLIST_LABEL =
      "//tools/allowlists/config_feature_flag:config_feature_flag_setter";

  /** Constructs a definition for the attribute used to restrict access to config_feature_flag. */
  public static Attribute.Builder<Label> getSetterAllowlistAttribute(
      RuleDefinitionEnvironment env) {
    return Allowlist.getAttributeFromAllowlistName(SETTER_ALLOWLIST_NAME)
        .value(env.getToolsLabel(SETTER_ALLOWLIST_LABEL));
  }

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    List<String> specifiedValues = ruleContext.attributes().get("allowed_values", STRING_LIST);
    ImmutableSet<String> values = ImmutableSet.copyOf(specifiedValues);
    if (values.size() != specifiedValues.size()) {
      ImmutableMultiset<String> groupedValues = ImmutableMultiset.copyOf(specifiedValues);
      ImmutableList.Builder<String> duplicates = new ImmutableList.Builder<>();
      for (Multiset.Entry<String> value : groupedValues.entrySet()) {
        if (value.getCount() > 1) {
          duplicates.add(value.getElement());
        }
      }
      // This is a problem with attributes of config_feature_flag itself so throw error here.
      ruleContext.attributeError(
          "allowed_values",
          "cannot contain duplicates, but contained multiple of "
              + Starlark.repr(duplicates.build()));
    }

    Optional<String> defaultValue =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("default_value")
            ? Optional.of(ruleContext.attributes().get("default_value", STRING))
            : Optional.empty();
    if (defaultValue.isPresent() && !values.contains(defaultValue.get())) {
      // This is a problem with attributes of config_feature_flag itself so throw error here.
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

    Object rawStarlarkValue =
        ruleContext
            .getConfiguration()
            .getOptions()
            .getStarlarkOptions()
            .get(ruleContext.getLabel());
    if (!(rawStarlarkValue instanceof FeatureFlagValue)) {
      // Retain legacy behavior of treating feature flags that somehow are not FeatureFlagValue
      // as set to default
      rawStarlarkValue = FeatureFlagValue.DefaultValue.INSTANCE;
    }

    @Nullable
    ConfigFeatureFlagProvider provider =
        constructProvider((FeatureFlagValue) rawStarlarkValue, defaultValue, values, ruleContext);
    if (provider == null) {
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(provider)
        .build();
  }

  /**
   * Calculate and return a ConfigFeatureFlagProvider.
   *
   * <p>At this point any errors here are due to something being 'wrong' with the configuration. In
   * particular, this provider may be constructed BEFORE a rule transition sets the expected
   * configuration and thus want to defer errors until later in analysis when this value is actually
   * consumed as the errors would otherwise be unfixable.
   *
   * <p>An exception is made for if the value is explicitly set to value not in the allowed values
   * list (either on the commandline or as part of a previous transition). In that case, that is an
   * immediate error as can be fixed by ensuring those places set to allowed values. This is
   * consistent with the behavior of build_setting.
   */
  @Nullable
  private static ConfigFeatureFlagProvider constructProvider(
      FeatureFlagValue featureFlagValue,
      Optional<String> defaultValue,
      ImmutableSet<String> values,
      RuleContext ruleContext) {
    Predicate<String> isValidValue = Predicates.in(values);
    if (featureFlagValue instanceof FeatureFlagValue.SetValue) {
      String setValue = ((FeatureFlagValue.SetValue) featureFlagValue).value();
      if (!isValidValue.apply(setValue)) {
        // This is consistent with build_setting, which also immediate checks that
        // explicitly set values are valid values.
        ruleContext.ruleError(
            "value must be one of "
                + Starlark.repr(values.asList())
                + ", but was "
                + Starlark.repr(setValue));
        return null;
      }
      return ConfigFeatureFlagProvider.create(setValue, null, isValidValue);
    } else if (featureFlagValue.equals(FeatureFlagValue.DefaultValue.INSTANCE)) {
      if (!defaultValue.isPresent()) {
        // Should defer error in case value is set by upcoming rule transition.
        // (Although, rule authors could just add a default.)
        // build_setting always has a default so this can't happen for them.
        return ConfigFeatureFlagProvider.create(
            null,
            String.format(
                "Feature flag %s has no default but no value was explicitly specified.",
                ruleContext.getLabel()),
            isValidValue);
      }
      return ConfigFeatureFlagProvider.create(defaultValue.get(), null, isValidValue);
    } else if (featureFlagValue.equals(FeatureFlagValue.UnknownValue.INSTANCE)) {
      // Must defer error in case value is set by upcoming rule transition.
      // build_setting doesn't have trimming logic so this can't happen for them.
      return ConfigFeatureFlagProvider.create(
          null,
          String.format(
              "Feature flag %1$s was accessed in a configuration it is not present in. All "
                  + "targets which depend on %1$s directly or indirectly must name it in their "
                  + "transitive_configs attribute.",
              ruleContext.getLabel()),
          isValidValue);
    } else {
      throw new IllegalStateException("Impossible state for FeatureFlagValue: " + featureFlagValue);
    }
  }
}
