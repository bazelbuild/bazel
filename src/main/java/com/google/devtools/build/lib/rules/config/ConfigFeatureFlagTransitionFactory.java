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

import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.FunctionTransitionUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AllowlistChecker;
import com.google.devtools.build.lib.packages.NonconfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import java.util.Map;

/**
 * Transition factory which allows for setting the values of config_feature_flags below the rule it
 * is attached to based on one of that rule's attributes.
 *
 * <p>Currently, this is only intended for use by android_binary and other Android top-level rules.
 */
public class ConfigFeatureFlagTransitionFactory
    implements StarlarkExposedRuleTransitionFactory, ConfigurationTransitionApi {
  @Override
  public void addToRuleFromStarlark(RuleDefinitionEnvironment ctx, RuleClass.Builder builder) {
    builder.add(ConfigFeatureFlag.getAllowlistAttribute(ctx, attributeName));
    builder.addAllowlistChecker(
        AllowlistChecker.builder()
            .setAllowlistAttr(ConfigFeatureFlag.ALLOWLIST_NAME)
            .setErrorMessage("the attribute " + attributeName + " is not available in this package")
            .setLocationCheck(AllowlistChecker.LocationCheck.INSTANCE)
            .setAttributeSetTrigger(attributeName)
            .build());
    builder.add(ConfigFeatureFlag.getSetterAllowlistAttribute(ctx));
    builder.addAllowlistChecker(
        AllowlistChecker.builder()
            .setAllowlistAttr(ConfigFeatureFlag.SETTER_ALLOWLIST_NAME)
            .setErrorMessage(
                "the rule class is not allowed access to feature flags setter transition")
            .setLocationCheck(AllowlistChecker.LocationCheck.DEFINITION)
            .setAttributeSetTrigger(attributeName)
            .build());
  }

  /** Transition which resets the set of flag-value pairs to the map it was constructed with. */
  private static final class ConfigFeatureFlagValuesTransition implements PatchTransition {
    private final ImmutableSortedMap<Label, String> flagValues;
    private final int cachedHashCode;

    public ConfigFeatureFlagValuesTransition(Map<Label, String> flagValues) {
      this(ImmutableSortedMap.copyOf(flagValues), flagValues.hashCode());
    }

    ConfigFeatureFlagValuesTransition(
        ImmutableSortedMap<Label, String> flagValues, int cachedHashCode) {
      this.flagValues = ImmutableSortedMap.copyOf(flagValues);
      this.cachedHashCode = cachedHashCode;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(ConfigFeatureFlagOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      if (!options.contains(ConfigFeatureFlagOptions.class)) {
        return options.underlying();
      }
      BuildOptions toOptions = FeatureFlagValue.replaceFlagValues(options.underlying(), flagValues);
      // In legacy mode, need to update `affected by Starlark transition` to include changed flags.
      if (toOptions
          .get(CoreOptions.class)
          .outputDirectoryNamingScheme
          .equals(CoreOptions.OutputDirectoryNamingScheme.LEGACY)) {
        FunctionTransitionUtil.updateAffectedByStarlarkTransition(
            toOptions.get(CoreOptions.class),
            FunctionTransitionUtil.getAffectedByStarlarkTransitionViaDiff(
                toOptions, options.underlying()));
      }
      return toOptions;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof ConfigFeatureFlagValuesTransition configFeatureFlagValuesTransition
          && this.flagValues.equals(configFeatureFlagValuesTransition.flagValues);
    }

    @Override
    public int hashCode() {
      return cachedHashCode;
    }

    @Override
    public String toString() {
      return String.format("ConfigFeatureFlagValuesTransition{flagValues=%s}", flagValues);
    }
  }

  private final String attributeName;

  /**
   * Creates a transition factory which will generate a transition over a given rule which sets
   * exactly the flag values in the attribute with the given {@code attributeName} of that rule,
   * unsetting any flag values not listed there.
   *
   * <p>This attribute must not be a configured {@code LABEL_KEYED_STRING_DICT}. (No selects)
   */
  public ConfigFeatureFlagTransitionFactory(String attributeName) {
    this.attributeName = attributeName;
  }

  @Override
  public PatchTransition create(RuleTransitionData ruleData) {
    NonconfiguredAttributeMapper attrs = NonconfiguredAttributeMapper.of(ruleData.rule());
    if (attrs.isAttributeValueExplicitlySpecified(attributeName)) {
      return new ConfigFeatureFlagValuesTransition(
          attrs.get(attributeName, LABEL_KEYED_STRING_DICT));
    } else {
      return NoTransition.INSTANCE;
    }
  }

  @Override
  public TransitionType transitionType() {
    return TransitionType.RULE;
  }

  /**
   * Returns the attribute examined by this transition factory.
   */
  public String getAttributeName() {
    return this.attributeName;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ConfigFeatureFlagTransitionFactory configFeatureFlagTransitionFactory
        && this.attributeName.equals(configFeatureFlagTransitionFactory.attributeName);
  }

  @Override
  public int hashCode() {
    return attributeName.hashCode();
  }

  @Override
  public String toString() {
    return String.format("ConfigFeatureFlagTransitionFactory{attributeName=%s}", attributeName);
  }
}
