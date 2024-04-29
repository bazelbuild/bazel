// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.FunctionTransitionUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NonconfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;

/**
 * A transition factory for trimming feature flags manually via an attribute which specifies the
 * feature flags used by transitive dependencies.
 */
public class ConfigFeatureFlagTaggedTrimmingTransitionFactory
    implements TransitionFactory<RuleTransitionData> {

  /** Applies manual trimming to the given set of flags. */
  public static final class ConfigFeatureFlagTaggedTrimmingTransition implements PatchTransition {
    public static final ConfigFeatureFlagTaggedTrimmingTransition EMPTY =
        new ConfigFeatureFlagTaggedTrimmingTransition(ImmutableSortedSet.of());

    private final ImmutableSortedSet<Label> flags;
    private final int cachedHashCode;

    ConfigFeatureFlagTaggedTrimmingTransition(ImmutableSortedSet<Label> flags) {
      this.flags = flags;
      this.cachedHashCode = this.flags.hashCode();
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(ConfigFeatureFlagOptions.class, CoreOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      if (!(options.contains(ConfigFeatureFlagOptions.class)
          && options.get(ConfigFeatureFlagOptions.class)
              .enforceTransitiveConfigsForConfigFeatureFlag)) {
        return options.underlying();
      }
      BuildOptions toOptions = FeatureFlagValue.trimFlagValues(options.underlying(), flags);
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
      return other
              instanceof
              ConfigFeatureFlagTaggedTrimmingTransition configFeatureFlagTaggedTrimmingTransition
          && this.flags.equals(configFeatureFlagTaggedTrimmingTransition.flags);
    }

    @Override
    public int hashCode() {
      return cachedHashCode;
    }

    @Override
    public String toString() {
      return String.format("ConfigFeatureFlagTaggedTrimmingTransition{flags=%s}", flags);
    }
  }

  private final String attributeName;

  public ConfigFeatureFlagTaggedTrimmingTransitionFactory(String attributeName) {
    this.attributeName = attributeName;
  }

  @Override
  public PatchTransition create(RuleTransitionData ruleData) {
    NonconfiguredAttributeMapper attrs = NonconfiguredAttributeMapper.of(ruleData.rule());
    RuleClass ruleClass = ruleData.rule().getRuleClassObject();

    if (AliasProvider.mayBeAlias(ruleData.rule())) {
      // As a convenience, do not require transitive_config to be set for alias rule.
      return NoTransition.INSTANCE;
    }

    if (ruleClass.getName().equals(ConfigRuleClasses.ConfigFeatureFlagRule.RULE_NAME)) {
      return new ConfigFeatureFlagTaggedTrimmingTransition(
          ImmutableSortedSet.of(ruleData.rule().getLabel()));
    }

    ImmutableSortedSet.Builder<Label> requiredLabelsBuilder =
        new ImmutableSortedSet.Builder<>(Ordering.natural());
    if (attrs.isAttributeValueExplicitlySpecified(attributeName)
        && !attrs.get(attributeName, NODEP_LABEL_LIST).isEmpty()) {
      // Entries starting with //command_line_option[:/] represent native options and are not
      // relevant for this transition. Non-existent flags already do not error so this skipping
      // is done out of an abundance of caution and as a statement of intent for the future.
      for (Label entry : attrs.get(attributeName, NODEP_LABEL_LIST)) {
        String packageName = entry.getPackageName();
        if (packageName.equals("command_line_option")
            || packageName.startsWith("command_line_option/")) {
          continue;
        }
        requiredLabelsBuilder.add(entry);
      }
    }
    if (ruleClass.getTransitionFactory() instanceof ConfigFeatureFlagTransitionFactory) {
      String settingAttribute =
          ((ConfigFeatureFlagTransitionFactory) ruleClass.getTransitionFactory())
              .getAttributeName();
      // Because the process of setting a flag also creates a dependency on that flag, we need to
      // include all the set flags, even if they aren't actually declared as used by this rule.
      requiredLabelsBuilder.addAll(attrs.get(settingAttribute, LABEL_KEYED_STRING_DICT).keySet());
    }

    ImmutableSortedSet<Label> requiredLabels = requiredLabelsBuilder.build();
    if (requiredLabels.isEmpty()) {
      return ConfigFeatureFlagTaggedTrimmingTransition.EMPTY;
    }

    return new ConfigFeatureFlagTaggedTrimmingTransition(requiredLabels);
  }

  @Override
  public TransitionType transitionType() {
    return TransitionType.RULE;
  }
}
