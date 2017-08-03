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
// limitations under the License.
package com.google.devtools.build.lib.rules.core;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.Builder;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.featurecontrol.FeaturePolicyLoader;
import com.google.devtools.build.lib.analysis.featurecontrol.FeaturePolicyOptions;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlag;
import com.google.devtools.build.lib.rules.test.TestConfiguration;

/** A set of basic rules - Bazel won't work correctly without these. */
public final class CoreRules implements RuleSet {
  public static final CoreRules INSTANCE = new CoreRules();

  public static final ImmutableSet<String> FEATURE_POLICY_FEATURES =
      ImmutableSet.of(ConfigFeatureFlag.POLICY_NAME);

  private CoreRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(Builder builder) {
    builder.addConfigurationOptions(FeaturePolicyOptions.class);
    builder.addConfigurationFragment(new FeaturePolicyLoader(FEATURE_POLICY_FEATURES));
    builder.addDynamicTransitionMaps(BaseRuleClasses.DYNAMIC_TRANSITIONS_MAP);

    builder.addConfig(TestConfiguration.TestOptions.class, new TestConfiguration.Loader());
    builder.addRuleDefinition(new BaseRuleClasses.RootRule());
    builder.addRuleDefinition(new BaseRuleClasses.BaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.RuleBase());
    builder.addRuleDefinition(new BaseRuleClasses.MakeVariableExpandingRule());
    builder.addRuleDefinition(new BaseRuleClasses.BinaryBaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.TestBaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.ErrorRule());
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of();
  }
}
