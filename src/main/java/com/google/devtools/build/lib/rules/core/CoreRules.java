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
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.test.TestTrimmingTransitionFactory;

/** A set of basic rules - Bazel won't work correctly without these. */
public final class CoreRules implements RuleSet {
  public static final CoreRules INSTANCE = new CoreRules();

  private CoreRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.setShouldInvalidateCacheForOptionDiff(
        TestConfiguration.SHOULD_INVALIDATE_FOR_OPTION_DIFF);
    builder.addConfigurationFragment(TestConfiguration.class);
    builder.addTrimmingTransitionFactory(new TestTrimmingTransitionFactory());
    builder.addRuleDefinition(new BaseRuleClasses.NativeBuildRule());
    builder.addRuleDefinition(new BaseRuleClasses.NativeActionCreatingRule());
    builder.addRuleDefinition(new BaseRuleClasses.MakeVariableExpandingRule());
    builder.addRuleDefinition(new BaseRuleClasses.BinaryBaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.TestBaseRule());
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of();
  }
}
