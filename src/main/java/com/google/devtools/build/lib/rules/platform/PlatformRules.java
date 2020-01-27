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
// limitations under the License.
package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.PlatformConfigurationLoader;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformBootstrap;

/**
 * Rules for supporting different platforms in Bazel.
 */
public class PlatformRules implements RuleSet {
  public static final PlatformRules INSTANCE = new PlatformRules();

  protected PlatformRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationOptions(PlatformOptions.class);
    builder.addConfigurationFragment(new PlatformConfigurationLoader());

    builder.addRuleDefinition(new PlatformBaseRule());
    builder.addRuleDefinition(new ConstraintSettingRule());
    builder.addRuleDefinition(new ConstraintValueRule());
    builder.addRuleDefinition(new PlatformRule());

    builder.addRuleDefinition(new ToolchainRule());

    builder.addSkylarkBootstrap(new PlatformBootstrap(new PlatformCommon()));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE);
  }
}
