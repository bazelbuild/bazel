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
package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.bazel.rules.genrule.BazelGenRuleRule;
import com.google.devtools.build.lib.rules.ToolchainType.ToolchainTypeRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.genrule.GenRuleBaseRule;

/** Rules for toolchain support in Bazel. */
public class ToolchainRules implements RuleSet {
  public static final ToolchainRules INSTANCE = new ToolchainRules();

  private ToolchainRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addRuleDefinition(new ToolchainTypeRule());
    builder.addRuleDefinition(new GenRuleBaseRule());
    builder.addRuleDefinition(new BazelGenRuleRule());
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE, JavaRules.INSTANCE);
  }
}
