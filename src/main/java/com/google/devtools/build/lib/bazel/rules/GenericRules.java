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
import com.google.devtools.build.lib.analysis.constraints.EnvironmentRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelFilegroupRule;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.Alias.AliasRule;
import com.google.devtools.build.lib.rules.LabelBuildSettings.LabelBuildFlagRule;
import com.google.devtools.build.lib.rules.LabelBuildSettings.LabelBuildSettingRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.genquery.GenQueryRule;
import com.google.devtools.build.lib.rules.starlarkdocextract.StarlarkDocExtractRule;
import com.google.devtools.build.lib.rules.test.TestSuiteRule;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import net.starlark.java.eval.FlagGuardedValue;

/**
 * A set of generic rules that provide miscellaneous capabilities to Bazel.
 */
public class GenericRules implements RuleSet {
  public static final GenericRules INSTANCE = new GenericRules();

  private GenericRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addRuleDefinition(new EnvironmentRule());

    builder.addRuleDefinition(new AliasRule());
    builder.addRuleDefinition(new BazelFilegroupRule());
    builder.addRuleDefinition(new TestSuiteRule());
    GenQueryRule.register(builder);
    builder.addRuleDefinition(new LabelBuildSettingRule());
    builder.addRuleDefinition(new LabelBuildFlagRule());
    builder.addRuleDefinition(new StarlarkDocExtractRule());

    try {
      builder.addWorkspaceFilePrefix(
          ResourceFileLoader.loadResource(BazelRuleClassProvider.class, "tools.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    // TODO(#11437): It'd be nice to hide this definition behind a static helper, but the most apt
    // place would be as a static method of InternalModule.java in lib.packages, and that package
    // can't accept a ConfiguredRuleClassProvider.Builder. The alternative is to use a Bootstrap,
    // but that idiom should probably be deprecated.
    builder.addBzlToplevel(
        "_builtins_dummy",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            BuildLanguageOptions.EXPERIMENTAL_BUILTINS_DUMMY, "original value"));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE);
  }
}
