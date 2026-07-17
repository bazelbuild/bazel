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
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.extra.ActionListenerRule;
import com.google.devtools.build.lib.rules.extra.ExtraActionRule;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaPluginsFlagAliasRule;
import com.google.devtools.build.lib.rules.java.JavaStarlarkCommon;
import net.starlark.java.eval.Starlark;

/** Rules for Java support in Bazel. */
public class JavaRules implements RuleSet {
  public static final JavaRules INSTANCE = new JavaRules();

  private JavaRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationFragment(JavaConfiguration.class);

    builder.addRuleDefinition(new JavaPluginsFlagAliasRule());

    builder.addRuleDefinition(new ExtraActionRule());
    builder.addRuleDefinition(new ActionListenerRule());

    builder.addBzlToplevel("java_common", Starlark.NONE);
    builder.addStarlarkBuiltinsInternal(
        "java_common_internal_do_not_use", new JavaStarlarkCommon(BazelJavaSemantics.INSTANCE));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
  }
}
