// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NoConfigTransition}. */
@RunWith(JUnit4.class)
public class NoConfigTransitionTest extends BuildViewTestCase {
  // Custom rule that self-transitions to NoConfigTransition.
  private static final MockRule NO_CONFIG_RULE =
      () ->
          MockRule.define(
              "no_config_rule", (builder, env) -> builder.cfg(NoConfigTransition.INSTANCE));

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder().addRuleDefinition(NO_CONFIG_RULE);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void noConfigTransitionPreventsConfiguredTargetForking() throws Exception {
    // Write a custom Starlark rule that arbitrarily transitions its configuration. Have two
    // instances of that rule transition to different configurations and each depend on the same
    // no_config_rule. We expect there to be only one instance of the no_config_rule.
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//...',",
        "    ],",
        ")");
    scratch.file(
        "foo/defs.bzl",
        "def _my_transition_impl(settings, attr):",
        "  return {'//command_line_option:compiler': attr.compiler}",
        "my_transition = transition(",
        "  implementation = _my_transition_impl,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:compiler'],",
        ")",
        "",
        "transition_compiler_rule = rule(",
        "  implementation = lambda ctx: [],",
        "  cfg = my_transition,",
        "  attrs = {",
        "    'compiler': attr.string(),",
        "    'dep': attr.label(),",
        "    '_allowlist_function_transition':",
        "      attr.label(default = '//tools/allowlists/function_transition_allowlist'),",
        "  },",
        ")");
    scratch.file(
        "foo/BUILD",
        "load(':defs.bzl', 'transition_compiler_rule')",
        "no_config_rule(name = 'config_free_target')",
        "transition_compiler_rule(",
        "    name = 'parent1',",
        "    compiler = 'one',",
        "    dep = ':config_free_target',",
        ")",
        "transition_compiler_rule(",
        "    name = 'parent2',",
        "    compiler = 'two',",
        "    dep = ':config_free_target',",
        ")");

    ConfiguredTarget parent1 = getConfiguredTarget("//foo:parent1");
    ConfiguredTarget parent2 = getConfiguredTarget("//foo:parent2");

    assertThat(parent1.getConfigurationKey()).isNotEqualTo(parent2.getConfigurationKey());
    assertThat(getDirectPrerequisite(parent1, "//foo:config_free_target"))
        .isSameInstanceAs(getDirectPrerequisite(parent2, "//foo:config_free_target"));
  }
}
