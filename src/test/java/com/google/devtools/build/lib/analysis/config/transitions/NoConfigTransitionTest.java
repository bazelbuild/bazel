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
import com.google.devtools.build.lib.analysis.config.CoreOptions;
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
              "no_config_rule",
              (builder, env) -> builder.cfg(unused -> NoConfigTransition.INSTANCE));

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder().addRuleDefinition(NO_CONFIG_RULE);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void preventsConfiguredTargetForking() throws Exception {
    // Write a custom Starlark rule that arbitrarily transitions its configuration. Have two
    // instances of that rule transition to different configurations and each depend on the same
    // no_config_rule. We expect there to be only one instance of the no_config_rule.
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = [
                "//...",
            ],
        )
        """);
    scratch.file(
        "foo/defs.bzl",
        """
        # Define the flag to transition on:
        FlagInfo = provider(fields = {"value": "The value."})
        custom_flag = rule(
            implementation = lambda ctx: FlagInfo(value = ctx.build_setting_value),
            build_setting = config.string(flag = True),
        )

        # Define the transitioning rule:
        my_transition = transition(
            implementation = lambda settings, attr: {"//foo:my_flag": attr.flag_value},
            inputs = [],
            outputs = ["//foo:my_flag"],
        )

        transition_rule = rule(
            implementation = lambda ctx: [],
            cfg = my_transition,
            attrs = {
                "flag_value": attr.string(),
                "dep": attr.label(),
            },
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":defs.bzl", "custom_flag", "transition_rule")

        custom_flag(
            name = "my_flag",
            build_setting_default = "default flag value",
        )

        no_config_rule(name = "config_free_target")

        transition_rule(
            name = "parent1",
            dep = ":config_free_target",
            flag_value = "parent1 setting",
        )

        transition_rule(
            name = "parent2",
            dep = ":config_free_target",
            flag_value = "parent2 different setting",
        )
        """);

    ConfiguredTarget parent1 = getConfiguredTarget("//foo:parent1");
    ConfiguredTarget parent2 = getConfiguredTarget("//foo:parent2");

    assertThat(parent1.getConfigurationKey()).isNotEqualTo(parent2.getConfigurationKey());
    assertThat(getDirectPrerequisite(parent1, "//foo:config_free_target"))
        .isSameInstanceAs(getDirectPrerequisite(parent2, "//foo:config_free_target"));
  }

  @Test
  public void preventsConfiguredTargetForkingOnCoreOptions() throws Exception {
    // Ideally NoConfigTransition would contain empty BuildOptions. In practice it keeps CoreOptions
    // because core Blaze logic reads CoreOptions (see NoConfigTransition for details). Crucially,
    // it contains CoreOptions default values - not the values inherited from the source config.
    // So we still expect config forking to be impossible.
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = [
                "//...",
            ],
        )
        """);
    scratch.file(
        "foo/defs.bzl",
        """
        def _my_transition_impl(settings, attr):
            return {"//command_line_option:features": [attr.feature]}

        my_transition = transition(
            implementation = _my_transition_impl,
            inputs = [],
            outputs = ["//command_line_option:features"],
        )

        transition_features_rule = rule(
            implementation = lambda ctx: [],
            cfg = my_transition,
            attrs = {
                "feature": attr.string(),
                "dep": attr.label(),
            },
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":defs.bzl", "transition_features_rule")

        no_config_rule(name = "config_free_target")

        transition_features_rule(
            name = "parent1",
            dep = ":config_free_target",
            feature = "one",
        )

        transition_features_rule(
            name = "parent2",
            dep = ":config_free_target",
            feature = "two",
        )
        """);

    ConfiguredTarget parent1 = getConfiguredTarget("//foo:parent1");
    ConfiguredTarget parent2 = getConfiguredTarget("//foo:parent2");

    // Sanity check: ensure the flag we're changing is actually in CoreOptions. If you're moving the
    // flag out of CoreOptions, replace it with another CoreOptions flag.
    assertThat(
            parent1
                .getConfigurationKey()
                .getOptions()
                .get(CoreOptions.class)
                .getClass()
                .getField("defaultFeatures"))
        .isNotNull();
    assertThat(parent1.getConfigurationKey()).isNotEqualTo(parent2.getConfigurationKey());
    assertThat(getDirectPrerequisite(parent1, "//foo:config_free_target"))
        .isSameInstanceAs(getDirectPrerequisite(parent2, "//foo:config_free_target"));
  }
}
