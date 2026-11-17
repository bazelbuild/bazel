// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the flags of target platforms that transitions change to. */
@RunWith(JUnit4.class)
public final class StarlarkTransitionPlatformFlagsTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  @Before
  public void writePlatformsAndTransitions() throws Exception {
    writePlatforms("p2");
    scratch.file(
        "test/transitions.bzl",
        """
def _set_foo_impl(settings, attr):
    return {"//command_line_option:foo": "foo_set"}

set_foo = transition(
    implementation = _set_foo_impl,
    inputs = [],
    outputs = ["//command_line_option:foo"],
)

def _foo_to_bar_impl(settings, attr):
    return {
        "//command_line_option:bar": settings["//command_line_option:foo"] + "_to_bar",
    }

foo_to_bar = transition(
    implementation = _foo_to_bar_impl,
    inputs = ["//command_line_option:foo"],
    outputs = ["//command_line_option:bar"],
)

def _platform_transition(platform, foo = None):
    def _impl(settings, attr):
        result = {"//command_line_option:platforms": [platform]}
        if foo:
            result["//command_line_option:foo"] = foo
        return result

    outputs = ["//command_line_option:platforms"]
    if foo:
        outputs.append("//command_line_option:foo")
    return transition(implementation = _impl, inputs = [], outputs = outputs)

to_p1 = _platform_transition("//platforms:p1")
to_p2 = _platform_transition("//platforms:p2")
to_no_flags = _platform_transition("//platforms:no_flags")
to_p2_and_set_foo = _platform_transition("//platforms:p2", foo = "foo_set")
to_starlark_flags = _platform_transition("//platforms:starlark_flags")
to_flag_alias = _platform_transition("//platforms:flag_alias")

def _split_starlark_flags_and_no_flags_impl(settings, attr):
    return {
        "starlark_flags": {"//command_line_option:platforms": ["//platforms:starlark_flags"]},
        "no_flags": {"//command_line_option:platforms": ["//platforms:no_flags"]},
    }

split_starlark_flags_and_no_flags = transition(
    implementation = _split_starlark_flags_and_no_flags_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _read_alias_and_to_starlark_flags_impl(settings, attr):
    return {"//command_line_option:platforms": ["//platforms:starlark_flags"]}

read_alias_and_to_starlark_flags = transition(
    implementation = _read_alias_and_to_starlark_flags_impl,
    inputs = ["//flags:universal_alias"],
    outputs = ["//command_line_option:platforms"],
)

def _print_and_to_bad_scope_impl(settings, attr):
    print("applying print_and_to_bad_scope")
    return {"//command_line_option:platforms": ["//platforms:bad_scope"]}

print_and_to_bad_scope = transition(
    implementation = _print_and_to_bad_scope_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _split_platforms_impl(settings, attr):
    return {
        "p2": {"//command_line_option:platforms": ["//platforms:p2"]},
        "p3": {"//command_line_option:platforms": ["//platforms:p3"]},
        "same": {
            "//command_line_option:platforms": settings["//command_line_option:platforms"],
        },
    }

split_platforms = transition(
    implementation = _split_platforms_impl,
    inputs = ["//command_line_option:platforms"],
    outputs = ["//command_line_option:platforms"],
)

def _print_foo_impl(settings, attr):
    print("foo=" + settings["//command_line_option:foo"])
    return {}

print_foo = transition(
    implementation = _print_foo_impl,
    inputs = ["//command_line_option:foo"],
    outputs = [],
)

def _print_and_to_missing_impl(settings, attr):
    print("applying print_and_to_missing")
    return {"//command_line_option:platforms": ["//platforms:does_not_exist"]}

print_and_to_missing = transition(
    implementation = _print_and_to_missing_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)
""");
    scratch.file(
        "flags/defs.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {"scope": attr.string()},
        )
        """);
    scratch.file(
        "flags/BUILD",
        """
        load(":defs.bzl", "string_flag")

        string_flag(
            name = "universal",
            build_setting_default = "default",
            scope = "universal",
            visibility = ["//visibility:public"],
        )

        string_flag(
            name = "target",
            build_setting_default = "default",
            scope = "target",
            visibility = ["//visibility:public"],
        )

        string_flag(
            name = "custom_exec",
            build_setting_default = "default",
            scope = "exec:--//flags:host",
            visibility = ["//visibility:public"],
        )

        string_flag(
            name = "host",
            build_setting_default = "from_host",
            visibility = ["//visibility:public"],
        )

        string_flag(
            name = "bad_scope",
            build_setting_default = "default",
            scope = "exec:--//flags:does_not_exist",
            visibility = ["//visibility:public"],
        )

        alias(
            name = "universal_alias",
            actual = ":universal",
            visibility = ["//visibility:public"],
        )
        """);
    useConfiguration("--platforms=//platforms:p1");
  }

  private void writePlatforms(String p2Foo) throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        String.format(
            """
            platform(
                name = "p1",
                flags = ["--foo=p1"],
            )

            platform(
                name = "p2",
                flags = ["--foo=%s"],
            )

            platform(
                name = "p3",
                flags = ["--foo=p3"],
            )

            platform(name = "no_flags")

            platform(
                name = "starlark_flags",
                flags = [
                    "--//flags:universal=from_platform",
                    "--//flags:target=from_platform",
                    "--//flags:custom_exec=from_platform",
                ],
            )

            platform(
                name = "flag_alias",
                flags = ["--flag_alias=universal=//flags:universal"],
            )

            platform(
                name = "bad_scope",
                flags = ["--//flags:bad_scope=from_platform"],
            )
            """,
            p2Foo));
  }

  private void writeRuleWithAttributeTransition(String transition) throws Exception {
    scratch.overwriteFile(
        "test/rules.bzl",
        String.format(
            """
            load(
                "//test:transitions.bzl",
                "foo_to_bar",
                "print_and_to_bad_scope",
                "print_and_to_missing",
                "print_foo",
                "read_alias_and_to_starlark_flags",
                "set_foo",
                "split_platforms",
                "split_starlark_flags_and_no_flags",
                "to_flag_alias",
                "to_no_flags",
                "to_p1",
                "to_p2",
                "to_p2_and_set_foo",
                "to_starlark_flags",
            )

            my_rule = rule(
                implementation = lambda ctx: [],
                attrs = {"dep": attr.label(cfg = %s)},
            )

            simple = rule(implementation = lambda ctx: [])
            """,
            transition));
    scratch.overwriteFile(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule", "simple")

        my_rule(name = "test", dep = ":dep")

        simple(name = "dep")
        """);
  }

  private BuildConfigurationValue depConfiguration() throws Exception {
    return getConfiguration(
        getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep"));
  }

  private DummyTestOptions depOptions() throws Exception {
    return depConfiguration().getOptions().get(DummyTestOptions.class);
  }

  @Test
  public void topLevel_getsPlatformFlags() throws Exception {
    writeRuleWithAttributeTransition("\"target\"");

    assertThat(
            getConfiguration(getConfiguredTarget("//test:test"))
                .getOptions()
                .get(DummyTestOptions.class)
                .getFoo())
        .isEqualTo("p1");
  }

  @Test
  public void transitionWithoutPlatformChange_overridesPlatformFlag() throws Exception {
    writeRuleWithAttributeTransition("set_foo");

    assertThat(depOptions().getFoo()).isEqualTo("foo_set");
  }

  @Test
  public void ruleTransitionWithoutPlatformChange_overridesPlatformFlag() throws Exception {
    scratch.file(
        "test/rules.bzl",
        """
        load("//test:transitions.bzl", "set_foo")

        my_rule = rule(
            implementation = lambda ctx: [],
            cfg = set_foo,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule")

        my_rule(name = "test")
        """);

    assertThat(
            getConfiguration(getConfiguredTarget("//test:test"))
                .getOptions()
                .get(DummyTestOptions.class)
                .getFoo())
        .isEqualTo("foo_set");
  }

  @Test
  public void transitionChangingPlatform_getsPlatformFlags() throws Exception {
    writeRuleWithAttributeTransition("to_p2");

    assertThat(depOptions().getFoo()).isEqualTo("p2");
  }

  @Test
  public void transitionChangingPlatformAndFlag_getsPlatformFlags() throws Exception {
    writeRuleWithAttributeTransition("to_p2_and_set_foo");

    assertThat(depOptions().getFoo()).isEqualTo("p2");
  }

  @Test
  public void transitionChangingToPlatformWithoutFlags_keepsFlags() throws Exception {
    writeRuleWithAttributeTransition("to_no_flags");

    assertThat(depOptions().getFoo()).isEqualTo("p1");
  }

  @Test
  public void composition_platformChangeThenFlag() throws Exception {
    writeRuleWithAttributeTransition("to_p2.and_then(set_foo)");

    assertThat(depOptions().getFoo()).isEqualTo("foo_set");
  }

  @Test
  public void composition_platformChangeThenReadFlag() throws Exception {
    writeRuleWithAttributeTransition("to_p2.and_then(foo_to_bar)");

    assertThat(depOptions().getFoo()).isEqualTo("p2");
    assertThat(depOptions().getBar()).isEqualTo("p2_to_bar");
  }

  @Test
  public void composition_flagThenPlatformChange() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(to_p2)");

    assertThat(depOptions().getFoo()).isEqualTo("p2");
  }

  @Test
  public void composition_toOtherPlatformAndBack_reappliesPlatformFlags() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(to_no_flags).and_then(to_p1)");

    assertThat(depOptions().getFoo()).isEqualTo("p1");
  }

  @Test
  public void composition_splitPlatformChangeThenReadFlag() throws Exception {
    writeRuleWithAttributeTransition("split_platforms.and_then(foo_to_bar)");

    ImmutableSet<String> bars =
        getDirectPrerequisites(getConfiguredTarget("//test:test")).stream()
            .filter(ct -> ct.getLabel().toString().equals("//test:dep"))
            .map(ct -> getConfiguration(ct).getOptions().get(DummyTestOptions.class).getBar())
            .collect(toImmutableSet());

    assertThat(bars).containsExactly("p1_to_bar", "p2_to_bar", "p3_to_bar");
  }

  @Test
  public void composition_intermediatePlatformFlagsChange() throws Exception {
    writeRuleWithAttributeTransition("to_p2.and_then(foo_to_bar).and_then(to_p1)");
    assertThat(depOptions().getBar()).isEqualTo("p2_to_bar");

    writePlatforms("p2_changed");
    // Keeps the configurations and thus the transition cache.
    invalidatePackages(/* alsoConfigs= */ false);

    assertThat(depOptions().getBar()).isEqualTo("p2_changed_to_bar");
  }

  @Test
  public void composition_platformChange_eventsReportedOnce() throws Exception {
    writeRuleWithAttributeTransition("print_foo.and_then(to_p2).and_then(print_foo)");

    getConfiguredTarget("//test:test");

    assertContainsEventWithFrequency("foo=p1", 1);
    assertContainsEventWithFrequency("foo=p2", 1);
  }

  @Test
  public void composition_invalidIntermediatePlatform_reportsEvents() throws Exception {
    writeRuleWithAttributeTransition("print_and_to_missing.and_then(set_foo)");
    reporter.removeHandler(failFastHandler);

    assertThat(getConfiguredTarget("//test:test")).isNull();

    assertContainsEvent("applying print_and_to_missing");
    assertContainsEvent("//platforms:does_not_exist");
  }

  private static final Label UNIVERSAL_FLAG = Label.parseCanonicalUnchecked("//flags:universal");
  private static final Label CUSTOM_EXEC_FLAG =
      Label.parseCanonicalUnchecked("//flags:custom_exec");

  @Test
  public void composition_platformChangeThenExec_propagatesStarlarkFlagsByScope() throws Exception {
    writeRuleWithAttributeTransition("to_starlark_flags.and_then(config.exec())");

    assertThat(depConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(UNIVERSAL_FLAG, "from_platform", CUSTOM_EXEC_FLAG, "from_host");
  }

  @Test
  public void composition_platformChangeThenExec_doesNotPropagateTargetScopedStarlarkFlag()
      throws Exception {
    writeRuleWithAttributeTransition("to_starlark_flags.and_then(config.exec())");
    useConfiguration(
        "--platforms=//platforms:p1", "--noincompatible_exclude_starlark_flags_from_exec_config");

    assertThat(depConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(UNIVERSAL_FLAG, "from_platform");
  }

  @Test
  public void composition_splitPlatformChangeThenExec_scopesOnlyApplyToOwnBranch()
      throws Exception {
    writeRuleWithAttributeTransition("split_starlark_flags_and_no_flags.and_then(config.exec())");

    ImmutableSet<Map<Label, Object>> starlarkOptions =
        getDirectPrerequisites(getConfiguredTarget("//test:test")).stream()
            .filter(ct -> ct.getLabel().toString().equals("//test:dep"))
            .map(ct -> getConfiguration(ct).getOptions().getStarlarkOptions())
            .collect(toImmutableSet());

    assertThat(starlarkOptions)
        .containsExactly(
            ImmutableMap.of(UNIVERSAL_FLAG, "from_platform", CUSTOM_EXEC_FLAG, "from_host"),
            ImmutableMap.of());
  }

  @Test
  public void composition_readAliasThenPlatformSetsActualThenExec() throws Exception {
    writeRuleWithAttributeTransition("read_alias_and_to_starlark_flags.and_then(config.exec())");

    assertThat(depConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(UNIVERSAL_FLAG, "from_platform", CUSTOM_EXEC_FLAG, "from_host");
  }

  @Test
  public void composition_intermediatePlatformSetsFlagAlias_fails() throws Exception {
    writeRuleWithAttributeTransition("to_flag_alias.and_then(set_foo)");
    reporter.removeHandler(failFastHandler);

    assertThat(getConfiguredTarget("//test:test")).isNull();

    assertContainsEvent("//platforms:flag_alias can't set --flag_alias");
  }

  @Test
  public void composition_invalidIntermediatePlatformFlagScope_reportsEvents() throws Exception {
    writeRuleWithAttributeTransition("print_and_to_bad_scope.and_then(config.exec())");
    reporter.removeHandler(failFastHandler);

    assertThat(getConfiguredTarget("//test:test")).isNull();

    assertContainsEvent("applying print_and_to_bad_scope");
    assertContainsEvent("//flags:does_not_exist");
  }

  @Test
  public void execTransitionToSamePlatform_getsPlatformFlags() throws Exception {
    writeRuleWithAttributeTransition("config.exec()");
    useConfiguration("--platforms=//platforms:p1", "--extra_execution_platforms=//platforms:p1");

    assertThat(depOptions().getFoo()).isEqualTo("p1");
  }

  @Test
  public void composition_execToSamePlatformThenReadFlag() throws Exception {
    writeRuleWithAttributeTransition("config.exec().and_then(foo_to_bar)");
    useConfiguration("--platforms=//platforms:p1", "--extra_execution_platforms=//platforms:p1");

    assertThat(depOptions().getBar()).isEqualTo("p1_to_bar");
  }
}
