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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code transition.and_then}, which composes configuration transitions in Starlark. */
@RunWith(JUnit4.class)
public final class StarlarkTransitionCompositionTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  /**
   * Writes Starlark transitions on the {@code --foo} and {@code --bar} native options: {@code
   * set_foo} and {@code set_bar} set them to fixed values, while {@code foo_to_bar} reads {@code
   * --foo} and writes {@code --bar} based on it (so applying it after {@code set_foo} proves the
   * second transition sees the first one's output).
   */
  private void writeTransitions() throws Exception {
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

        def _set_bar_impl(settings, attr):
            return {"//command_line_option:bar": "bar_set"}

        set_bar = transition(
            implementation = _set_bar_impl,
            inputs = [],
            outputs = ["//command_line_option:bar"],
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

        def _split_foo_impl(settings, attr):
            return {
                "a": {"//command_line_option:foo": "a"},
                "b": {"//command_line_option:foo": "b"},
            }

        split_foo = transition(
            implementation = _split_foo_impl,
            inputs = [],
            outputs = ["//command_line_option:foo"],
        )

        def _split_bar_impl(settings, attr):
            return {
                "x": {"//command_line_option:bar": "x"},
                "y": {"//command_line_option:bar": "y"},
            }

        split_bar = transition(
            implementation = _split_bar_impl,
            inputs = [],
            outputs = ["//command_line_option:bar"],
        )
        """);
  }

  private DummyTestOptions optionsOf(ConfiguredTarget target) {
    return getConfiguration(target).getOptions().get(DummyTestOptions.class);
  }

  @Test
  public void ruleTransition_starlarkThenStarlark() throws Exception {
    writeTransitions();
    scratch.file(
        "test/rules.bzl",
        """
        load("//test:transitions.bzl", "foo_to_bar", "set_foo")

        my_rule = rule(
            implementation = lambda ctx: [],
            cfg = set_foo.and_then(foo_to_bar),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule")

        my_rule(name = "test")
        """);

    ConfiguredTarget test = getConfiguredTarget("//test:test");

    assertThat(optionsOf(test).getFoo()).isEqualTo("foo_set");
    assertThat(optionsOf(test).getBar()).isEqualTo("foo_set_to_bar");
  }

  @Test
  public void attributeTransition_starlarkThenStarlark() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(foo_to_bar)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(optionsOf(dep).getFoo()).isEqualTo("foo_set");
    assertThat(optionsOf(dep).getBar()).isEqualTo("foo_set_to_bar");
    assertThat(getConfiguration(dep).isExecConfiguration()).isFalse();
  }

  @Test
  public void attributeTransition_starlarkThenExec() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(config.exec())");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
    // set_foo runs before the exec transition, which resets --foo back to its default.
    assertThat(optionsOf(dep).getFoo()).isEmpty();
  }

  @Test
  public void attributeTransition_execThenStarlark() throws Exception {
    writeRuleWithAttributeTransition("config.exec().and_then(set_foo)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    // set_foo runs after the exec transition, so its effect is visible.
    assertThat(optionsOf(dep).getFoo()).isEqualTo("foo_set");
    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
  }

  @Test
  public void attributeTransition_starlarkThenExecThenStarlark() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(config.exec()).and_then(set_bar)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(optionsOf(dep).getBar()).isEqualTo("bar_set");
    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
    // set_foo runs before the exec transition, which resets --foo back to its default.
    assertThat(optionsOf(dep).getFoo()).isEmpty();
  }

  @Test
  public void attributeTransition_starlarkThenStarlarkThenExec() throws Exception {
    writeRuleWithAttributeTransition("set_foo.and_then(set_bar).and_then(config.exec())");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
    // Both Starlark transitions run before the exec transition, which resets --foo and --bar.
    assertThat(optionsOf(dep).getFoo()).isEmpty();
    assertThat(optionsOf(dep).getBar()).isEmpty();
  }

  @Test
  public void attributeTransition_execThenStarlarkThenStarlark() throws Exception {
    writeRuleWithAttributeTransition("config.exec().and_then(set_foo).and_then(foo_to_bar)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    // Both Starlark transitions run after the exec transition, and foo_to_bar reads set_foo's
    // output.
    assertThat(optionsOf(dep).getFoo()).isEqualTo("foo_set");
    assertThat(optionsOf(dep).getBar()).isEqualTo("foo_set_to_bar");
    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
  }

  @Test
  public void attributeTransition_execThenStarlark_readsUnsetStarlarkFlag() throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition(
        "default", "config.exec().and_then(flag_to_foo)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(optionsOf(dep).getFoo()).isEqualTo("flag_default");
    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
  }

  @Test
  public void attributeTransition_execThenStarlark_readsStarlarkFlagResetByExec() throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition(
        "target", "config.exec().and_then(flag_to_foo)");
    useConfiguration("--//test:flag=custom");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    // The exec transition resets the target-scoped flag to its default.
    assertThat(optionsOf(dep).getFoo()).isEqualTo("flag_default");
    assertThat(getConfiguration(dep).getOptions().getStarlarkOptions()).isEmpty();
  }

  @Test
  public void attributeTransition_execThenStarlark_readsStarlarkFlagPropagatedByExec()
      throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition(
        "universal", "config.exec().and_then(flag_to_foo)");
    useConfiguration("--//test:flag=custom");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(optionsOf(dep).getFoo()).isEqualTo("custom");
    assertThat(getConfiguration(dep).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//test:flag"), "custom");
  }

  @Test
  public void attributeTransition_starlarkThenExec_resetsStarlarkFlag() throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition("target", "set_flag.and_then(config.exec())");
    useConfiguration("--//test:flag=custom");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    // set_flag runs before the exec transition, which resets the target-scoped flag.
    assertThat(getConfiguration(dep).getOptions().getStarlarkOptions()).isEmpty();
    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
  }

  @Test
  public void attributeTransition_starlarkThenExec_customExecScopeReadsResetHostFlag()
      throws Exception {
    scratch.file(
        "test/rules.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {"scope": attr.string()},
        )

        def _reset_host_flag_impl(settings, attr):
            return {"//test:host_flag": "host_default"}

        reset_host_flag = transition(
            implementation = _reset_host_flag_impl,
            inputs = [],
            outputs = ["//test:host_flag"],
        )

        my_rule = rule(
            implementation = lambda ctx: [],
            attrs = {"dep": attr.label(cfg = reset_host_flag.and_then(config.exec()))},
        )

        simple = rule(implementation = lambda ctx: [])
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule", "simple", "string_flag")

        string_flag(
            name = "flag",
            build_setting_default = "flag_default",
            scope = "exec:--//test:host_flag",
        )

        string_flag(
            name = "host_flag",
            build_setting_default = "host_default",
            scope = "target",
        )

        my_rule(name = "test", dep = ":dep")

        simple(name = "dep")
        """);
    useConfiguration(
        "--flag_alias=host_flag=//test:host_flag",
        "--//test:flag=custom",
        "--//test:host_flag=custom_host");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    // The exec transition sets the flag to the value of the host flag, which reset_host_flag reset.
    assertThat(getConfiguration(dep).getOptions().getStarlarkOptions())
        .containsEntry(Label.parseCanonicalUnchecked("//test:flag"), "host_default");
  }

  @Test
  public void attributeTransition_starlarkThenStarlark_readsStarlarkFlagSetViaAlias()
      throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition(
        "default", "set_flag_via_alias.and_then(alias_to_foo)");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(optionsOf(dep).getFoo()).isEqualTo("flag_set");
    assertThat(getConfiguration(dep).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//test:flag"), "flag_set");
  }

  @Test
  public void attributeTransition_starlarkThenStarlark_validatesBothTransitions() throws Exception {
    writeRuleWithStarlarkFlagAndAttributeTransition(
        "default", "set_flag_to_int.and_then(set_flag)");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:test");

    // set_flag overwrites the invalid value set by set_flag_to_int.
    assertContainsEvent("expected value of type 'string' for //test:flag, but got 1 (int)");
  }

  @Test
  public void attributeTransition_splitComposedWithSplit() throws Exception {
    writeTransitions();
    scratch.file(
        "test/rules.bzl",
        """
        load("//test:transitions.bzl", "split_bar", "split_foo")

        my_rule = rule(
            implementation = lambda ctx: [],
            attrs = {"deps": attr.label_list(cfg = split_foo.and_then(split_bar))},
        )

        simple = rule(implementation = lambda ctx: [])
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule", "simple")

        my_rule(name = "test", deps = [":dep"])

        simple(name = "dep")
        """);

    ConfiguredTarget test = getConfiguredTarget("//test:test");

    ImmutableSet<String> combos =
        getDirectPrerequisites(test).stream()
            .filter(ct -> ct.getLabel().toString().equals("//test:dep"))
            .map(ct -> optionsOf(ct).getFoo() + "," + optionsOf(ct).getBar())
            .collect(toImmutableSet());

    assertThat(combos).containsExactly("a,x", "a,y", "b,x", "b,y");
  }

  @Test
  public void noOpTransitionComposition() throws Exception {
    // config.target() is a no-op and may be freely composed even alongside the exec transition.
    writeRuleWithAttributeTransition("config.target().and_then(set_foo).and_then(config.exec())");

    ConfiguredTarget dep = getDirectPrerequisite(getConfiguredTarget("//test:test"), "//test:dep");

    assertThat(getConfiguration(dep).isExecConfiguration()).isTrue();
    // set_foo runs before the exec transition, which resets --foo back to its default.
    assertThat(optionsOf(dep).getFoo()).isEmpty();
  }

  @Test
  public void execComposedWithExec_fails() throws Exception {
    writeTransitions();
    scratch.file(
        "test/rules.bzl",
        """
        bad_transition = config.exec().and_then(config.exec())

        my_rule = rule(
            implementation = lambda ctx: [],
            attrs = {"dep": attr.label(cfg = bad_transition)},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule")

        my_rule(name = "test", dep = ":dep")

        filegroup(name = "dep")
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:test");

    assertContainsEvent("can't compose two exec transitions");
    // The error reports both the use site (the `cfg=` attribute) and the original `.and_then`
    // call site, so users can find the offending composition even when it is defined in a
    // different .bzl file.
    assertContainsEvent("attrs = {\"dep\": attr.label(cfg = bad_transition)},");
    assertContainsEvent("composed at /workspace/test/rules.bzl:1:");
  }

  @Test
  public void execInRuleTransition_fails() throws Exception {
    writeTransitions();
    scratch.file(
        "test/rules.bzl",
        """
        load("//test:transitions.bzl", "set_foo")

        my_rule = rule(
            implementation = lambda ctx: [],
            cfg = set_foo.and_then(config.exec()),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule")

        my_rule(name = "test")
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:test");

    assertContainsEvent("can only be used as an attribute transition");
  }

  /**
   * Writes a {@code //test:flag} string flag with the given scope, an alias {@code //test:alias}
   * for it and a rule whose {@code dep} attribute uses {@code composedTransition}, which can refer
   * to {@code set_flag} (sets the flag), {@code set_flag_via_alias} (sets the flag through the
   * alias), {@code set_flag_to_int} (sets the flag to an invalid value), {@code flag_to_foo}
   * (copies the flag to {@code --foo}) and {@code alias_to_foo} (copies the flag to {@code --foo}
   * through the alias).
   */
  private void writeRuleWithStarlarkFlagAndAttributeTransition(
      String flagScope, String composedTransition) throws Exception {
    scratch.file(
        "test/rules.bzl",
        String.format(
            """
            string_flag = rule(
                implementation = lambda ctx: [],
                build_setting = config.string(flag = True),
                attrs = {"scope": attr.string()},
            )

            def _set_flag_impl(settings, attr):
                return {"//test:flag": "flag_set"}

            set_flag = transition(
                implementation = _set_flag_impl,
                inputs = [],
                outputs = ["//test:flag"],
            )

            def _set_flag_via_alias_impl(settings, attr):
                return {"//test:alias": "flag_set"}

            set_flag_via_alias = transition(
                implementation = _set_flag_via_alias_impl,
                inputs = [],
                outputs = ["//test:alias"],
            )

            def _set_flag_to_int_impl(settings, attr):
                return {"//test:flag": 1}

            set_flag_to_int = transition(
                implementation = _set_flag_to_int_impl,
                inputs = [],
                outputs = ["//test:flag"],
            )

            def _flag_to_foo_impl(settings, attr):
                return {"//command_line_option:foo": settings["//test:flag"]}

            flag_to_foo = transition(
                implementation = _flag_to_foo_impl,
                inputs = ["//test:flag"],
                outputs = ["//command_line_option:foo"],
            )

            def _alias_to_foo_impl(settings, attr):
                return {"//command_line_option:foo": settings["//test:alias"]}

            alias_to_foo = transition(
                implementation = _alias_to_foo_impl,
                inputs = ["//test:alias"],
                outputs = ["//command_line_option:foo"],
            )

            my_rule = rule(
                implementation = lambda ctx: [],
                attrs = {"dep": attr.label(cfg = %s)},
            )

            simple = rule(implementation = lambda ctx: [])
            """,
            composedTransition));
    scratch.file(
        "test/BUILD",
        String.format(
            """
            load("//test:rules.bzl", "my_rule", "simple", "string_flag")

            string_flag(
                name = "flag",
                build_setting_default = "flag_default",
                scope = "%s",
            )

            alias(name = "alias", actual = ":flag")

            my_rule(name = "test", dep = ":dep")

            simple(name = "dep")
            """,
            flagScope));
  }

  private void writeRuleWithAttributeTransition(String composedTransition) throws Exception {
    writeTransitions();
    scratch.file(
        "test/rules.bzl",
        String.format(
            """
            load("//test:transitions.bzl", "foo_to_bar", "set_bar", "set_foo")

            my_rule = rule(
                implementation = lambda ctx: [],
                attrs = {"dep": attr.label(cfg = %s)},
            )

            simple = rule(implementation = lambda ctx: [])
            """,
            composedTransition));
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule", "simple")

        my_rule(name = "test", dep = ":dep")

        simple(name = "dep")
        """);
  }
}
