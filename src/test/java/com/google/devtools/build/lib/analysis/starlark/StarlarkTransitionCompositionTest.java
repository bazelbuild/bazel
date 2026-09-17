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
