// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.ToolchainCollection.DEFAULT_EXEC_GROUP_NAME;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for exec groups. Functionality related to rule context tested in {@link
 * com.google.devtools.build.lib.starlark.StarlarkRuleContextTest}.
 */
@RunWith(JUnit4.class)
public class StarlarkExecGroupTest extends BuildViewTestCase {

  @Before
  public final void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_exec_groups");
  }

  /**
   * Sets up two toolchains types, each with a single toolchain implementation and a single
   * exec_compatible_with platform.
   *
   * <p>toolchain_type_1 -> foo_toolchain -> exec_compatible_with platform_1 toolchain_type_2 ->
   * bar_toolchain -> exec_compatible_with platform_2
   */
  private void createToolchainsAndPlatforms() throws Exception {
    scratch.file(
        "rule/test_toolchain.bzl",
        "def _impl(ctx):",
        "    return [platform_common.ToolchainInfo()]",
        "test_toolchain = rule(",
        "    implementation = _impl,",
        ")");
    scratch.file(
        "rule/BUILD",
        "exports_files(['test_toolchain/bzl'])",
        "toolchain_type(name = 'toolchain_type_1')",
        "toolchain_type(name = 'toolchain_type_2')");
    scratch.file(
        "toolchain/BUILD",
        "load('//rule:test_toolchain.bzl', 'test_toolchain')",
        "test_toolchain(",
        "    name = 'foo',",
        ")",
        "toolchain(",
        "    name = 'foo_toolchain',",
        "    toolchain_type = '//rule:toolchain_type_1',",
        "    target_compatible_with = ['//platform:constraint_1'],",
        "    exec_compatible_with = ['//platform:constraint_1'],",
        "    toolchain = ':foo',",
        ")",
        "test_toolchain(",
        "    name = 'bar',",
        ")",
        "toolchain(",
        "    name = 'bar_toolchain',",
        "    toolchain_type = '//rule:toolchain_type_2',",
        "    target_compatible_with = ['//platform:constraint_1'],",
        "    exec_compatible_with = ['//platform:constraint_2'],",
        "    toolchain = ':bar',",
        ")");

    scratch.file(
        "platform/BUILD",
        "constraint_setting(name = 'setting')",
        "constraint_value(",
        "    name = 'constraint_1',",
        "    constraint_setting = ':setting',",
        ")",
        "constraint_value(",
        "    name = 'constraint_2',",
        "    constraint_setting = ':setting',",
        ")",
        "platform(",
        "    name = 'platform_1',",
        "    constraint_values = [':constraint_1'],",
        ")",
        "platform(",
        "    name = 'platform_2',",
        "    constraint_values = [':constraint_2'],",
        ")");

    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:bar_toolchain",
        "--platforms=//platform:platform_1",
        "--extra_execution_platforms=//platform:platform_1,//platform:platform_2");
  }

  @Test
  public void testExecGroupTransition() throws Exception {
    createToolchainsAndPlatforms();

    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  return [MyInfo(dep = ctx.attr.dep, exec_group_dep = ctx.attr.exec_group_dep)]",
        "with_transition = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'exec_group_dep': attr.label(cfg = config.exec('watermelon')),",
        "    'dep': attr.label(cfg = 'exec'),",
        "  },",
        "  exec_groups = {",
        "    'watermelon': exec_group(toolchains = ['//rule:toolchain_type_2']),",
        "  },",
        "  toolchains = ['//rule:toolchain_type_1'],",
        ")",
        "def _impl2(ctx):",
        "  return []",
        "simple_rule = rule(implementation = _impl2)");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_transition', 'simple_rule')",
        "with_transition(name = 'parent', dep = ':child', exec_group_dep = ':other-child')",
        "simple_rule(name = 'child')",
        "simple_rule(name = 'other-child')");

    ConfiguredTarget target = getConfiguredTarget("//test:parent");
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:defs.bzl", ImmutableMap.of()), "MyInfo");
    BuildConfiguration dep =
        getConfiguration((ConfiguredTarget) ((StructImpl) target.get(key)).getValue("dep"));
    BuildConfiguration execGroupDep =
        getConfiguration(
            (ConfiguredTarget) ((StructImpl) target.get(key)).getValue("exec_group_dep"));

    assertThat(dep.getFragment(PlatformConfiguration.class).getTargetPlatform())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platform:platform_1"));
    assertThat(execGroupDep.getFragment(PlatformConfiguration.class).getTargetPlatform())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platform:platform_2"));
  }

  @Test
  public void testInvalidExecGroupTransition() throws Exception {
    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  return []",
        "with_transition = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'exec_group_dep': attr.label(cfg = config.exec('blueberry')),",
        "  },",
        ")",
        "def _impl2(ctx):",
        "  return []",
        "simple_rule = rule(implementation = _impl2)");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_transition', 'simple_rule')",
        "with_transition(name = 'parent', exec_group_dep = ':child')",
        "simple_rule(name = 'child')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:parent");
    assertContainsEvent(
        "Attr 'exec_group_dep' declares a transition for non-existent exec group 'blueberry'");
  }

  @Test
  public void testExecGroupActionHasExecGroupPlatform() throws Exception {
    createToolchainsAndPlatforms();
    writeRuleWithActionsAndWatermelonExecGroup();

    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_actions')",
        "with_actions(",
        "  name = 'papaya',",
        "  output = 'out.txt',",
        "  watermelon_output = 'watermelon_out.txt'",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//test:papaya");

    assertThat(
            getGeneratingAction(target, "test/watermelon_out.txt")
                .getOwner()
                .getExecutionPlatform()
                .label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platform:platform_2"));
    assertThat(
            getGeneratingAction(target, "test/out.txt").getOwner().getExecutionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platform:platform_1"));
  }

  @Test
  public void testActionDeclaresInvalidExecGroup() throws Exception {
    createToolchainsAndPlatforms();

    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  watermelon_out_file = ctx.outputs.watermelon_output",
        "  ctx.actions.run_shell(",
        "    inputs = [],",
        "    outputs = [watermelon_out_file],",
        "    arguments = [watermelon_out_file.path],",
        "    command = 'echo hello > \"$1\"',",
        "    exec_group = 'honeydew',",
        "  )",
        "with_actions = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'watermelon_output': attr.output(),",
        "  },",
        "  exec_groups = {",
        "    'watermelon': exec_group(toolchains = ['//rule:toolchain_type_2']),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_actions')",
        "with_actions(",
        "  name = 'papaya',",
        "  watermelon_output = 'watermelon_out.txt'",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:papaya");
    assertContainsEvent("Action declared for non-existent exec group 'honeydew'");
  }

  @Test
  public void testCannotNameExecGroupDefaultName() throws Exception {
    createToolchainsAndPlatforms();

    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  exec_groups = {",
        "    '"
            + DEFAULT_EXEC_GROUP_NAME
            + "': exec_group(toolchains = ['//rule:toolchain_type_2']),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD", //
        "load('//test:defs.bzl', 'my_rule')",
        "my_rule(name = 'papaya')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:papaya");
    assertContainsEvent("Exec group name '" + DEFAULT_EXEC_GROUP_NAME + "' is not a valid name");
  }

  private void writeRuleWithActionsAndWatermelonExecGroup() throws Exception {
    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  watermelon_out_file = ctx.outputs.watermelon_output",
        "  ctx.actions.run_shell(",
        "    inputs = [],",
        "    outputs = [watermelon_out_file],",
        "    arguments = [watermelon_out_file.path],",
        "    command = 'echo hello > \"$1\"',",
        "    exec_group = 'watermelon',",
        "  )",
        "  out_file = ctx.outputs.output",
        "  ctx.actions.run_shell(",
        "    inputs = [],",
        "    outputs = [out_file],",
        "    arguments = [out_file.path],",
        "    command = 'echo hello > \"$1\"',",
        "  )",
        "with_actions = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'watermelon_output': attr.output(),",
        "    'output': attr.output(),",
        "  },",
        "  exec_groups = {",
        "    'watermelon': exec_group(toolchains = ['//rule:toolchain_type_2']),",
        "  },",
        "  toolchains = ['//rule:toolchain_type_1'],",
        ")");
  }

  @Test
  public void testSetExecGroupExecProperty() throws Exception {
    createToolchainsAndPlatforms();
    writeRuleWithActionsAndWatermelonExecGroup();

    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_actions')",
        "with_actions(",
        "  name = 'papaya',",
        "  output = 'out.txt',",
        "  watermelon_output = 'watermelon_out.txt',",
        "  exec_properties = {",
        "    'color': 'orange',",
        "    'ripeness': 'ripe',",
        "    'watermelon.color': 'pink',",
        "    'watermelon.season': 'summer',",
        "  },",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//test:papaya");

    assertThat(
            getGeneratingAction(target, "test/watermelon_out.txt").getOwner().getExecProperties())
        .containsExactly("color", "pink", "season", "summer", "ripeness", "ripe");
    assertThat(getGeneratingAction(target, "test/out.txt").getOwner().getExecProperties())
        .containsExactly("color", "orange", "ripeness", "ripe");
  }

  @Test
  public void testSetUnknownExecGroup() throws Exception {
    createToolchainsAndPlatforms();
    writeRuleWithActionsAndWatermelonExecGroup();

    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_actions')",
        "with_actions(",
        "  name = 'papaya',",
        "  output = 'out.txt',",
        "  watermelon_output = 'watermelon_out.txt',",
        "  exec_properties = {",
        "    'color': 'orange',",
        "    'watermelon.color': 'pink',",
        "    'blueberry.season': 'summer',", // non-existent exec group
        "  },",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:papaya");
    assertContainsEvent("errors encountered while analyzing target '//test:papaya'");
  }

  @Test
  public void testInheritsRuleRequirements() throws Exception {
    createToolchainsAndPlatforms();
    scratch.file(
        "test/defs.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  exec_groups = {",
        "    'watermelon': exec_group(copy_from_rule = True),",
        "  },",
        "  exec_compatible_with = ['//platform:constraint_1'],",
        "  toolchains = ['//rule:toolchain_type_1'],",
        ")");
    scratch.file("test/BUILD", "load('//test:defs.bzl', 'my_rule')", "my_rule(name = 'papaya')");

    ConfiguredTarget ct = getConfiguredTarget("//test:papaya");
    assertThat(getRuleContext(ct).getRule().getRuleClassObject().getExecGroups())
        .containsExactly(
            "watermelon",
            ExecGroup.createCopied(
                ImmutableSet.of(Label.parseAbsoluteUnchecked("//rule:toolchain_type_1")),
                ImmutableSet.of(Label.parseAbsoluteUnchecked("//platform:constraint_1"))));
  }
}
