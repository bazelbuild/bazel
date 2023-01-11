// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.ExecGroup;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for automatic exec groups. */
@RunWith(JUnit4.class)
public class AutoExecGroupsTest extends BuildViewTestCase {
  /**
   * Sets up two toolchains types, each with a single toolchain implementation and a single
   * exec_compatible_with platform.
   *
   * <p>toolchain_type_1 -> foo_toolchain -> exec_compatible_with platform_1 toolchain_type_2 ->
   * bar_toolchain -> exec_compatible_with platform_2
   */
  @Before
  public void createToolchainsAndPlatforms() throws Exception {
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
        "    target_compatible_with = ['//platforms:constraint_1'],",
        "    exec_compatible_with = ['//platforms:constraint_1'],",
        "    toolchain = ':foo',",
        ")",
        "test_toolchain(",
        "    name = 'bar',",
        ")",
        "toolchain(",
        "    name = 'bar_toolchain',",
        "    toolchain_type = '//rule:toolchain_type_2',",
        "    target_compatible_with = ['//platforms:constraint_1'],",
        "    exec_compatible_with = ['//platforms:constraint_2'],",
        "    toolchain = ':bar',",
        ")");

    scratch.file(
        "platforms/BUILD",
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
        "    exec_properties = {",
        "        'watermelon.ripeness': 'unripe',",
        "        'watermelon.color': 'red',",
        "    },",
        ")");
  }

  @Before
  public void setup() throws Exception {
    useConfiguration();
  }

  @Override
  public void useConfiguration(String... args) throws Exception {
    String[] flags = {
      "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:bar_toolchain",
      "--platforms=//platforms:platform_1",
      "--extra_execution_platforms=//platforms:platform_1,//platforms:platform_2"
    };

    super.useConfiguration(ObjectArrays.concat(flags, args, String.class));
  }

  /**
   * Creates custom rule which produces action with `actionParameters`, adds `extraAttributes` and
   * defines `toolchains`.
   */
  private void createCustomRule(String actionParameters, String extraAttributes, String toolchains)
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file(ctx.label.name + '_dummy_output.jar')",
        "  ctx.actions.run(",
        "    ",
        actionParameters,
        "    outputs = [output_jar],",
        "    executable = '//toolchain:foo_toolchain',",
        "  )",
        "  return [DefaultInfo(files = depset([output_jar]))]",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = 'exec'),",
        "     ",
        extraAttributes,
        "  },",
        "  toolchains = " + toolchains + ",",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'custom_rule')",
        "custom_rule(name = 'custom_rule_name')");
  }

  @Test
  public void automaticExecutionGroups_disabledAndAttributeFalse_disabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = False),",
        /* toolchains= */ "['//rule:toolchain_type_1']");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isEmpty();
  }

  @Test
  public void automaticExecutionGroups_disabledAndAttributeTrue_enabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = True),",
        /* toolchains= */ "['//rule:toolchain_type_1']");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isNotEmpty();
  }

  @Test
  public void automaticExecutionGroups_disabledAndAttributeNotSet_disabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isEmpty();
  }

  @Test
  public void automaticExecutionGroups_enabledAndAttributeFalse_disabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = False),",
        /* toolchains= */ "['//rule:toolchain_type_1']");
    useConfiguration("--incompatible_auto_exec_groups");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isEmpty();
  }

  @Test
  public void automaticExecutionGroups_enabledAndAttributeTrue_enabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = True)",
        /* toolchains= */ "['//rule:toolchain_type_1']");
    useConfiguration("--incompatible_auto_exec_groups");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isNotEmpty();
  }

  @Test
  public void automaticExecutionGroups_enabledAndAttributeNotSet_enabled() throws Exception {
    createCustomRule(
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']");
    useConfiguration("--incompatible_auto_exec_groups");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ImmutableMap<String, ExecGroup> execGroups =
        getRuleContext(target).getExecGroups().execGroups();

    assertThat(execGroups).isNotEmpty();
  }
}
