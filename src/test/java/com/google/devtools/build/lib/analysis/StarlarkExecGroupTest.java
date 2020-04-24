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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for exec groups. Functionality related to rule context tested in {@link
 * com.google.devtools.build.lib.skylark.SkylarkRuleContextTest}.
 */
@RunWith(JUnit4.class)
public class StarlarkExecGroupTest extends BuildViewTestCase {

  @Before
  public final void setUp() throws Exception {
    setStarlarkSemanticsOptions("--experimental_exec_groups");
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
  }

  @Test
  public void testExecGroupTransition() throws Exception {
    createToolchainsAndPlatforms();
    useConfiguration(
        "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:bar_toolchain",
        "--platforms=//platform:platform_1",
        "--extra_execution_platforms=//platform:platform_1,//platform:platform_2");

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

    assertThat(dep.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseAbsoluteUnchecked("//platform:platform_1"));
    assertThat(execGroupDep.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseAbsoluteUnchecked("//platform:platform_2"));
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
}
