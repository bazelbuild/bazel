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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for StarlarkAttributeTransitionProvider. */
@RunWith(JUnit4.class)
public class StarlarkAttrTransitionProviderTest extends BuildViewTestCase {

  private void writeWhitelistFile() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/skylark/...',",
        "    ],",
        ")");
  }

  private void writeBasicTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {",
        "      't0': {'//command_line_option:cpu': 'k8'},",
        "      't1': {'//command_line_option:cpu': 'armeabi-v7a'},",
        "  }",
        "my_transition = transition(implementation = transition_func, inputs = [],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return struct(",
        "    split_attr_deps = ctx.split_attr.deps,",
        "    split_attr_dep = ctx.split_attr.dep,",
        "    k8_deps = ctx.split_attr.deps.get('k8', None),",
        "    attr_deps = ctx.attr.deps,",
        "    attr_dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'deps': attr.label_list(cfg = my_transition),",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', deps = [':main1', ':main2'], dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])",
        "cc_binary(name = 'main2', srcs = ['main2.c'])");
  }

  @Test
  public void testFunctionSplitTransitionCheckSplitAttrDeps() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckSplitAttrDeps(getConfiguredTarget("//test/skylark:test"));
  }

  @Test
  public void testFunctionSplitTransitionCheckSplitAttrDep() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckSplitAttrDep(getConfiguredTarget("//test/skylark:test"));
  }

  @Test
  public void testFunctionSplitTransitionCheckAttrDeps() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckAttrDeps(getConfiguredTarget("//test/skylark:test"));
  }

  @Test
  public void testFunctionSplitTransitionCheckAttrDep() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckAttrDep(getConfiguredTarget("//test/skylark:test"));
  }

  @Test
  public void testFunctionSplitTransitionCheckK8Deps() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckK8Deps(getConfiguredTarget("//test/skylark:test"));
  }

  @Test
  public void testTargetNotInWhitelist() throws Exception {
    writeBasicTestFiles();
    scratch.file(
        "test/not_whitelisted/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/not_whitelisted:test");
    assertContainsEvent("Non-whitelisted use of Starlark transition");
  }

  private void testSplitTransitionCheckSplitAttrDeps(ConfiguredTarget target) throws Exception {
    // Check that ctx.split_attr.deps has this structure:
    // {
    //   "k8": [ConfiguredTarget],
    //   "armeabi-v7a": [ConfiguredTarget],
    // }
    @SuppressWarnings("unchecked")
    Map<String, List<ConfiguredTarget>> splitDeps =
        (Map<String, List<ConfiguredTarget>>) target.get("split_attr_deps");
    assertThat(splitDeps).containsKey("k8");
    assertThat(splitDeps).containsKey("armeabi-v7a");
    assertThat(splitDeps.get("k8")).hasSize(2);
    assertThat(splitDeps.get("armeabi-v7a")).hasSize(2);
    assertThat(getConfiguration(splitDeps.get("k8").get(0)).getCpu()).isEqualTo("k8");
    assertThat(getConfiguration(splitDeps.get("k8").get(1)).getCpu()).isEqualTo("k8");
    assertThat(getConfiguration(splitDeps.get("armeabi-v7a").get(0)).getCpu())
        .isEqualTo("armeabi-v7a");
    assertThat(getConfiguration(splitDeps.get("armeabi-v7a").get(1)).getCpu())
        .isEqualTo("armeabi-v7a");
  }

  private void testSplitTransitionCheckSplitAttrDep(ConfiguredTarget target) throws Exception {
    // Check that ctx.split_attr.dep has this structure (that is, that the values are not lists):
    // {
    //   "k8": ConfiguredTarget,
    //   "armeabi-v7a": ConfiguredTarget,
    // }
    @SuppressWarnings("unchecked")
    Map<String, ConfiguredTarget> splitDep =
        (Map<String, ConfiguredTarget>) target.get("split_attr_dep");
    assertThat(splitDep).containsKey("k8");
    assertThat(splitDep).containsKey("armeabi-v7a");
    assertThat(getConfiguration(splitDep.get("k8")).getCpu()).isEqualTo("k8");
    assertThat(getConfiguration(splitDep.get("armeabi-v7a")).getCpu()).isEqualTo("armeabi-v7a");
  }

  private void testSplitTransitionCheckAttrDeps(ConfiguredTarget target) throws Exception {
    // The regular ctx.attr.deps should be a single list with all the branches of the split merged
    // together (i.e. for aspects).
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> attrDeps = (List<ConfiguredTarget>) target.get("attr_deps");
    assertThat(attrDeps).hasSize(4);
    ListMultimap<String, Object> attrDepsMap = ArrayListMultimap.create();
    for (ConfiguredTarget ct : attrDeps) {
      attrDepsMap.put(getConfiguration(ct).getCpu(), target);
    }
    assertThat(attrDepsMap).valuesForKey("k8").hasSize(2);
    assertThat(attrDepsMap).valuesForKey("armeabi-v7a").hasSize(2);
  }

  private void testSplitTransitionCheckAttrDep(ConfiguredTarget target) throws Exception {
    // Check that even though my_rule.dep is defined as a single label, ctx.attr.dep is still a list
    // with multiple ConfiguredTarget objects because of the two different CPUs.
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> attrDep = (List<ConfiguredTarget>) target.get("attr_dep");
    assertThat(attrDep).hasSize(2);
    ListMultimap<String, Object> attrDepMap = ArrayListMultimap.create();
    for (ConfiguredTarget ct : attrDep) {
      attrDepMap.put(getConfiguration(ct).getCpu(), target);
    }
    assertThat(attrDepMap).valuesForKey("k8").hasSize(1);
    assertThat(attrDepMap).valuesForKey("armeabi-v7a").hasSize(1);
  }

  private void testSplitTransitionCheckK8Deps(ConfiguredTarget target) throws Exception {
    // Check that the deps were correctly accessed from within Skylark.
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> k8Deps = (List<ConfiguredTarget>) target.get("k8_deps");
    assertThat(k8Deps).hasSize(2);
    assertThat(getConfiguration(k8Deps.get(0)).getCpu()).isEqualTo("k8");
    assertThat(getConfiguration(k8Deps.get(1)).getCpu()).isEqualTo("k8");
  }

  private void writeReadSettingsTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  transitions = {}",
        "  for cpu in settings['//command_line_option:fat_apk_cpu']:",
        "    transitions[cpu] = {",
        "      '//command_line_option:cpu': cpu,",
        "    }",
        "  return transitions",
        "my_transition = transition(implementation = transition_func, ",
        "  inputs = ['//command_line_option:fat_apk_cpu'],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return struct(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
  }

  @Test
  public void testReadSettingsSplitDepAttrDep() throws Exception {
    // Check that ctx.split_attr.dep has this structure:
    // {
    //   "k8": ConfiguredTarget,
    //   "armeabi-v7a": ConfiguredTarget,
    // }
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    writeReadSettingsTestFiles();

    useConfiguration("--fat_apk_cpu=k8,armeabi-v7a");
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:test");

    @SuppressWarnings("unchecked")
    Map<String, ConfiguredTarget> splitDep =
        (Map<String, ConfiguredTarget>) target.get("split_attr_dep");
    assertThat(splitDep).containsKey("k8");
    assertThat(splitDep).containsKey("armeabi-v7a");
    assertThat(getConfiguration(splitDep.get("k8")).getCpu()).isEqualTo("k8");
    assertThat(getConfiguration(splitDep.get("armeabi-v7a")).getCpu()).isEqualTo("armeabi-v7a");
  }

  private void writeOptionConversionTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {",
        "    '//command_line_option:cpu': 'armeabi-v7a',",
        "    '//command_line_option:dynamic_mode': 'off',",
        "    '//command_line_option:crosstool_top': '//android/crosstool:everything',",
        "  }",
        "my_transition = transition(implementation = transition_func, inputs = [],",
        "  outputs = ['//command_line_option:cpu',",
        "            '//command_line_option:dynamic_mode',",
        "            '//command_line_option:crosstool_top'])",
        "def impl(ctx): ",
        "  return struct(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
  }

  @Test
  public void testOptionConversionCpu() throws Exception {
    writeOptionConversionTestFiles();
    BazelMockAndroidSupport.setupNdk(mockToolsConfig);

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:test");

    @SuppressWarnings("unchecked")
    Map<String, ConfiguredTarget> splitDep =
        (Map<String, ConfiguredTarget>) target.get("split_attr_dep");
    assertThat(splitDep).containsKey("armeabi-v7a");
    assertThat(getConfiguration(splitDep.get("armeabi-v7a")).getCpu()).isEqualTo("armeabi-v7a");
  }

  @Test
  public void testUndeclaredOptionKey() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "transition function returned undeclared output '//command_line_option:cpu'");
  }

  @Test
  public void testDeclaredOutputNotReturned() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:cpu',",
        "             '//command_line_option:host_cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "transition outputs [//command_line_option:host_cpu] were not "
            + "defined by transition function");
  }

  @Test
  public void testSettingsContainOnlyInputs() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  if (len(settings) != 2",
        "      or (not settings['//command_line_option:host_cpu'])",
        "      or (not settings['//command_line_option:cpu'])):",
        "    fail()",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = ['//command_line_option:host_cpu',",
        "            '//command_line_option:cpu'],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    assertThat(getConfiguredTarget("//test/skylark:test")).isNotNull();
  }

  @Test
  public void testInvalidInputKey() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = ['cpu'], outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "invalid transition input 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  @Test
  public void testInvalidNativeOptionInput() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = ['//command_line_option:foo', '//command_line_option:bar'],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "transition inputs [//command_line_option:foo, //command_line_option:bar] "
            + "do not correspond to valid settings");
  }

  @Test
  public void testInvalidNativeOptionOutput() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:foobarbaz': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = ['//command_line_option:cpu'], outputs = ['//command_line_option:foobarbaz'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "transition output '//command_line_option:foobarbaz' "
            + "does not correspond to a valid setting");
  }

  @Test
  public void testInvalidOutputKey() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [], outputs = ['cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "invalid transition output 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  @Test
  public void testInvalidOptionValue() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 1}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [], outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent("Invalid value type for option 'cpu'");
  }

  @Test
  public void testDuplicateOutputs() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 1}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:cpu',",
        "             '//command_line_option:foo',",
        "             '//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent("duplicate transition output '//command_line_option:cpu'");
  }

  @Test
  public void testInvalidNativeOptionOutput_analysisTest() throws Exception {
    scratch.file(
        "test/skylark/my_rule.bzl",
        "my_transition = analysis_test_transition(",
        "  settings = {'//command_line_option:foobarbaz': 'k8'})",
        "def impl(ctx): ",
        "  return []",
        "my_rule_test = rule(",
        "  implementation = impl,",
        "  analysis_test = True,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule_test')",
        "my_rule_test(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "transition output '//command_line_option:foobarbaz' "
            + "does not correspond to a valid setting");
  }

  @Test
  public void testInvalidOutputKey_analysisTest() throws Exception {
    scratch.file(
        "test/skylark/my_rule.bzl",
        "my_transition = analysis_test_transition(",
        "  settings = {'cpu': 'k8'})",
        "def impl(ctx): ",
        "  return []",
        "my_rule_test = rule(",
        "  implementation = impl,",
        "  analysis_test = True,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_rule_test')",
        "my_rule_test(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "invalid transition output 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  @Test
  public void testCannotTransitionWithoutFlag() throws Exception {
    writeBasicTestFiles();
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=false");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "Starlark-defined transitions on rule attributes is experimental and disabled by default");
  }

  private void writeBuildSettingsBzl() throws Exception {
    scratch.file(
        "test/skylark/build_settings.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl, build_setting = config.int(flag=True))");
  }

  private void writeRulesWithAttrTransitionBzl() throws Exception {
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//test/skylark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/skylark:the-answer': 42}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/skylark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return struct(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "      default = '//tools/whitelists/function_transition_whitelist'),",
        "  }",
        ")",
        "def _dep_rule_impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.attr.fact[BuildSettingInfo].value)]",
        "dep_rule_impl = rule(",
        "  implementation = _dep_rule_impl,",
        "  attrs = {",
        "    'fact': attr.label(default = '//test/skylark:the-answer'),",
        "  }",
        ")");
  }

  @Test
  public void testTransitionOnBuildSetting_fromDefault() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api=true");
    writeWhitelistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "load('//test/skylark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer', build_setting_default = 0)");

    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getConfiguredTarget("//test/skylark:test").get("dep"));
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test/skylark:the-answer")))
        .isEqualTo(42);
  }

  @Test
  public void testTransitionOnBuildSetting_fromCommandLine() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api=true");
    writeWhitelistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "load('//test/skylark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer', build_setting_default = 0)");

    useConfiguration(ImmutableMap.of("//test/skylark:the-answer", 7));
    ConfiguredTarget test = getConfiguredTarget("//test/skylark:test");
    assertThat(
            getConfiguration(test)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test/skylark:the-answer")))
        .isEqualTo(7);

    ConfiguredTarget dep = Iterables.getOnlyElement((List<ConfiguredTarget>) test.get("dep"));
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test/skylark:the-answer")))
        .isEqualTo(42);
  }

  @Test
  public void testTransitionOnBuildSetting_badValue() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_build_setting_api=true", "--experimental_starlark_config_transitions");
    writeWhitelistFile();
    writeBuildSettingsBzl();
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//test/skylark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/skylark:the-answer': 'What do you get if you multiply six by nine?'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/skylark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return struct(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "      default = '//tools/whitelists/function_transition_whitelist'),",
        "  }",
        ")",
        "def _dep_rule_impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.attr.fact[BuildSettingInfo].value)]",
        "dep_rule_impl = rule(",
        "  implementation = _dep_rule_impl,",
        "  attrs = {",
        "    'fact': attr.label(default = '//test/skylark:the-answer'),",
        "  }",
        ")");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "load('//test/skylark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(",
        "  name = 'the-answer',",
        "  build_setting_default = 0,",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "expected value of type 'int' for //test/skylark:the-answer, "
            + "but got \"What do you get if you multiply six by nine?\" (string)");
  }

  @Test
  public void testTransitionOnBuildSetting_noSuchTarget() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_build_setting_api=true", "--experimental_starlark_config_transitions");
    writeWhitelistFile();
    writeRulesWithAttrTransitionBzl();
    // Still need to write this file in order not to rewrite rules.bzl file (has loads from this
    // file)
    writeBuildSettingsBzl();
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "no such target '//test/skylark:the-answer': target "
            + "'the-answer' not declared in package");
  }

  @Test
  public void testTransitionOnBuildSetting_notABuildSetting() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_build_setting_api=true", "--experimental_starlark_config_transitions");
    writeWhitelistFile();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/skylark/build_settings.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl)");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "load('//test/skylark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "attempting to transition on '//test/skylark:the-answer' which "
            + "is not a build setting");
  }

  @Test
  public void testTransitionOnBuildSetting_aliasedBuildSetting() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api=true");
    writeWhitelistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'my_rule')",
        "load('//test/skylark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-secret-answer', build_setting_default = 0)",
        "alias(name = 'the-answer', actual = ':the-secret-answer')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test");
    assertContainsEvent(
        "attempting to transition on '//test/skylark:the-answer' which "
            + "is not a build setting");
  }

  @Test
  public void testOptionConversionDynamicMode() throws Exception {
    // TODO(waltl): check that dynamic_mode is parsed properly.
  }

  @Test
  public void testOptionConversionCrosstoolTop() throws Exception {
    // TODO(waltl): check that crosstool_top is parsed properly.
  }
}
