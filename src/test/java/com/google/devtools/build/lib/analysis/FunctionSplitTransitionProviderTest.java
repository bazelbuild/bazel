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
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for FunctionSplitTransitionProvider.
 */
@RunWith(JUnit4.class)
public class FunctionSplitTransitionProviderTest extends BuildViewTestCase {
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

    scratch.file(
        "test/skylark/my_rule.bzl",
        "def transition_func(settings):",
        "  return {",
        "      't0': {'cpu': 'k8'},",
        "      't1': {'cpu': 'armeabi-v7a'},",
        "  }",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
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
    assertThat(getConfiguration(splitDeps.get("armeabi-v7a").get(0)).getCpu()).isEqualTo("armeabi-v7a");
    assertThat(getConfiguration(splitDeps.get("armeabi-v7a").get(1)).getCpu()).isEqualTo("armeabi-v7a");
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
        "def transition_func(settings):",
        "  transitions = {}",
        "  for cpu in settings['fat_apk_cpu']:",
        "    transitions[cpu] = {",
        "      'cpu': cpu,",
        "    }",
        "  return transitions",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
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
        "def transition_func(settings):",
        "  return {",
        "    'cpu': 'armeabi-v7a',",
        "    'dynamic_mode': 'off',",
        "    'crosstool_top': '//android/crosstool:everything',",
        "  }",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
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
  public void testOptionConversionDynamicMode() throws Exception {
    // TODO(waltl): check that dynamic_mode is parsed properly.
  }

  @Test
  public void testOptionConversionCrosstoolTop() throws Exception {
    // TODO(waltl): check that crosstool_top is parsed properly.
  }
}
