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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.StarlarkRuleTransitionProviderTest.DummyTestLoader;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for StarlarkAttributeTransitionProvider. */
@RunWith(JUnit4.class)
public class StarlarkAttrTransitionProviderTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(new DummyTestLoader());
    return builder.build();
  }

  @Before
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  private void writeWhitelistFile() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
  }

  private void writeBasicTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "test/skylark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return [",
        "    {'//command_line_option:cpu': 'k8'},",
        "    {'//command_line_option:cpu': 'armeabi-v7a'}",
        "  ]",
        "my_transition = transition(implementation = transition_func, inputs = [],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return MyInfo(",
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
  public void testStarlarkSplitTransitionSplitAttr() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {",
        "      'amsterdam': {'//command_line_option:test_arg': ['stroopwafel']},",
        "      'paris': {'//command_line_option:test_arg': ['crepe']},",
        "  }",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:test_arg']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/skylark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly("amsterdam", "paris");
    assertThat(
            getConfiguration(splitAttr.get("amsterdam"))
                .getOptions()
                .get(TestOptions.class)
                .testArguments)
        .containsExactly("stroopwafel");
    assertThat(
            getConfiguration(splitAttr.get("paris"))
                .getOptions()
                .get(TestOptions.class)
                .testArguments)
        .containsExactly("crepe");
  }

  @Test
  public void testStarlarkListSplitTransitionSplitAttr() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return [",
        "      {'//command_line_option:test_arg': ['stroopwafel']},",
        "      {'//command_line_option:test_arg': ['crepe']},",
        "  ]",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:test_arg']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/skylark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly("0", "1");
    assertThat(
            getConfiguration(splitAttr.get("0")).getOptions().get(TestOptions.class).testArguments)
        .containsExactly("stroopwafel");
    assertThat(
            getConfiguration(splitAttr.get("1")).getOptions().get(TestOptions.class).testArguments)
        .containsExactly("crepe");
  }

  @Test
  public void testStarlarkPatchTransitionSplitAttr() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:test_arg': ['stroopwafel']}",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:test_arg']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/skylark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly(Starlark.NONE);
    assertThat(
            getConfiguration(splitAttr.get(Starlark.NONE))
                .getOptions()
                .get(TestOptions.class)
                .testArguments)
        .containsExactly("stroopwafel");
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
  public void testTargetAndRuleNotInWhitelist() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "not_whitelisted/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return [",
        "    {'//command_line_option:cpu': 'k8'},",
        "    {'//command_line_option:cpu': 'armeabi-v7a'}",
        "  ]",
        "my_transition = transition(implementation = transition_func, inputs = [],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return MyInfo(",
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
        "not_whitelisted/BUILD",
        "load('//not_whitelisted:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//not_whitelisted:test");
    assertContainsEvent("Non-whitelisted use of Starlark transition");
  }

  private void testSplitTransitionCheckAttrDeps(ConfiguredTarget target) throws Exception {
    // The regular ctx.attr.deps should be a single list with all the branches of the split merged
    // together (i.e. for aspects).
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> attrDeps =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_deps");
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
    List<ConfiguredTarget> attrDep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(attrDep).hasSize(2);
    ListMultimap<String, Object> attrDepMap = ArrayListMultimap.create();
    for (ConfiguredTarget ct : attrDep) {
      attrDepMap.put(getConfiguration(ct).getCpu(), target);
    }
    assertThat(attrDepMap).valuesForKey("k8").hasSize(1);
    assertThat(attrDepMap).valuesForKey("armeabi-v7a").hasSize(1);
  }

  private void writeReadSettingsTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  transitions = []",
        "  for cpu in settings['//command_line_option:fat_apk_cpu']:",
        "    transitions.append({'//command_line_option:cpu': cpu,})",
        "  return transitions",
        "my_transition = transition(implementation = transition_func, ",
        "  inputs = ['//command_line_option:fat_apk_cpu'],",
        "  outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return MyInfo(attr_dep = ctx.attr.dep)",
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
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    writeReadSettingsTestFiles();

    useConfiguration("--fat_apk_cpu=k8,armeabi-v7a");
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:test");

    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> splitDep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(splitDep).hasSize(2);
    assertThat(
            splitDep.stream().map(ct -> getConfiguration(ct).getCpu()).collect(Collectors.toList()))
        .containsExactly("k8", "armeabi-v7a");
  }

  private void writeOptionConversionTestFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
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
        "  return MyInfo(attr_dep = ctx.attr.dep)",
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
    List<ConfiguredTarget> dep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(dep).hasSize(1);
    assertThat(getConfiguration(Iterables.getOnlyElement(dep)).getCpu()).isEqualTo("armeabi-v7a");
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
        "  inputs = ['//command_line_option:foop', '//command_line_option:barp'],",
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
        "transition inputs [//command_line_option:foop, //command_line_option:barp] "
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test/skylark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/skylark:the-answer': 42}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/skylark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
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

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>)
                getMyInfoFromTarget(getConfiguredTarget("//test/skylark:test")).getValue("dep"));
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

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test/skylark:the-answer")))
        .isEqualTo(42);
  }

  private CoreOptions getCoreOptions(ConfiguredTarget target) {
    return getConfiguration(target).getOptions().get(CoreOptions.class);
  }

  @Test
  public void testOutputDirHash_multipleNativeOptionTransitions() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_impl(settings, attr):",
        "  return {'//command_line_option:foo': 'foosball'}",
        "foo_transition = transition(implementation = _foo_impl, inputs = [],",
        "  outputs = ['//command_line_option:foo'])",
        "def _bar_impl(settings, attr):",
        "  return {'//command_line_option:bar': 'barsball'}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = ['//command_line_option:bar'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'foo_transition', 'bar_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = foo_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bar_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    List<String> affectedOptions = getCoreOptions(test).affectedByStarlarkTransition;

    assertThat(affectedOptions).containsExactly("foo");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    affectedOptions = getCoreOptions(dep).affectedByStarlarkTransition;

    assertThat(affectedOptions).containsExactly("foo", "bar");

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo("ST-" + new Fingerprint().addString("foo=foosball").hexDigestAndReset());

    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment)
        .isEqualTo(
            "ST-"
                + new Fingerprint()
                    .addString("bar=barsball")
                    .addString("foo=foosball")
                    .hexDigestAndReset());
  }

  // Test that a no-op starlark transition to an already starlark transitioned configuration
  // results in the same configuration.
  @Test
  public void testOutputDirHash_noop_changeToSameState() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _bar_impl(settings, attr):",
        "  return {'//test:bar': 'barsball'}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = ['//test:bar'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'bar_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = bar_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bar_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'string_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "string_flag(name = 'bar', build_setting_default = '')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo(getCoreOptions(dep).transitionDirectoryNameFragment);
  }

  @Test
  public void testOutputDirHash_noop_emptyReturns() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _bar_impl(settings, attr):",
        "  return {}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = [])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'bar_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = bar_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bar_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo(getCoreOptions(dep).transitionDirectoryNameFragment);
  }

  // Test that a no-op starlark transition to the top level configuration results in a
  // different configuration.
  // TODO(bazel-team): This can be optimized. Make these the same configuration.
  @Test
  public void testOutputDirHash_noop_changeToDifferentStateAsTopLevel() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _bar_impl(settings, attr):",
        "  return {'//test:bar': 'barsball'}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = ['//test:bar'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'bar_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bar_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'string_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "string_flag(name = 'bar', build_setting_default = '')");

    // Use configuration that is same as what the transition will change.
    useConfiguration(ImmutableMap.of("//test:bar", "barsball"));

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment).isNull();
    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment).isNotNull();
  }

  // Test that setting all starlark options back to default != null hash of top level.
  // We could set some starlark options on the command line but we don't count this as a starlark
  // transition to the command line configuration will always have a null values for
  // {@code transitionDirectoryNameFragment}.
  //
  // e.g. for a build setting //foo whose default value is "foop" the following sequence
  //
  // (CommandLine) //foo=blah -> (StarlarkTransition) //foo=foop
  //
  // must create a non-null hash for after the StarlarkTransition even though later on we empty
  // the default out of the starlark map (In StarlarkTransition#validate)
  // TODO(bazel-team): This can be optimized. Make these the same configuration.
  @Test
  public void testOutputDirHash_multipleStarlarkOptionTransitions_backToDefaultCommandLine()
      throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_two_impl(settings, attr):",
        "  return {'//test:foo': 'foosballerina'}",
        "foo_two_transition = transition(implementation = _foo_two_impl, inputs = [],",
        "  outputs = ['//test:foo'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'foo_two_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = foo_two_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'string_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "string_flag(name = 'foo', build_setting_default = 'foosballerina')");

    useConfiguration(ImmutableMap.of("//test:foo", "foosball"));

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>)
                getMyInfoFromTarget(getConfiguredTarget("//test")).getValue("dep"));

    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment).isNotNull();
  }

  // Test that setting a starlark option to default (if it was already at default) doesn't
  // produce the same hash. This is because we do hashing  before scrubbing default values
  // out of {@code BuildOptions}.
  // TODO(bazel-team): This can be optimized. Make these the same configuration.
  @Test
  public void testOutputDirHash_multipleStarlarkOptionTransitions_backToDefault() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_two_impl(settings, attr):",
        "  return {'//test:foo': 'foosballerina'}",
        "foo_two_transition = transition(implementation = _foo_two_impl, inputs = [],",
        "  outputs = ['//test:foo'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'foo_two_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = foo_two_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'string_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "string_flag(name = 'foo', build_setting_default = 'foosballerina')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment)
        .isNotEqualTo(getCoreOptions(test).transitionDirectoryNameFragment);
  }

  /** See comment above {@link FunctionTransitionUtil#updateOutputDirectoryNameFragment} */
  // TODO(bazel-team): This can be optimized. Make these the same configuration.
  @Test
  public void testOutputDirHash_starlarkOption_differentBoolRepresentationsNotEquals()
      throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_impl(settings, attr):",
        "  return {'//test:foo': 1}",
        "foo_transition = transition(implementation = _foo_impl, inputs = [],",
        "  outputs = ['//test:foo'])",
        "def _foo_two_impl(settings, attr):",
        "  return {'//test:foo': True}",
        "foo_two_transition = transition(implementation = _foo_two_impl, inputs = [],",
        "  outputs = ['//test:foo'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'foo_transition', 'foo_two_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = foo_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = foo_two_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "bool_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.bool(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'bool_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "bool_flag(name = 'foo', build_setting_default = False)");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo("ST-" + new Fingerprint().addString("//test:foo=1").hexDigestAndReset());
    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment)
        .isEqualTo("ST-" + new Fingerprint().addString("//test:foo=true").hexDigestAndReset());
  }

  @Test
  public void testOutputDirHash_nativeOption_differentBoolRepresentationsEquals() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _bool_impl(settings, attr):",
        "  return {'//command_line_option:bool': '1'}",
        "bool_transition = transition(implementation = _bool_impl, inputs = [],",
        "  outputs = ['//command_line_option:bool'])",
        "def _bool_two_impl(settings, attr):",
        "  return {'//command_line_option:bool': 'true'}",
        "bool_two_transition = transition(implementation = _bool_two_impl, inputs = [],",
        "  outputs = ['//command_line_option:bool'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'bool_transition', 'bool_two_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = bool_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bool_two_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo(getCoreOptions(dep).transitionDirectoryNameFragment);
  }

  @Test
  public void testOutputDirHash_multipleStarlarkTransitions() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_impl(settings, attr):",
        "  return {'//test:foo': 'foosball'}",
        "foo_transition = transition(implementation = _foo_impl, inputs = [],",
        "  outputs = ['//test:foo'])",
        "def _bar_impl(settings, attr):",
        "  return {'//test:bar': 'barsball'}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = ['//test:bar'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'foo_transition', 'bar_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = foo_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = bar_transition), ",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'string_flag', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')",
        "string_flag(name = 'foo', build_setting_default = '')",
        "string_flag(name = 'bar', build_setting_default = '')");

    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    List<String> affectedOptions =
        getConfiguration(dep).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    // Assert that affectedOptions is empty but final fragment is still different.
    assertThat(affectedOptions).isEmpty();
    assertThat(getCoreOptions(test).transitionDirectoryNameFragment)
        .isEqualTo("ST-" + new Fingerprint().addString("//test:foo=foosball").hexDigestAndReset());
    assertThat(getCoreOptions(dep).transitionDirectoryNameFragment)
        .isEqualTo(
            "ST-"
                + new Fingerprint()
                    .addString("//test:bar=barsball")
                    .addString("//test:foo=foosball")
                    .hexDigestAndReset());
  }

  // This test is massive but mostly exists to ensure that all the parts are working together
  // properly amidst multiple complicated transitions.
  @Test
  public void testOutputDirHash_multipleMixedTransitions() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _foo_impl(settings, attr):",
        "  return {'//command_line_option:foo': 'foosball'}",
        "foo_transition = transition(implementation = _foo_impl, inputs = [],",
        "  outputs = ['//command_line_option:foo'])",
        "def _bar_impl(settings, attr):",
        "  return {'//command_line_option:bar': 'barsball'}",
        "bar_transition = transition(implementation = _bar_impl, inputs = [],",
        "  outputs = ['//command_line_option:bar'])",
        "def _zee_impl(settings, attr):",
        "  return {'//test:zee': 'zeesball'}",
        "zee_transition = transition(implementation = _zee_impl, inputs = [],",
        "  outputs = ['//test:zee'])",
        "def _xan_impl(settings, attr):",
        "  return {'//test:xan': 'xansball'}",
        "xan_transition = transition(implementation = _xan_impl, inputs = [],",
        "  outputs = ['//test:xan'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load(",
        "  '//test:transitions.bzl',",
        "  'foo_transition',",
        "  'bar_transition',",
        "  'zee_transition',",
        "  'xan_transition')",
        "def _impl_a(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule_a = rule(",
        "  implementation = _impl_a,",
        "  cfg = foo_transition,", // transition #1
        "  attrs = {",
        "    'dep': attr.label(cfg = zee_transition), ", // transition #2
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _impl_b(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule_b = rule(",
        "  implementation = _impl_b,",
        "  cfg = bar_transition,", // transition #3
        "  attrs = {",
        "    'dep': attr.label(cfg = xan_transition), ", // transition #4
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.string(flag = True),",
        ")",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule_a', 'my_rule_b', 'string_flag', 'simple')",
        "my_rule_a(name = 'top', dep = ':middle')",
        "my_rule_b(name = 'middle', dep = 'bottom')",
        "simple(name = 'bottom')",
        "string_flag(name = 'zee', build_setting_default = '')",
        "string_flag(name = 'xan', build_setting_default = '')");

    // test:top (foo_transition)
    ConfiguredTarget top = getConfiguredTarget("//test:top");

    List<String> affectedOptionsTop =
        getConfiguration(top).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsTop).containsExactly("foo");
    assertThat(getConfiguration(top).getOptions().getStarlarkOptions()).isEmpty();
    assertThat(getCoreOptions(top).transitionDirectoryNameFragment)
        .isEqualTo("ST-" + new Fingerprint().addString("foo=foosball").hexDigestAndReset());

    // test:middle (foo_transition, zee_transition, bar_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget middle =
        Iterables.getOnlyElement((List<ConfiguredTarget>) getMyInfoFromTarget(top).getValue("dep"));

    List<String> affectedOptionsMiddle =
        getConfiguration(middle).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsMiddle).containsExactly("foo", "bar");
    assertThat(getConfiguration(middle).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseAbsoluteUnchecked("//test:zee"), "zeesball"));
    assertThat(getCoreOptions(middle).transitionDirectoryNameFragment)
        .isEqualTo(
            "ST-"
                + new Fingerprint()
                    .addString("//test:zee=zeesball")
                    .addString("bar=barsball")
                    .addString("foo=foosball")
                    .hexDigestAndReset());

    // test:bottom (foo_transition, zee_transition, bar_transition, xan_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget bottom =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(middle).getValue("dep"));

    List<String> affectedOptionsBottom =
        getConfiguration(bottom).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsBottom).containsExactly("foo", "bar");
    assertThat(getConfiguration(bottom).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseAbsoluteUnchecked("//test:zee"), "zeesball"),
            Maps.immutableEntry(Label.parseAbsoluteUnchecked("//test:xan"), "xansball"));
    assertThat(getCoreOptions(bottom).transitionDirectoryNameFragment)
        .isEqualTo(
            "ST-"
                + new Fingerprint()
                    .addString("//test:xan=xansball")
                    .addString("//test:zee=zeesball")
                    .addString("bar=barsball")
                    .addString("foo=foosball")
                    .hexDigestAndReset());
  }

  @Test
  public void testTransitionOnBuildSetting_badValue() throws Exception {
    setSkylarkSemanticsOptions(
        "--experimental_build_setting_api=true", "--experimental_starlark_config_transitions");
    writeWhitelistFile();
    writeBuildSettingsBzl();
    scratch.file(
        "test/skylark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test/skylark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/skylark:the-answer': 'What do you get if you multiply six by nine?'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/skylark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
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

  /**
   * Regression test for b/147245129.
   *
   * <p>This tests that when exec transitions are applied from target configurations that are
   * identical except for different Starlark flags, outputs do not conflict.
   */
  @Test
  public void testBuildSettingTransitionsWorkWithExecTransitions() throws Exception {
    writeWhitelistFile();
    // This setup creates an int_flag_reading_rule whose output is the value of an int_flag (which
    // guarantees actions in configurations with different Starlark flag values are different). It
    // then makes this a genrule exec tool (so it applies after an exec transition). And finally
    // creates a build_setting_changing_rule that changes the int_flag's value and depends on the
    // genrule. So building the genrule at both the top-level and under the
    // build_setting_changing_rule triggers the test scenario.
    scratch.file(
        "test/skylark/rules.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl, build_setting = config.int())",
        "def _transition_impl(settings, attr):",
        "  return {'//test/skylark:the-answer': 42}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/skylark:the-answer'])",
        "def _int_impl(ctx):",
        "  value = ctx.attr._int_dep[BuildSettingInfo].value",
        "  ctx.actions.write(ctx.outputs.out, str(value))",
        "int_flag_reading_rule = rule(",
        "  implementation = _int_impl,",
        "  attrs = {",
        "    '_int_dep': attr.label(default = '//test/skylark:the-answer'),",
        "    'out': attr.output()",
        "  })",
        "def _rule_impl(ctx):",
        "  pass",
        "build_setting_changing_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition, allow_single_file = True),",
        "    '_whitelist_function_transition': attr.label(",
        "      default = '//tools/whitelists/function_transition_whitelist'),",
        "  })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'build_setting_changing_rule', 'int_flag',",
        "  'int_flag_reading_rule')",
        "int_flag(",
        "    name = 'the-answer',",
        "    build_setting_default = 0)",
        "genrule(",
        "    name = 'with_exec_tool',",
        "    srcs = [],",
        "    outs = ['with_exec_tool.out'],",
        "    cmd = 'echo hi > $@',",
        "    exec_tools = [':int_reader'])",
        "int_flag_reading_rule(",
        "    name = 'int_reader',",
        "    out = 'int_reader.out')",
        "build_setting_changing_rule(",
        "    name = 'transitioner',",
        "    dep = ':with_exec_tool')");
    // Note: calling getConfiguredTarget for each target doesn't activate conflict detection.
    update(
        ImmutableList.of("//test/skylark:transitioner", "//test/skylark:with_exec_tool.out"),
        /*keepGoing=*/ false,
        LOADING_PHASE_THREADS,
        /*doAnalysis=*/ true,
        new EventBus());
    assertNoEvents();
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
