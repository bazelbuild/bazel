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

import static com.google.common.collect.Multimaps.toMultimap;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.skyframe.BuildConfigurationFunction;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkAttributeTransitionProvider}. */
@RunWith(JUnit4.class)
public final class StarlarkAttrTransitionProviderTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  @Before
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");
  }

  private static StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget)
      throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//myinfo:myinfo.bzl"), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  private void writeAllowlistFile() throws Exception {
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
  }

  private void writeBasicTestFiles() throws Exception {
    writeAllowlistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    attr_deps = ctx.split_attr.deps,",
        "    attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'deps': attr.label_list(cfg = my_transition),",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', deps = [':main1', ':main2'], dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])",
        "cc_binary(name = 'main2', srcs = ['main2.c'])");
  }

  @Test
  public void testStarlarkSplitTransitionSplitAttr() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {",
        "      'amsterdam': {'//command_line_option:foo': 'stroopwafel'},",
        "      'paris': {'//command_line_option:foo': 'crepe'},",
        "  }",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:foo']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/starlark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly("amsterdam", "paris");
    assertThat(
            getConfiguration(splitAttr.get("amsterdam"))
                .getOptions()
                .get(DummyTestOptions.class)
                .foo)
        .isEqualTo("stroopwafel");
    assertThat(
            getConfiguration(splitAttr.get("paris")).getOptions().get(DummyTestOptions.class).foo)
        .isEqualTo("crepe");
  }

  @Test
  public void testStarlarkListSplitTransitionSplitAttr() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return [",
        "      {'//command_line_option:foo': 'stroopwafel'},",
        "      {'//command_line_option:foo': 'crepe'},",
        "  ]",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:foo']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/starlark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly("0", "1");
    assertThat(getConfiguration(splitAttr.get("0")).getOptions().get(DummyTestOptions.class).foo)
        .isEqualTo("stroopwafel");
    assertThat(getConfiguration(splitAttr.get("1")).getOptions().get(DummyTestOptions.class).foo)
        .isEqualTo("crepe");
  }

  @Test
  public void testStarlarkPatchTransitionSplitAttr() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:foo': 'stroopwafel'}",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:foo']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'simple_rule', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    @SuppressWarnings("unchecked")
    Map<Object, ConfiguredTarget> splitAttr =
        (Map<Object, ConfiguredTarget>)
            getMyInfoFromTarget(getConfiguredTarget("//test/starlark:test"))
                .getValue("split_attr_dep");
    assertThat(splitAttr.keySet()).containsExactly(Starlark.NONE);
    assertThat(
            getConfiguration(splitAttr.get(Starlark.NONE))
                .getOptions()
                .get(DummyTestOptions.class)
                .foo)
        .isEqualTo("stroopwafel");
  }

  @Test
  public void testStarlarkConfigSplitAttr() throws Exception {
    // This is a customized test case for b/152078818, where a starlark transition that takes a
    // starlark config as input caused a failure when no custom values were provided for the config.
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _build_setting_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _build_setting_impl,",
        "  build_setting = config.string(flag=True)",
        ")",
        "def transition_func(settings, attr):",
        "  return {'amsterdam': {'//command_line_option:foo': 'stroopwafel'}}",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = ['//test/starlark:custom_arg'],",
        "  outputs = ['//command_line_option:foo']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(split_attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")",
        "def _s_impl_e(ctx):",
        "  return []",
        "simple_rule = rule(_s_impl_e)");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'simple_rule', 'my_rule', 'string_flag')",
        "string_flag(name='custom_arg', build_setting_default='ski')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple_rule(name = 'dep')");

    // Run the analysis phase with the default options, i.e. no custom flags first.
    getConfiguredTarget("//test/starlark:test");

    // b/152078818 was unique in that an error was hidden until the next run due to how event replay
    // was done. Test it by supplying a value to the starlark config, which should trigger the
    // analysis phase again.
    useConfiguration(ImmutableMap.of("//test/starlark:custom_arg", "snowboard"));
    getConfiguredTarget("//test/starlark:test");
  }

  @Test
  public void testFunctionSplitTransitionCheckAttrDeps() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckAttrDeps(getConfiguredTarget("//test/starlark:test"));
  }

  @Test
  public void testFunctionSplitTransitionCheckAttrDep() throws Exception {
    writeBasicTestFiles();
    testSplitTransitionCheckAttrDep(getConfiguredTarget("//test/starlark:test"));
  }

  @Test
  public void testNullTransitionAttributeWithMissingSkyframeDep() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {'//test/starlark/nested:flag': True}",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//test/starlark/nested:flag']",
        ")",
        "def _impl(ctx): ",
        "  return MyInfo(data = ctx.attr.data)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'data': attr.label_list(cfg = my_transition, allow_files = True, default = []),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")",
        "def _basic_impl(ctx):",
        "  return []",
        "bool_flag = rule(",
        "  implementation = _basic_impl,",
        "  build_setting = config.bool(flag = True),",
        ")");

    scratch.file(
        "test/starlark/nested/BUILD",
        "load('//test/starlark:rules.bzl', 'bool_flag')",
        "bool_flag(name = 'flag', build_setting_default = False)");
    scratch.file("test/starlark/some_file.txt", "Random content");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "my_rule(name = 'test', data = ['some_file.txt'])");

    assertThat(getConfiguredTarget("//test/starlark:test")).isNotNull();
  }

  @Test
  public void testTargetAndRuleNotInAllowlist() throws Exception {
    writeAllowlistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "not_allowlisted/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "not_allowlisted/BUILD",
        "load('//not_allowlisted:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//not_allowlisted:test");
    assertContainsEvent("Non-allowlisted use of Starlark transition");
  }

  private void testSplitTransitionCheckAttrDeps(ConfiguredTarget target) throws Exception {
    @SuppressWarnings("unchecked")
    Dict<String, List<ConfiguredTarget>> attrDeps =
        (Dict<String, List<ConfiguredTarget>>) getMyInfoFromTarget(target).getValue("attr_deps");
    assertThat(attrDeps.size()).isEqualTo(2);
    ListMultimap<String, Object> attrDepsMap =
        attrDeps.values().stream()
            .flatMap(Collection::stream)
            .map(ct -> getConfiguration(ct).getCpu())
            .collect(toMultimap(cpu -> cpu, (ignored) -> target, ArrayListMultimap::create));
    assertThat(attrDepsMap).valuesForKey("k8").hasSize(2);
    assertThat(attrDepsMap).valuesForKey("armeabi-v7a").hasSize(2);
  }

  private void testSplitTransitionCheckAttrDep(ConfiguredTarget target) throws Exception {
    // Check that even though my_rule.dep is defined as a single label, ctx.attr.dep is still a list
    // with multiple ConfiguredTarget objects because of the two different CPUs.
    @SuppressWarnings("unchecked")
    Dict<String, ConfiguredTarget> attrDep =
        (Dict<String, ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(attrDep.size()).isEqualTo(2);
    ListMultimap<String, Object> attrDepMap =
        attrDep.values().stream()
            .map(ct -> getConfiguration(ct).getCpu())
            .collect(toMultimap(cpu -> cpu, (ignored) -> target, ArrayListMultimap::create));
    assertThat(attrDepMap).valuesForKey("k8").hasSize(1);
    assertThat(attrDepMap).valuesForKey("armeabi-v7a").hasSize(1);
  }

  private void writeReadSettingsTestFiles() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "  return MyInfo(attr_dep = ctx.split_attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
  }

  @Test
  public void testReadSettingsSplitDepAttrDep() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    writeReadSettingsTestFiles();

    useConfiguration("--fat_apk_cpu=k8,armeabi-v7a");
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:test");

    @SuppressWarnings("unchecked")
    Dict<String, ConfiguredTarget> splitDep =
        (Dict<String, ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(splitDep.size()).isEqualTo(2);
    List<String> cpus =
        splitDep.values().stream()
            .map(ct -> getConfiguration(ct).getCpu())
            .collect(Collectors.toList());
    assertThat(cpus).containsExactly("k8", "armeabi-v7a");
  }

  private void writeOptionConversionTestFiles() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
  }

  @Test
  public void testOptionConversionCpu() throws Exception {
    writeOptionConversionTestFiles();
    BazelMockAndroidSupport.setupNdk(mockToolsConfig); // cc_binary needs this

    ConfiguredTarget target = getConfiguredTarget("//test/starlark:test");

    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> dep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("attr_dep");
    assertThat(dep).hasSize(1);
    assertThat(getConfiguration(Iterables.getOnlyElement(dep)).getCpu()).isEqualTo("armeabi-v7a");
  }

  private void writeReadAndPassthroughOptionsTestFiles() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/skylark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "settings_under_test = {",
        "  '//command_line_option:cpu': 'armeabi-v7a',",
        "  '//command_line_option:compilation_mode': 'dbg',",
        "  '//command_line_option:crosstool_top': '@//android/crosstool:everything',",
        "  '//command_line_option:platform_suffix': 'my-platform-suffix',",
        "}",
        "def set_options_transition_func(settings, attr):",
        "  return settings_under_test",
        "def passthrough_transition_func(settings, attr):",
        // All values in this test should be possible to copy within Starlark.
        "  ret = dict(settings)",
        // All values in this test should be possible to read within Starlark.
        // This does not mean that it is possible to set a string value for all settings,
        // e.g. //command_line_option:bazes should be set to a list of strings.
        "  for key, expected_value in settings_under_test.items():",
        "    if str(ret[key]) != expected_value:",
        "      fail('%s did not pass through, got %r expected %r' %",
        "        (key, str(ret[key]), expected_value))",
        "  ret['//command_line_option:bazes'] = ['ok']",
        "  return ret",
        "my_set_options_transition = transition(",
        "  implementation = set_options_transition_func,",
        "  inputs = [],",
        "  outputs = settings_under_test.keys())",
        "my_passthrough_transition = transition(",
        "  implementation = passthrough_transition_func,",
        "  inputs = settings_under_test.keys(),",
        "  outputs = ['//command_line_option:bazes'] + settings_under_test.keys())",
        "def impl(ctx): ",
        "  return MyInfo(attr_dep = ctx.attr.dep)",
        "my_set_options_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_set_options_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })",
        "my_passthrough_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_passthrough_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:my_rule.bzl', 'my_passthrough_rule', 'my_set_options_rule')",
        "my_set_options_rule(name = 'top', dep = ':test')",
        "my_passthrough_rule(name = 'test', dep = ':main')",
        "cc_binary(name = 'main', srcs = ['main.c'])");
  }

  @Test
  public void testCompilationModeReadableInStarlarkTransitions() throws Exception {
    writeReadAndPassthroughOptionsTestFiles();
    BazelMockAndroidSupport.setupNdk(mockToolsConfig); // cc_binary needs this

    ConfiguredTarget topTarget = getConfiguredTarget("//test/skylark:top");

    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> topDep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(topTarget).getValue("attr_dep");
    assertThat(topDep).hasSize(1);
    ConfiguredTarget testTarget = Iterables.getOnlyElement(topDep);
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> testDep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(testTarget).getValue("attr_dep");
    assertThat(testDep).hasSize(1);
    ConfiguredTarget mainTarget = Iterables.getOnlyElement(testDep);
    assertThat(getConfiguration(mainTarget).getOptions().get(DummyTestOptions.class).bazes)
        .containsExactly("ok");
  }

  @Test
  public void testUndeclaredOptionKey() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'k8'}",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "transition function returned undeclared output '//command_line_option:cpu'");
  }

  @Test
  public void testDeclaredOutputNotReturned() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "transition outputs [//command_line_option:host_cpu] were not "
            + "defined by transition function");
  }

  @Test
  public void testSettingsContainOnlyInputs() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    assertThat(getConfiguredTarget("//test/starlark:test")).isNotNull();
  }

  @Test
  public void testInvalidInputKey() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "invalid transition input 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  @Test
  public void testInvalidNativeOptionInput() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "transition inputs [//command_line_option:barp, //command_line_option:foop] "
            + "do not correspond to valid settings");
  }

  @Test
  public void testInvalidNativeOptionOutput() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "transition output '//command_line_option:foobarbaz' "
            + "does not correspond to a valid setting");
  }

  @Test
  public void testBannedNativeOptionOutput() throws Exception {
    // Just picked an arbitrary incompatible_ flag; however, could be any flag
    // besides incompatible_enable_cc_toolchain_resolution (and might not even need to be real).
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:incompatible_merge_genfiles_directory': True}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [], outputs = ['//command_line_option:incompatible_merge_genfiles_directory'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "Invalid transition output '//command_line_option:incompatible_merge_genfiles_directory'. "
            + "Cannot transition on --experimental_* or --incompatible_* options");
  }

  @Test
  public void testAllowIncompatibleEnableCcToolchainResolution() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:incompatible_enable_cc_toolchain_resolution': True}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = ['//command_line_option:incompatible_enable_cc_toolchain_resolution'], ",
        "  outputs = ['//command_line_option:incompatible_enable_cc_toolchain_resolution'])",
        "def impl(ctx): ",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "genrule(name = 'main1', outs = ['out.txt'], cmd = 'echo true > $@')");
    // Actually using cc_binary instead of genrule would require also mocking up
    // platforms-based toolchain resolution as well and this is tested elsewhere.

    ConfiguredTarget target = getConfiguredTarget("//test/starlark:test");
    @SuppressWarnings("unchecked")
    List<ConfiguredTarget> dep =
        (List<ConfiguredTarget>) getMyInfoFromTarget(target).getValue("dep");
    assertThat(dep).hasSize(1);

    assertThat(
            getConfiguration(Iterables.getOnlyElement(dep))
                .getFragment(CppConfiguration.class)
                .enableCcToolchainResolution())
        .isTrue();
  }

  @Test
  public void testInvalidOutputKey() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "invalid transition output 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  @Test
  public void testInvalidOptionValue() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent("Invalid value type for option 'cpu'");
  }

  @Test
  public void testDuplicateOutputs() throws Exception {
    writeAllowlistFile();

    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent("duplicate transition output '//command_line_option:cpu'");
  }

  @Test
  public void testInvalidNativeOptionOutput_analysisTest() throws Exception {
    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule_test')",
        "my_rule_test(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "transition output '//command_line_option:foobarbaz' "
            + "does not correspond to a valid setting");
  }

  @Test
  public void testInvalidOutputKey_analysisTest() throws Exception {
    scratch.file(
        "test/starlark/my_rule.bzl",
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
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule_test')",
        "my_rule_test(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "invalid transition output 'cpu'. If this is intended as a native option, "
            + "it must begin with //command_line_option:");
  }

  private void writeBuildSettingsBzl() throws Exception {
    scratch.file(
        "test/starlark/build_settings.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl, build_setting = config.int(flag=True))");
  }

  private void writeRulesWithAttrTransitionBzl() throws Exception {
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test/starlark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/starlark:the-answer': 42}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/starlark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  }",
        ")",
        "def _dep_rule_impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.attr.fact[BuildSettingInfo].value)]",
        "dep_rule_impl = rule(",
        "  implementation = _dep_rule_impl,",
        "  attrs = {",
        "    'fact': attr.label(default = '//test/starlark:the-answer'),",
        "  }",
        ")");
  }

  private CoreOptions getCoreOptions(ConfiguredTarget target) {
    return getConfiguration(target).getOptions().get(CoreOptions.class);
  }

  private ImmutableMap<Label, Object> getStarlarkOptions(ConfiguredTarget target) {
    return getConfiguration(target).getOptions().getStarlarkOptions();
  }

  private Object getStarlarkOption(ConfiguredTarget target, String absName) {
    return getStarlarkOptions(target).get(Label.parseCanonicalUnchecked(absName));
  }

  private String getTransitionDirectoryNameFragment(ConfiguredTarget target) {
    return getConfiguration(target).getTransitionDirectoryNameFragment();
  }

  @Test
  public void testTransitionOnBuildSetting_fromDefault() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer', build_setting_default = 0)");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>)
                getMyInfoFromTarget(getConfiguredTarget("//test/starlark:test")).getValue("dep"));
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//test/starlark:the-answer")))
        .isEqualTo(StarlarkInt.of(42));
  }

  @Test
  public void testTransitionOnBuildSetting_fromCommandLine() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer', build_setting_default = 0)");

    useConfiguration(ImmutableMap.of("//test/starlark:the-answer", 7));
    ConfiguredTarget test = getConfiguredTarget("//test/starlark:test");
    assertThat(
            getConfiguration(test)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//test/starlark:the-answer")))
        .isEqualTo(7);

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//test/starlark:the-answer")))
        .isEqualTo(StarlarkInt.of(42));
  }

  @Test
  public void testTransitionBackToStarlarkDefaultOK() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _transition_impl(settings, attr):",
        "  return {",
        "    '//test/starlark:the-answer': attr.answer_for_dep,",
        "    '//test/starlark:did-transition': 1,",
        "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/starlark:the-answer', '//test/starlark:did-transition']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    'answer_for_dep': attr.int(),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  }",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep1', answer_for_dep=0)",
        "my_rule(name = 'dep1', dep = ':dep2', answer_for_dep=42)",
        "my_rule(name = 'dep2', dep = ':dep3', answer_for_dep=0)",
        "my_rule(name = 'dep3')",
        "int_flag(name = 'the-answer', build_setting_default=0)",
        "int_flag(name = 'did-transition', build_setting_default=0)");
    useConfiguration(ImmutableMap.of(), "--cpu=FOO");

    ConfiguredTarget test = getConfiguredTarget("//test/starlark:test");

    // '//test/starlark:did-transition ensures ST-hash is 'turned on' since :test has no ST-hash
    //   and thus will trivially have a unique getTransitionDirectoryNameFragment

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep1 =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep2 =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(dep1).getValue("dep"));

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep3 =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(dep2).getValue("dep"));

    // These must be true
    assertThat(getTransitionDirectoryNameFragment(dep1))
        .isNotEqualTo(getTransitionDirectoryNameFragment(dep2));

    assertThat(getTransitionDirectoryNameFragment(dep2))
        .isNotEqualTo(getTransitionDirectoryNameFragment(dep3));

    // TODO(blaze-configurability-team): When "affected by starlark transition" is gone,
    //    will be equal and thus getTransitionDirectoryNameFragment can be equal.
    if (!getConfiguration(dep1).equals(getConfiguration(dep3))) {
      assertThat(getTransitionDirectoryNameFragment(dep1))
          .isNotEqualTo(getTransitionDirectoryNameFragment(dep3));
    }
  }

  @Test
  public void testTransitionOnBuildSetting_onlyTransitionsAffectsDirectory() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer', build_setting_default = 0)",
        "int_flag(name = 'cmd-line-option', build_setting_default = 0)");

    useConfiguration(ImmutableMap.of("//test/starlark:cmd-line-option", 100), "--cpu=FOO");

    ConfiguredTarget test = getConfiguredTarget("//test/starlark:test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    // Assert starlark option set via transition.
    assertThat(getStarlarkOption(dep, "//test/starlark:the-answer")).isEqualTo(StarlarkInt.of(42));

    // Assert starlark option set via command line.
    assertThat(getStarlarkOption(dep, "//test/starlark:cmd-line-option")).isEqualTo(100);

    // Assert native option set via command line.
    assertThat(getCoreOptions(dep).cpu).isEqualTo("FOO");

    // Assert that transitionDirectoryNameFragment is only affected by options
    // set via transitions. Not by native or starlark options set via command line,
    // never touched by any transition.
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test/starlark:the-answer=42")));
  }

  @Test
  public void testTransitionOnCompilationMode_hasNoHash() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _transition_impl(settings, attr):",
        "  return {",
        "    '//command_line_option:compilation_mode': attr.cmode_for_dep,",
        "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:compilation_mode']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    'cmode_for_dep': attr.string(),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  }",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep1', cmode_for_dep='opt')",
        "my_rule(name = 'dep1', dep = ':dep2', cmode_for_dep='fastbuild')",
        "my_rule(name = 'dep2')");
    useConfiguration(ImmutableMap.of(), "--compilation_mode=fastbuild");

    ConfiguredTarget test = getConfiguredTarget("//test/starlark:test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep1 =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep2 =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(dep1).getValue("dep"));

    // Assert transitionDirectoryNameFragment is empty for all configurations
    assertThat(getTransitionDirectoryNameFragment(test)).isEmpty();
    assertThat(getTransitionDirectoryNameFragment(dep1)).isEmpty();
    assertThat(getTransitionDirectoryNameFragment(dep2)).isEmpty();

    // test and dep1 should have different configurations b/c compilation_mode changed
    assertThat(getConfiguration(test)).isNotEqualTo(getConfiguration(dep1));

    // test and dep2 should have identical configurations
    assertThat(getConfiguration(test)).isEqualTo(getConfiguration(dep2));
  }

  private void writeFilesWithMultipleNativeOptionTransitions() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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
  }

  @Test
  public void testOutputDirHash_multipleNativeOptionTransitions_legacyNaming() throws Exception {
    writeFilesWithMultipleNativeOptionTransitions();

    useConfiguration("--experimental_output_directory_naming_scheme=legacy");
    ConfiguredTarget test = getConfiguredTarget("//test");

    List<String> affectedOptions = getCoreOptions(test).affectedByStarlarkTransition;

    assertThat(affectedOptions).containsExactly("//command_line_option:foo");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    affectedOptions = getCoreOptions(dep).affectedByStarlarkTransition;

    assertThat(affectedOptions)
        .containsExactly("//command_line_option:foo", "//command_line_option:bar");

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//command_line_option:foo=foosball")));

    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball", "//command_line_option:foo=foosball")));
  }

  @Test
  public void testOutputDirHash_multipleNativeOptionTransitions_diffNaming() throws Exception {
    writeFilesWithMultipleNativeOptionTransitions();

    useConfiguration("--experimental_output_directory_naming_scheme=diff_against_baseline");
    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//command_line_option:foo=foosball")));

    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball", "//command_line_option:foo=foosball")));
  }

  @Test
  public void testOutputDirHash_onlyExec_diffDynamic() throws Exception {
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = 'exec'), ",
        "  })",
        "def _basic_impl(ctx):",
        "  return []",
        "simple = rule(_basic_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple')",
        "my_rule(name = 'test', dep = ':dep')",
        "simple(name = 'dep')");

    useConfiguration("--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    ConfiguredTarget test = getConfiguredTarget("//test");

    ConfiguredTarget dep = (ConfiguredTarget) getMyInfoFromTarget(test).getValue("dep");

    assertThat(getTransitionDirectoryNameFragment(test)).isEmpty();

    // Until platforms is EXPLICIT_IN_OUTPUT_PATH, it will change here as well.
    // But, nothing else should be different.
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:platforms="
                        + getConfiguration(dep)
                            .getOptions()
                            .get(PlatformOptions.class)
                            .platforms)));
  }

  @Test
  public void testOutputDirHash_starlarkRevertedByExec_diffDynamic() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _some_impl(settings, attr):",
        "  return {'//command_line_option:set_by_exec': 'at_target'}",
        "some_transition = transition(implementation = _some_impl, inputs = [],",
        "  outputs = ['//command_line_option:set_by_exec'])");
    scratch.file(
        "test/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test:transitions.bzl', 'some_transition')",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = some_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = 'exec'), ",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    useConfiguration("--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    ConfiguredTarget test = getConfiguredTarget("//test");

    ConfiguredTarget dep = (ConfiguredTarget) getMyInfoFromTarget(test).getValue("dep");

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//command_line_option:set_by_exec=at_target")));

    // Until platforms is EXPLICIT_IN_OUTPUT_PATH, it will change here as well.
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:platforms="
                        + getConfiguration(dep)
                            .getOptions()
                            .get(PlatformOptions.class)
                            .platforms)));
  }

  // Test that a no-op starlark transition to an already starlark transitioned configuration
  // results in the same configuration.
  @Test
  public void testOutputDirHash_noop_changeToSameState() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(getTransitionDirectoryNameFragment(dep));
  }

  @Test
  public void testOutputDirHash_noop_emptyReturns() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(getTransitionDirectoryNameFragment(dep));
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
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    assertThat(getTransitionDirectoryNameFragment(dep)).isNotEmpty();
  }

  /** See comment above {@link FunctionTransitionUtil#updateOutputDirectoryNameFragment} */
  // TODO(bazel-team): This can be optimized. Make these the same configuration.
  @Test
  public void testOutputDirHash_starlarkOption_differentBoolRepresentationsNotEquals()
      throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:foo=1")));
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:foo=true")));
  }

  @Test
  public void testOutputDirHash_nativeOption_differentBoolRepresentationsEquals() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(getTransitionDirectoryNameFragment(dep));
  }

  private void writeFilesWithMultipleStarlarkTransitions() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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
  }

  @Test
  public void testOutputDirHash_multipleStarlarkTransitions_legacyNaming() throws Exception {
    writeFilesWithMultipleStarlarkTransitions();

    useConfiguration("--experimental_output_directory_naming_scheme=legacy");
    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    List<String> affectedOptions =
        getConfiguration(dep).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptions).containsExactly("//test:bar", "//test:foo");
    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:foo=foosball")));
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:bar=barsball", "//test:foo=foosball")));
  }

  @Test
  public void testOutputDirHash_multipleStarlarkTransitions_diffNaming() throws Exception {
    writeFilesWithMultipleStarlarkTransitions();

    useConfiguration("--experimental_output_directory_naming_scheme=diff_against_baseline");
    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));

    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:foo=foosball")));
    assertThat(getTransitionDirectoryNameFragment(dep))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//test:bar=barsball", "//test:foo=foosball")));
  }

  private void writeFilesWithMultipleMixedTransitions() throws Exception {
    writeAllowlistFile();
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
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })",
        "def _impl_b(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule_b = rule(",
        "  implementation = _impl_b,",
        "  cfg = bar_transition,", // transition #3
        "  attrs = {",
        "    'dep': attr.label(cfg = xan_transition), ", // transition #4
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
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
  }

  // This test is massive but mostly exists to ensure that all the parts are working together
  // properly amidst multiple complicated transitions.
  @Test
  public void testOutputDirHash_multipleMixedTransitions_legacyNaming() throws Exception {
    writeFilesWithMultipleMixedTransitions();

    // test:top (foo_transition)
    useConfiguration("--experimental_output_directory_naming_scheme=legacy");
    ConfiguredTarget top = getConfiguredTarget("//test:top");

    List<String> affectedOptionsTop =
        getConfiguration(top).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsTop).containsExactly("//command_line_option:foo");
    assertThat(getConfiguration(top).getOptions().getStarlarkOptions()).isEmpty();
    assertThat(getTransitionDirectoryNameFragment(top))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//command_line_option:foo=foosball")));

    // test:middle (foo_transition, zee_transition, bar_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget middle =
        Iterables.getOnlyElement((List<ConfiguredTarget>) getMyInfoFromTarget(top).getValue("dep"));

    List<String> affectedOptionsMiddle =
        getConfiguration(middle).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsMiddle)
        .containsExactly("//command_line_option:foo", "//command_line_option:bar", "//test:zee");

    assertThat(getConfiguration(middle).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:zee"), "zeesball"));

    assertThat(getTransitionDirectoryNameFragment(middle))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball",
                    "//command_line_option:foo=foosball",
                    "//test:zee=zeesball")));

    // test:bottom (foo_transition, zee_transition, bar_transition, xan_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget bottom =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(middle).getValue("dep"));

    List<String> affectedOptionsBottom =
        getConfiguration(bottom).getOptions().get(CoreOptions.class).affectedByStarlarkTransition;

    assertThat(affectedOptionsBottom)
        .containsExactly(
            "//command_line_option:foo", "//command_line_option:bar", "//test:xan", "//test:zee");

    assertThat(getConfiguration(bottom).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:zee"), "zeesball"),
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:xan"), "xansball"));
    assertThat(getTransitionDirectoryNameFragment(bottom))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball", "//command_line_option:foo=foosball",
                    "//test:xan=xansball", "//test:zee=zeesball")));
  }

  @Test
  public void testOutputDirHash_multipleMixedTransitions_diffNaming() throws Exception {
    writeFilesWithMultipleMixedTransitions();

    // test:top (foo_transition)
    useConfiguration("--experimental_output_directory_naming_scheme=diff_against_baseline");
    ConfiguredTarget top = getConfiguredTarget("//test:top");

    assertThat(getConfiguration(top).getOptions().getStarlarkOptions()).isEmpty();
    assertThat(getTransitionDirectoryNameFragment(top))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of("//command_line_option:foo=foosball")));

    // test:middle (foo_transition, zee_transition, bar_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget middle =
        Iterables.getOnlyElement((List<ConfiguredTarget>) getMyInfoFromTarget(top).getValue("dep"));

    assertThat(getConfiguration(middle).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:zee"), "zeesball"));

    assertThat(getTransitionDirectoryNameFragment(middle))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball",
                    "//command_line_option:foo=foosball",
                    "//test:zee=zeesball")));

    // test:bottom (foo_transition, zee_transition, bar_transition, xan_transition)
    @SuppressWarnings("unchecked")
    ConfiguredTarget bottom =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(middle).getValue("dep"));

    assertThat(getConfiguration(bottom).getOptions().getStarlarkOptions().entrySet())
        .containsExactly(
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:zee"), "zeesball"),
            Maps.immutableEntry(Label.parseCanonicalUnchecked("//test:xan"), "xansball"));
    assertThat(getTransitionDirectoryNameFragment(bottom))
        .isEqualTo(
            BuildConfigurationFunction.transitionDirectoryNameFragment(
                ImmutableList.of(
                    "//command_line_option:bar=barsball", "//command_line_option:foo=foosball",
                    "//test:xan=xansball", "//test:zee=zeesball")));
  }

  @Test
  public void testTransitionOnBuildSetting_badValue() throws Exception {
    writeAllowlistFile();
    writeBuildSettingsBzl();
    scratch.file(
        "test/starlark/rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "load('//test/starlark:build_settings.bzl', 'BuildSettingInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//test/starlark:the-answer': 'What do you get if you multiply six by nine?'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/starlark:the-answer']",
        ")",
        "def _rule_impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  }",
        ")",
        "def _dep_rule_impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.attr.fact[BuildSettingInfo].value)]",
        "dep_rule_impl = rule(",
        "  implementation = _dep_rule_impl,",
        "  attrs = {",
        "    'fact': attr.label(default = '//test/starlark:the-answer'),",
        "  }",
        ")");

    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(",
        "  name = 'the-answer',",
        "  build_setting_default = 0,",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "expected value of type 'int' for //test/starlark:the-answer, "
            + "but got \"What do you get if you multiply six by nine?\" (string)");
  }

  @Test
  public void testTransitionOnBuildSetting_noSuchTarget() throws Exception {
    writeAllowlistFile();
    writeRulesWithAttrTransitionBzl();
    // Still need to write this file in order not to rewrite rules.bzl file (has loads from this
    // file)
    writeBuildSettingsBzl();
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "no such target '//test/starlark:the-answer': target "
            + "'the-answer' not declared in package");
  }

  @Test
  public void testTransitionOnBuildSetting_notABuildSetting() throws Exception {
    writeAllowlistFile();
    writeRulesWithAttrTransitionBzl();
    scratch.file(
        "test/starlark/build_settings.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl)");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'my_rule')",
        "load('//test/starlark:build_settings.bzl', 'int_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "int_flag(name = 'the-answer')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/starlark:test");
    assertContainsEvent(
        "attempting to transition on '//test/starlark:the-answer' which "
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
    writeAllowlistFile();
    // This setup creates an int_flag_reading_rule whose output is the value of an int_flag (which
    // guarantees actions in configurations with different Starlark flag values are different). It
    // then makes this a genrule exec tool (so it applies after an exec transition). And finally
    // creates a build_setting_changing_rule that changes the int_flag's value and depends on the
    // genrule. So building the genrule at both the top-level and under the
    // build_setting_changing_rule triggers the test scenario.
    scratch.file(
        "test/starlark/rules.bzl",
        "BuildSettingInfo = provider(fields = ['value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(value = ctx.build_setting_value)]",
        "int_flag = rule(implementation = _impl, build_setting = config.int())",
        "def _transition_impl(settings, attr):",
        "  return {'//test/starlark:the-answer': 42}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test/starlark:the-answer'])",
        "def _int_impl(ctx):",
        "  value = ctx.attr._int_dep[BuildSettingInfo].value",
        "  ctx.actions.write(ctx.outputs.out, str(value))",
        "int_flag_reading_rule = rule(",
        "  implementation = _int_impl,",
        "  attrs = {",
        "    '_int_dep': attr.label(default = '//test/starlark:the-answer'),",
        "    'out': attr.output()",
        "  })",
        "def _rule_impl(ctx):",
        "  pass",
        "build_setting_changing_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition, allow_single_file = True),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  })");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'build_setting_changing_rule', 'int_flag',",
        "  'int_flag_reading_rule')",
        "int_flag(",
        "    name = 'the-answer',",
        "    build_setting_default = 0)",
        "genrule(",
        "    name = 'with_exec_tool',",
        "    srcs = [],",
        "    outs = ['with_exec_tool.out'],",
        "    cmd = 'echo hi > $@',",
        "    tools = [':int_reader'])",
        "int_flag_reading_rule(",
        "    name = 'int_reader',",
        "    out = 'int_reader.out')",
        "build_setting_changing_rule(",
        "    name = 'transitioner',",
        "    dep = ':with_exec_tool')");
    // Note: calling getConfiguredTarget for each target doesn't activate conflict detection.
    update(
        ImmutableList.of("//test/starlark:transitioner", "//test/starlark:with_exec_tool.out"),
        /*keepGoing=*/ false,
        LOADING_PHASE_THREADS,
        /*doAnalysis=*/ true,
        new EventBus());
    assertNoEvents();
  }

  @Test
  public void starlarkSplitTransitionRequiredFragments() throws Exception {
    // All Starlark rule transitions are patch transitions, while all Starlark attribute transitions
    // are split transitions.
    writeAllowlistFile();
    scratch.file(
        "test/my_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def transition_func(settings, attr):",
        "  return [",
        "    {'//command_line_option:copt': []}", // --copt is a C++ option.
        "  ]",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:copt'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "cc_library(name = 'dep', srcs = ['dep.c'])");

    ConfiguredTargetAndData ct = getConfiguredTargetAndData("//test");
    assertNoEvents();
    ConfiguredAttributeMapper attributes = ct.getAttributeMapperForTesting();
    ConfigurationTransition attrTransition =
        attributes
            .getAttributeDefinition("dep")
            .getTransitionFactory()
            .create(AttributeTransitionData.builder().attributes(attributes).build());
    RequiredConfigFragmentsProvider.Builder requiredFragments =
        RequiredConfigFragmentsProvider.builder();
    attrTransition.addRequiredFragments(
        requiredFragments, ct.getConfiguration().getBuildOptionDetails());
    assertThat(requiredFragments.build().getOptionsClasses()).containsExactly(CppOptions.class);
  }

  /**
   * @param directRead if set to true, reads the output value directly from the input dict, else
   *     just passes in the same value as a string
   */
  private void testNoOpTransitionLeavesSameConfig_native(boolean directRead) throws Exception {
    writeAllowlistFile();

    String outputValue = directRead ? "settings['//command_line_option:foo']" : "'frisbee'";
    String inputs = directRead ? "['//command_line_option:foo']" : "[]";

    scratch.file(
        "test/defs.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _transition_impl(settings, attr):",
        "  return {'//command_line_option:foo': " + outputValue + "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = " + inputs + ",",
        "  outputs = ['//command_line_option:foo'],",
        ")",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')");

    useConfiguration("--foo=frisbee");
    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));
    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(getTransitionDirectoryNameFragment(dep));
  }

  @Test
  public void testNoOpTransitionLeavesSameConfig_native_directRead() throws Exception {
    testNoOpTransitionLeavesSameConfig_native(true);
  }

  @Test
  public void testNoOpTransitionLeavesSameConfig_native_setToSame() throws Exception {
    testNoOpTransitionLeavesSameConfig_native(false);
  }

  /**
   * @param directRead if set to true, reads the output value directly from the input dict, else
   *     just passes in the same value as a string
   * @param setToDefault if set to true, value getting passed through the transition is the default
   *     value of the build settings. Internally we don't keep default values in the build settings
   *     map inside {@link BuildOptions} so it's nice to test this separately.
   */
  private void testNoOpTransitionLeavesSameConfig_starlark(boolean directRead, boolean setToDefault)
      throws Exception {
    writeAllowlistFile();

    String outputValue = directRead ? "settings['//test:flag']" : "'frisbee'";
    String inputs = directRead ? "['//test:flag']" : "[]";
    String buildSettingsDefault = setToDefault ? "frisbee" : "waterpolo";

    scratch.file(
        "test/defs.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _flag_impl(ctx):",
        "  return []",
        "my_flag = rule(",
        "  implementation = _flag_impl,",
        "  build_setting = config.string(flag = True)",
        ")",
        "def _transition_impl(settings, attr):",
        "  return {'//test:flag': " + outputValue + "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = " + inputs + ",",
        "  outputs = ['//test:flag'],",
        ")",
        "def _impl(ctx):",
        "  return MyInfo(dep = ctx.attr.dep)",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "      default = '//tools/allowlists/function_transition_allowlist'),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'my_rule', 'my_flag')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')",
        "my_flag(name = 'flag', build_setting_default = '" + buildSettingsDefault + "')");

    // TODO(blaze-configurability-team): There is a bug in BuildViewTestCase that it does not audit
    //   these Starlark options at all (i.e. check they are the right type or that values at default
    //   are unset/set to null rather than explicitly set to the default.
    if (!buildSettingsDefault.equals("frisbee")) {
      useConfiguration(ImmutableMap.of("//test:flag", "frisbee"));
    }
    ConfiguredTarget test = getConfiguredTarget("//test");

    @SuppressWarnings("unchecked")
    ConfiguredTarget dep =
        Iterables.getOnlyElement(
            (List<ConfiguredTarget>) getMyInfoFromTarget(test).getValue("dep"));
    assertThat(getTransitionDirectoryNameFragment(test))
        .isEqualTo(getTransitionDirectoryNameFragment(dep));
  }

  @Test
  public void testNoOpTransitionLeavesSameConfig_starlark_directRead() throws Exception {
    testNoOpTransitionLeavesSameConfig_starlark(true, false);
  }

  @Test
  public void testNoOpTransitionLeavesSameConfig_starlark_setToSame() throws Exception {
    testNoOpTransitionLeavesSameConfig_starlark(false, false);
  }

  @Test
  public void testNoOpTransitionLeavesSameConfig_starlark_setToDefault() throws Exception {
    testNoOpTransitionLeavesSameConfig_starlark(false, true);
  }

  @Test
  public void testOptionConversionDynamicMode() {
    // TODO(waltl): check that dynamic_mode is parsed properly.
  }

  @Test
  public void testOptionConversionCrosstoolTop() throws Exception {
    // TODO(waltl): check that crosstool_top is parsed properly.
  }

  /**
   * Changing --cpu implicitly changes the target platform. Test that the old value of --platforms
   * gets cleared out (platform mappings can then kick in to set --platforms correctly).
   */
  @Test
  public void testImplicitPlatformsChange() throws Exception {
    writeAllowlistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file("platforms/BUILD", "platform(name = 'my_platform', constraint_values = [])");
    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:cpu': 'armeabi-v7a'}",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [], outputs = ['//command_line_option:cpu'])",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    useConfiguration("--platforms=//platforms:my_platform");
    ConfiguredTarget dep =
        getDirectPrerequisite(getConfiguredTarget("//test/starlark:test"), "//test/starlark:main1");
    // When --platforms is empty and no platform mapping triggers, PlatformMappingValue sets
    // --platforms to PlatformOptions.computeTargetPlatform(), which defaults to the host.
    assertThat(getConfiguration(dep).getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(
            Label.parseCanonicalUnchecked(
                TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT + ":host"));
  }

  @Test
  public void testExplicitPlatformsChange() throws Exception {
    writeAllowlistFile();
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "platforms/BUILD",
        "platform(name = 'my_platform', constraint_values = [])",
        "platform(name = 'my_other_platform', constraint_values = [])");
    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {",
        "    '//command_line_option:cpu': 'armeabi-v7a',",
        "    '//command_line_option:platforms': ['//platforms:my_other_platform']",
        "  }",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [],",
        "  outputs = [",
        "    '//command_line_option:cpu',",
        "    '//command_line_option:platforms'",
        "  ]",
        ")",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    useConfiguration("--platforms=//platforms:my_platform");
    ConfiguredTarget dep =
        getDirectPrerequisite(getConfiguredTarget("//test/starlark:test"), "//test/starlark:main1");
    assertThat(getConfiguration(dep).getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseCanonicalUnchecked("//platforms:my_other_platform"));
  }

  /* If the transition doesn't change --cpu, it doesn't constitute a platform change. */
  @Test
  public void testNoPlatformChange() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "platforms/BUILD",
        "platform(name = 'my_platform',",
        "    parents = ['" + TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT + ":host'],",
        "    constraint_values = [],",
        ")");
    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        "  return {",
        "    '//command_line_option:foo': 'blah',",
        "  }",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [],",
        "  outputs = [",
        "    '//command_line_option:foo',",
        "  ]",
        ")",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    useConfiguration("--platforms=//platforms:my_platform");
    ConfiguredTarget dep =
        getDirectPrerequisite(getConfiguredTarget("//test/starlark:test"), "//test/starlark:main1");
    assertThat(getConfiguration(dep).getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseCanonicalUnchecked("//platforms:my_platform"));
  }

  /*
   * If the transition claims to change --cpu but doesn't, it doesn't constitute a platform change
   * and also doesn't affect any other options (such as affectedByStarlarkTransition).
   */
  @Test
  public void testCpuNoOpChangeIsFullyNoOp() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "platforms/BUILD",
        "platform(name = 'my_platform',",
        "    parents = ['" + TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT + ":host'],",
        "    constraint_values = [],",
        ")");
    scratch.file(
        "test/starlark/my_rule.bzl",
        "def transition_func(settings, attr):",
        // Leave --cpu unchanged, but still trigger the full transition logic that would be
        // bypassed by returning {}.
        "  return settings",
        "my_transition = transition(implementation = transition_func,",
        "  inputs = [",
        "    '//command_line_option:cpu',",
        "  ],",
        "  outputs = [",
        "    '//command_line_option:cpu',",
        "  ]",
        ")",
        "def impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':main1')",
        "cc_binary(name = 'main1', srcs = ['main1.c'])");

    ConfiguredTarget topLevel = getConfiguredTarget("//test/starlark:test");
    ConfiguredTarget dep =
        getDirectPrerequisite(getConfiguredTarget("//test/starlark:test"), "//test/starlark:main1");
    assertThat(getConfiguration(dep).getOptions())
        .isEqualTo(getConfiguration(topLevel).getOptions());
  }

  @Test
  public void testEffectiveNoopTransitionTrimsInputBuildSettings() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "def _string_impl(ctx):",
        "    return []",
        "string_flag = rule(",
        "    implementation = _string_impl,",
        "    build_setting = config.string(flag = True),",
        ")",
        "def _no_op_transition_impl(settings, attr):",
        "    return {",
        "        '//test/starlark:input_and_output': settings['//test/starlark:input_and_output'],",
        "        '//test/starlark:output_only': 'output_only_default',",
        "    }",
        "_no_op_transition = transition(",
        "    implementation = _no_op_transition_impl,",
        "    inputs = [",
        "        '//test/starlark:input_only',",
        "        '//test/starlark:input_and_output',",
        "    ],",
        "    outputs = [",
        "        '//test/starlark:input_and_output',",
        "        '//test/starlark:output_only',",
        "    ],",
        ")",
        "def _apply_transition_impl(ctx):",
        "    ctx.actions.symlink(",
        "        output = ctx.outputs.out,",
        "        target_file = ctx.file.target,",
        "    )",
        "    return [DefaultInfo(executable = ctx.outputs.out)]",
        "apply_transition = rule(",
        "    implementation = _apply_transition_impl,",
        "    attrs = {",
        "        'target': attr.label(",
        "            cfg = _no_op_transition,",
        "            allow_single_file = True,",
        "        ),",
        "        'out': attr.output(),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist',",
        "        ),",
        "    },",
        "    executable = False,",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'apply_transition', 'string_flag')",
        "",
        "string_flag(",
        "    name = 'input_only',",
        "    build_setting_default = 'input_only_default',",
        ")",
        "string_flag(",
        "    name = 'input_and_output',",
        "    build_setting_default = 'input_and_output_default',",
        ")",
        "string_flag(",
        "    name = 'output_only',",
        "    build_setting_default = 'output_only_default',",
        ")",
        "cc_binary(",
        "    name = 'main',",
        "    srcs = ['main.cc'],",
        ")",
        "apply_transition(",
        "    name = 'transitioned_main',",
        "    target = ':main',",
        "    out = 'main_out',",
        ")");

    update(
        ImmutableList.of("//test/starlark:main", "//test/starlark:transitioned_main"),
        /*keepGoing=*/ false,
        LOADING_PHASE_THREADS,
        /*doAnalysis=*/ true,
        new EventBus());
    assertNoEvents();
  }

  @Test
  public void testExplicitNoopTransitionTrimsInputBuildSettings() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "def _string_impl(ctx):",
        "    return []",
        "string_flag = rule(",
        "    implementation = _string_impl,",
        "    build_setting = config.string(flag = True),",
        ")",
        "def _no_op_transition_impl(settings, attr):",
        "    return {}",
        "_no_op_transition = transition(",
        "    implementation = _no_op_transition_impl,",
        "    inputs = [",
        "        '//test/starlark:input_only',",
        "        '//test/starlark:input_and_output',",
        "    ],",
        "    outputs = [",
        "        '//test/starlark:input_and_output',",
        "        '//test/starlark:output_only',",
        "    ],",
        ")",
        "def _apply_transition_impl(ctx):",
        "    ctx.actions.symlink(",
        "        output = ctx.outputs.out,",
        "        target_file = ctx.file.target,",
        "    )",
        "    return [DefaultInfo(executable = ctx.outputs.out)]",
        "apply_transition = rule(",
        "    implementation = _apply_transition_impl,",
        "    attrs = {",
        "        'target': attr.label(",
        "            cfg = _no_op_transition,",
        "            allow_single_file = True,",
        "        ),",
        "        'out': attr.output(),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist',",
        "        ),",
        "    },",
        "    executable = False,",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'apply_transition', 'string_flag')",
        "",
        "string_flag(",
        "    name = 'input_only',",
        "    build_setting_default = 'input_only_default',",
        ")",
        "string_flag(",
        "    name = 'input_and_output',",
        "    build_setting_default = 'input_and_output_default',",
        ")",
        "string_flag(",
        "    name = 'output_only',",
        "    build_setting_default = 'output_only_default',",
        ")",
        "cc_binary(",
        "    name = 'main',",
        "    srcs = ['main.cc'],",
        ")",
        "apply_transition(",
        "    name = 'transitioned_main',",
        "    target = ':main',",
        "    out = 'main_out',",
        ")");

    update(
        ImmutableList.of("//test/starlark:main", "//test/starlark:transitioned_main"),
        /*keepGoing=*/ false,
        LOADING_PHASE_THREADS,
        /*doAnalysis=*/ true,
        new EventBus());
    assertNoEvents();
  }

  @Test
  public void testTransitionPreservesAllowMultipleDefault() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "P = provider(fields = ['value'])",
        "def _s_impl(ctx):",
        "    return [P(value = ctx.build_setting_value)]",
        "def _t_impl(settings, attr):",
        "    if 'foo' in settings['//test/starlark:a']:",
        "        return {'//test/starlark:b': ['bar']}",
        "    else:",
        "        return {'//test/starlark:b': ['baz']}",
        "def _r_impl(ctx):",
        "    pass",
        "s = rule(",
        "    implementation = _s_impl,",
        "    build_setting = config.string(allow_multiple = True, flag = True),",
        ")",
        "t = transition(",
        "    implementation = _t_impl,",
        "    inputs = ['//test/starlark:a'],",
        "    outputs = ['//test/starlark:b'],",
        ")",
        "r = rule(",
        "    implementation = _r_impl,",
        "    attrs = {",
        "        'setting': attr.label(cfg = t),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist',",
        "        ),",
        "    },",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load(':rules.bzl', 's', 'r')",
        "s(",
        "  name = 'a',",
        "  build_setting_default = '',",
        ")",
        "s(",
        "  name = 'b',",
        "  build_setting_default = '',",
        ")",
        "r(",
        "  name = 'c',",
        "  setting = ':b',",
        ")");
    update(
        ImmutableList.of("//test/starlark:c"),
        /*keepGoing=*/ false,
        LOADING_PHASE_THREADS,
        /*doAnalysis=*/ true,
        new EventBus());
    assertNoEvents();
  }

  @Test
  public void testTransitionPreservesNonDefaultInputOnlySetting() throws Exception {
    writeAllowlistFile();
    scratch.file(
        "test/starlark/rules.bzl",
        "def _string_impl(ctx):",
        "    return []",
        "string_flag = rule(",
        "    implementation = _string_impl,",
        "    build_setting = config.string(flag = True),",
        ")",
        "def _transition_impl(settings, attr):",
        "    return {",
        "        '//test/starlark:output_only': settings['//test/starlark:input_only'],",
        "    }",
        "_transition = transition(",
        "    implementation = _transition_impl,",
        "    inputs = [",
        "        '//test/starlark:input_only',",
        "    ],",
        "    outputs = [",
        "        '//test/starlark:output_only',",
        "    ],",
        ")",
        "def _apply_transition_impl(ctx):",
        "    ctx.actions.symlink(",
        "        output = ctx.outputs.out,",
        "        target_file = ctx.file.target,",
        "    )",
        "    return [DefaultInfo(executable = ctx.outputs.out)]",
        "apply_transition = rule(",
        "    implementation = _apply_transition_impl,",
        "    attrs = {",
        "        'target': attr.label(",
        "            cfg = _transition,",
        "            allow_single_file = True,",
        "        ),",
        "        'out': attr.output(),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist',",
        "        ),",
        "    },",
        "    executable = False,",
        ")");
    scratch.file(
        "test/starlark/BUILD",
        "load('//test/starlark:rules.bzl', 'apply_transition', 'string_flag')",
        "",
        "string_flag(",
        "    name = 'input_only',",
        "    build_setting_default = 'input_only_default',",
        ")",
        "string_flag(",
        "    name = 'output_only',",
        "    build_setting_default = 'output_only_default',",
        ")",
        "cc_binary(",
        "    name = 'main',",
        "    srcs = ['main.cc'],",
        ")",
        "apply_transition(",
        "    name = 'transitioned_main',",
        "    target = ':main',",
        "    out = 'main_out',",
        ")");

    useConfiguration(ImmutableMap.of("//test/starlark:input_only", "not_the_default"));
    Label inputOnlySetting = Label.parseCanonicalUnchecked("//test/starlark:input_only");
    ConfiguredTarget transitionedDep =
        getDirectPrerequisite(
            getConfiguredTarget("//test/starlark:transitioned_main"), "//test/starlark:main");
    assertThat(
            getConfiguration(transitionedDep)
                .getOptions()
                .getStarlarkOptions()
                .get(inputOnlySetting))
        .isEqualTo("not_the_default");
  }
}
