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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for StarlarkRuleTransitionProvider. */
@RunWith(JUnit4.class)
public class StarlarkRuleTransitionProviderTest extends BuildViewTestCase {

  /**
   * A fragment containing flags that exhibit different flag behaviors for easy testing purposes.
   */
  private static class DummyTestFragment extends Fragment {}

  /** Flags that exhibit an variety of flag behaviors. */
  public static class DummyTestOptions extends FragmentOptions {
    @Option(
        name = "nullable_option",
        converter = EmptyToNullLabelConverter.class,
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "An option that is sometimes set to null.")
    public Label nullable;

    @Option(
        name = "foo",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "A regular string-typed option")
    public String foo;

    @Option(
        name = "bar",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "A regular string-typed option")
    public String bar;

    @Option(
        name = "bool",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        help = "A regular bool-typed option")
    public boolean bool;
  }

  /** Loads a new {link @DummyTestFragment} instance. */
  protected static class DummyTestLoader implements ConfigurationFragmentFactory {

    @Override
    public Fragment create(BuildOptions buildOptions) throws InvalidConfigurationException {
      return new DummyTestFragment();
    }

    @Override
    public Class<? extends Fragment> creates() {
      return DummyTestFragment.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of(DummyTestOptions.class);
    }
  }

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(new DummyTestLoader());
    return builder.build();
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

  @Test
  public void testBadReturnTypeFromTransition() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return 'cpu=k8'",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Transition function must return a dictionary or list of dictionaries.");
  }

  @Test
  public void testOutputOnlyTransition() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  @Test
  public void testInputAndOutputTransition() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ",
        "    [settings['//command_line_option:test_arg'][0]+'->post-transition']}",
        "my_transition = transition(",
        "  implementation = _impl,",
        "  inputs = ['//command_line_option:test_arg'],",
        "  outputs = ['//command_line_option:test_arg'],",
        ")");

    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");

    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("pre-transition->post-transition");
  }

  @Test
  public void testBuildSettingCannotTransition() throws Exception {
    setStarlarkSemanticsOptions("--experimental_build_setting_api=true");
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  build_setting = config.string(),",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "Build setting rules cannot use the `cfg` param to apply transitions to themselves");
  }

  @Test
  public void testBadCfgInput() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = 'my_transition',",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "`cfg` must be set to a transition object initialized by the transition() function.");
  }

  @Test
  public void testMultipleReturnConfigs() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return [",
        "      {'//command_line_option:test_arg': ['split_one']},",
        "      {'//command_line_option:test_arg': ['split_two']},",
        "  ]",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "Rule transition only allowed to return a single transitioned configuration.");
  }

  @Test
  public void testCanDoBadStuffWithParameterizedTransitionsAndSelects() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  if (attr.my_configurable_attr):",
        "    return {'//command_line_option:test_arg': ['true']}",
        "  else:",
        "    return {'//command_line_option:test_arg': ['false']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    'my_configurable_attr': attr.bool(default = False),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'test',",
        "  my_configurable_attr = select({",
        "    '//conditions:default': False,",
        "    ':true-config': True,",
        "  })",
        ")",
        "config_setting(",
        "  name = 'true-config',",
        "  values = {'test_arg': 'true'},",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "No attribute 'my_configurable_attr'. "
            + "Either this attribute does not exist for this rule or is set by a select. "
            + "Starlark rule transitions currently cannot read attributes behind selects.");
  }

  @Test
  public void testLabelTypedAttrReturnsLabelNotDep() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  if attr.dict_attr[Label('//test:key')] == 'value':",
        "    return {'//command_line_option:test_arg': ['post-transition']}",
        "  else:",
        "    return {'//command_line_option:test_arg': ['uh-oh']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    'dict_attr': attr.label_keyed_string_dict(),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")",
        "simple_rule = rule(_impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "my_rule(",
        "  name = 'test',",
        "  dict_attr = {':key': 'value'},",
        ")",
        "simple_rule(name = 'key')");

    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  private static final String CUTE_ANIMAL_DEFAULT =
      "cows produce more milk when they listen to soothing music";

  private void writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _rule_impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");

    scratch.file(
        "test/build_settings.bzl",
        "def _impl(ctx):",
        "  return []",
        "string_flag = rule(implementation = _impl, build_setting = config.string(flag=True))");

    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "string_flag(",
        "  name = 'cute-animal-fact',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");
  }

  @Test
  public void testCannotTransitionOnBuildSettingWithoutFlag() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=false", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("transitions on Starlark-defined build settings is experimental");
  }

  @Test
  public void testTransitionOnBuildSetting_fromDefault() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(
            configuration
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("puffins mate for life");
  }

  @Test
  public void testTransitionOnBuildSetting_fromCommandLine() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "cats can't taste sugar"));

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(
            configuration
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("puffins mate for life");
  }

  @Test
  public void testTransitionOnBuildSetting_badValue() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 24}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "cats can't taste sugar"));

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "expected value of type 'string' for " + "//test:cute-animal-fact, but got 24 (int)");
  }

  @Test
  public void testTransitionOnBuildSetting_noSuchTarget() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:i-am-not-real': 'imaginary-friend'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:i-am-not-real']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "no such target '//test:i-am-not-real': target "
            + "'i-am-not-real' not declared in package 'test'");
  }

  @Test
  public void testTransitionOnBuildSetting_noSuchPackage() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//i-am-not-real': 'imaginary-friend'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//i-am-not-real']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("no such package 'i-am-not-real': Unable to find build setting package");
  }

  @Test
  public void testTransitionOnBuildSetting_notABuildSetting() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _rule_impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/build_settings.bzl",
        "def _impl(ctx):",
        "  return []",
        "non_build_setting = rule(implementation = _impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'non_build_setting')",
        "my_rule(name = 'test')",
        "non_build_setting(name = 'cute-animal-fact')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "attempting to transition on '//test:cute-animal-fact' which is not a build setting");
  }

  @Test
  public void testTransitionOnBuildSetting_dontStoreDefault() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': '" + CUTE_ANIMAL_DEFAULT + "'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "cats can't taste sugar"));

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().getStarlarkOptions())
        .doesNotContainKey(Label.parseAbsoluteUnchecked("//test:cute-animal-fact"));
  }

  @Test
  public void testTransitionReadsBuildSetting_fromDefault() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': settings['//test:cute-animal-fact']+' <- TRUE'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//test:cute-animal-fact'],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(
            configuration
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("cows produce more milk when they listen to soothing music <- TRUE");
  }

  @Test
  public void testTransitionReadsBuildSetting_fromCommandLine() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': settings['//test:cute-animal-fact']+' <- TRUE'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//test:cute-animal-fact'],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "rats are ticklish"));

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(
            configuration
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("rats are ticklish <- TRUE");
  }

  @Test
  public void testTransitionReadsBuildSetting_notABuildSetting() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': 'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//test:cute-animal-fact'],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _rule_impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/build_settings.bzl",
        "def _impl(ctx):",
        "  return []",
        "non_build_setting = rule(implementation = _impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'non_build_setting')",
        "my_rule(name = 'test')",
        "non_build_setting(name = 'cute-animal-fact')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "attempting to transition on '//test:cute-animal-fact' which is not a build setting");
  }

  @Test
  public void testTransitionReadsBuildSetting_noSuchTarget() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:cute-animal-fact': settings['//test:cute-animal-fact']+' <- TRUE'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//test:i-am-not-real'],",
        "  outputs = ['//test:cute-animal-fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "no such target '//test:i-am-not-real': target "
            + "'i-am-not-real' not declared in package 'test'");
  }

  @Test
  public void testAliasedBuildSetting() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:fact':  'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(name = 'fact', actual = ':cute-animal-fact')",
        "string_flag(",
        "  name = 'cute-animal-fact',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "rats are ticklish"));

    ImmutableMap<Label, Object> starlarkOptions =
        getConfiguration(getConfiguredTarget("//test")).getOptions().getStarlarkOptions();
    assertThat(starlarkOptions.get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("puffins mate for life");
    assertThat(starlarkOptions).doesNotContainKey(Label.parseAbsoluteUnchecked("//test:fact"));
    assertThat(starlarkOptions.keySet())
        .containsExactly(Label.parseAbsoluteUnchecked("//test:cute-animal-fact"));
  }

  @Test
  public void testAliasedBuildSetting_chainedAliases() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:fact':  'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(name = 'fact', actual = ':alias2')",
        "alias(name = 'alias2', actual = ':cute-animal-fact')",
        "string_flag(",
        "  name = 'cute-animal-fact',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");

    useConfiguration(ImmutableMap.of("//test:cute-animal-fact", "rats are ticklish"));

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(
            configuration
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseAbsoluteUnchecked("//test:cute-animal-fact")))
        .isEqualTo("puffins mate for life");
  }

  @Test
  public void testAliasedBuildSetting_configuredActualValue() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:fact':  'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:fact']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(",
        "  name = 'fact',",
        "  actual = select({",
        "    '//conditions:default': ':cute-animal-fact',",
        "    ':true-config': 'other-cute-animal-fact',",
        "  })",
        ")",
        "config_setting(",
        "  name = 'true-config',",
        "  values = {'test_arg': 'true'},",
        ")",
        "string_flag(",
        "  name = 'cute-animal-fact',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")",
        "string_flag(",
        "  name = 'other-cute-animal-fact',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "attempting to transition on aliased build setting '//test:fact', the actual value of"
            + " which uses select().");
  }

  @Test
  public void testAliasedBuildSetting_cyclicalAliases() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//test:alias1':  'puffins mate for life'}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//test:alias1']",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(name = 'alias1', actual = ':alias2')",
        "alias(name = 'alias2', actual = ':alias1')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Error with aliased build settings related to '//test:alias1'.");
  }

  @Test
  public void testAliasedBuildSetting_setAliasAndActual() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {",
        "    '//test:alias':  'puffins mate for life',",
        "    '//test:actual':  'cats cannot taste sugar',",
        "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = [",
        "    '//test:alias',",
        "    '//test:actual',",
        "  ]",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(name = 'alias', actual = ':actual')",
        "string_flag(",
        "  name = 'actual',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Error with aliased build settings related to '//test:actual'.");
  }

  @Test
  public void testAliasedBuildSetting_outputReturnMismatch() throws Exception {
    setStarlarkSemanticsOptions(
        "--experimental_starlark_config_transitions=true", "--experimental_build_setting_api");
    scratch.file(
        "test/transitions.bzl",
        "def _transition_impl(settings, attr):",
        "  return {",
        "    '//test:actual':  'cats cannot taste sugar',",
        "}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = [",
        "    '//test:alias',",
        "  ]",
        ")");
    writeRulesBuildSettingsAndBUILDforBuildSettingTransitionTests();
    scratch.overwriteFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "my_rule(name = 'test')",
        "alias(name = 'alias', actual = ':actual')",
        "string_flag(",
        "  name = 'actual',",
        "  build_setting_default = '" + CUTE_ANIMAL_DEFAULT + "',",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("transition function returned undeclared output '//test:actual'");
  }

  @Test
  public void testOneParamTransitionFunctionApiFails() throws Exception {
    setStarlarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("_impl() accepts no more than 1 positional argument but got 2");
  }

  @Test
  public void testCannotTransitionOnExperimentalFlag() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:experimental_build_setting_api': True}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:experimental_build_setting_api'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  },",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Cannot transition on --experimental_* or --incompatible_* options");
  }

  @Test
  public void testCannotTransitionWithoutWhitelist() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [],",
        ")");
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--test_arg=pre-transition");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Use of Starlark transition without whitelist");
  }

  @Test
  public void testNoNullOptionValues() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  if settings['//command_line_option:nullable_option'] == None:",
        "    return {'//command_line_option:test_arg': ['post-transition']}",
        "  else:",
        "    return {'//command_line_option:test_arg': settings['//command_line_option:test_arg']}",
        "my_transition = transition(implementation = _impl,",
        "  inputs = [",
        "    '//command_line_option:test_arg',",
        "    '//command_line_option:nullable_option'",
        "  ],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--nullable_option=", "--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  @Test
  public void testWhitelistOnRuleNotTargets() throws Exception {
    // whitelists //test/...
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file(
        "neverland/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");
    scratch.file("test/BUILD");
    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//neverland:test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  // TODO(juliexxia): flip this test when this isn't allowed anymore.
  @Test
  public void testWhitelistOnTargetsStillWorks() throws Exception {
    // whitelists //test/...
    writeWhitelistFile();
    scratch.file(
        "neverland/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['post-transition']}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "neverland/rules.bzl",
        "load('//neverland:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file(
        "test/BUILD", "load('//neverland:rules.bzl', 'my_rule')", "my_rule(name = 'test')");
    scratch.file("neverland/BUILD");
    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  /**
   * Regression test to ensure that an empty dict is not interpreted as a dict of dicts and
   * generates the proper error message.
   */
  @Test
  public void testEmptyReturnDict() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "transition outputs [//command_line_option:test_arg] were "
            + "not defined by transition function");
  }

  @Test
  public void composingTransitionReportsAllStarlarkErrors() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/build_settings.bzl",
        "def _impl(ctx):",
        "  return []",
        "string_flag = rule(implementation = _impl, build_setting = config.string(flag=True))");
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {}",
        "attr_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//test:attr_transition_output_flag'])",
        "self_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//test:self_transition_output_flag'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'attr_transition', 'self_transition')",
        "def _impl(ctx):",
        "  return []",
        "rule_with_attr_transition = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist'),",
        "    'deps': attr.label_list(cfg = attr_transition),",
        "  })",
        "rule_with_self_transition = rule(",
        "  implementation = _impl,",
        "  cfg = self_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist'),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'rule_with_attr_transition', 'rule_with_self_transition')",
        "load('//test:build_settings.bzl', 'string_flag')",
        "string_flag(name = 'attr_transition_output_flag', build_setting_default='')",
        "string_flag(name = 'self_transition_output_flag', build_setting_default='')",
        "rule_with_attr_transition(name = 'buildme', deps = [':adep'])",
        "rule_with_self_transition(name = 'adep')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:buildme");
    assertContainsEvent(
        "transition outputs [//test:attr_transition_output_flag] were not defined by transition "
            + "function");
    assertContainsEvent(
        "transition outputs [//test:self_transition_output_flag] were not defined by transition "
            + "function");
  }

  @Test
  public void testTransitionOnDefine() throws Exception {
    writeWhitelistFile();
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {'//command_line_option:define': 'chonky=true'}",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:define'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent("Starlark transition on --define not supported - try using build settings");
  }
}
