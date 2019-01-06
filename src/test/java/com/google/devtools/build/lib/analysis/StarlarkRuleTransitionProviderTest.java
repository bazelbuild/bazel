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
import static junit.framework.TestCase.fail;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for StarlarkRuleTransitionProvider. */
@RunWith(JUnit4.class)
public class StarlarkRuleTransitionProviderTest extends BuildViewTestCase {

  @Test
  public void testOutputOnlyTransition() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
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
        "my_rule = rule(implementation = _impl, cfg = my_transition)");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("post-transition");
  }

  @Test
  public void testInputAndOutputTransition() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
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
        "my_rule = rule(implementation = _impl, cfg = my_transition)");

    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    useConfiguration("--test_arg=pre-transition");

    BuildConfiguration configuration = getConfiguration(getConfiguredTarget("//test"));
    assertThat(configuration.getOptions().get(TestOptions.class).testArguments)
        .containsExactly("pre-transition->post-transition");
  }

  @Test
  public void testBuildSettingCannotTransition() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
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
        "  build_setting = config.string()",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "Build setting rules cannot use the `cfg` param to apply transitions to themselves");
  }

  @Test
  public void testBadCfgInput() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  cfg = 'my_transition',",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "`cfg` must be set to a transition object initialized by the transition() function.");
  }

  @Test
  public void testMultipleReturnConfigs() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
    scratch.file(
        "test/transitions.bzl",
        "def _impl(settings, attr):",
        "  return {",
        "      't0': {'//command_line_option:test_arg': ['split_one']},",
        "      't1': {'//command_line_option:test_arg': ['split_two']},",
        "  }",
        "my_transition = transition(implementation = _impl, inputs = [],",
        "  outputs = ['//command_line_option:test_arg'])");
    scratch.file(
        "test/rules.bzl",
        "load('//test:transitions.bzl', 'my_transition')",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(implementation = _impl, cfg = my_transition)");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    try {
      getConfiguredTarget("//test");
      fail("Should have failed");
    } catch (RuntimeException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Rule transition only allowed to return a single transitioned configuration.");
    }
  }

  @Test
  public void testCanDoBadStuffWithParameterizedTransitionsAndSelects() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
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
        "  attrs = {'my_configurable_attr': attr.bool(default = False)},",
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

    try {
      getConfiguredTarget("//test");
      fail("Should have failed");
    } catch (RuntimeException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "No attribute 'my_configurable_attr'. "
                  + "Either this attribute does not exist for this rule or is set by a select. "
                  + "Starlark rule transitions currently cannot read attributes behind selects.");
    }
  }

  @Test
  public void testLabelTypedAttrReturnsLabelNotDep() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");
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
        "  attrs = {'dict_attr': attr.label_keyed_string_dict()},",
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

  @Test
  public void testCannotTransitionWithoutFlag() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=false");
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
        "my_rule = rule(implementation = _impl, cfg = my_transition)");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'test')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test");
    assertContainsEvent(
        "transition() is experimental and disabled by default. This API is in "
            + "development and subject to change at any time. Use "
            + "--experimental_starlark_config_transitions to use this experimental API.");
  }
}
