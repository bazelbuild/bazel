// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.test.TestEnvironmentInfo;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark interaction with testing support. */
@RunWith(JUnit4.class)
public class StarlarkTestingModuleTest extends BuildViewTestCase {

  @Test
  public void testStarlarkRulePropagatesExecutionInfoProvider() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def my_rule_impl(ctx):",
        "  exec_info = testing.ExecutionInfo({'requires-darwin': '1'})",
        "  return [exec_info]",
        "my_rule = rule(implementation = my_rule_impl,",
        "  attrs = {},",
        ")");
    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'my_rule')",
        "my_rule(",
        "    name = 'my_target',",
        ")");

    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/apple_starlark:my_target");
    ExecutionInfo provider = starlarkTarget.get(ExecutionInfo.PROVIDER);

    assertThat(provider.getExecutionInfo().get("requires-darwin")).isEqualTo("1");
  }

  @Test
  public void testStarlarkRulePropagatesTestEnvironmentProvider() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def my_rule_impl(ctx):",
        "  test_env = testing.TestEnvironment({'XCODE_VERSION_OVERRIDE': '7.3.1'})",
        "  return [test_env]",
        "my_rule = rule(implementation = my_rule_impl,",
        "  attrs = {},",
        ")");
    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'my_rule')",
        "my_rule(",
        "    name = 'my_target',",
        ")");

    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/apple_starlark:my_target");
    TestEnvironmentInfo provider = starlarkTarget.get(TestEnvironmentInfo.PROVIDER);

    assertThat(provider.getEnvironment().get("XCODE_VERSION_OVERRIDE")).isEqualTo("7.3.1");
  }

  @Test
  public void testExecutionInfoProviderCanMarkTestAsLocal() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def my_rule_test_impl(ctx):",
        "  exec_info = testing.ExecutionInfo({'local': ''})",
        "  ctx.actions.write(ctx.outputs.executable, '', True)",
        "  return [exec_info]",
        "my_rule_test = rule(implementation = my_rule_test_impl,",
        "    test = True,",
        "    attrs = {},",
        ")");
    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'my_rule_test')",
        "my_rule_test(",
        "    name = 'my_target',",
        ")");

    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/apple_starlark:my_target");
    TestRunnerAction testAction =
        (TestRunnerAction)
            getGeneratingAction(TestProvider.getTestStatusArtifacts(starlarkTarget).get(0));

    assertThat(testAction.getTestProperties().isRemotable()).isFalse();
  }
}
