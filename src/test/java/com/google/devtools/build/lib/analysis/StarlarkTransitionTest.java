// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Scratch;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test of common logic between Starlark-defined transitions. Rule-transition- or
 * attr-transition-specific logic should be tested in {@link StarlarkRuleTransitionProviderTest} and
 * {@link StarlarkAttrTransitionProviderTest}.
 */
@RunWith(JUnit4.class)
public class StarlarkTransitionTest extends BuildViewTestCase {
  static void writeAllowlistFile(Scratch scratch) throws Exception {
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
  }

  @Test
  public void testDupeSettingsInInputsThrowsError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        "def _setting_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _setting_impl,",
        "  build_setting = config.string(flag=True),",
        ")",
        "def _transition_impl(settings, attr):",
        "  return {'//test:formation': 'mesa'}",
        "formation_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['@//test:formation', '//test:formation'],", // duplicates here
        "  outputs = ['//test:formation'],",
        ")",
        "def _impl(ctx):",
        "  return []",
        "state = rule(",
        "  implementation = _impl,",
        "  cfg = formation_transition,",
        "  attrs = {",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'state', 'string_flag')",
        "state(name = 'arizona')",
        "string_flag(name = 'formation', build_setting_default = 'canyon')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "Transition declares duplicate build setting '@//test:formation' in INPUTS");
  }

  @Test
  public void testDupeSettingsInOutputsThrowsError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        "def _setting_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _setting_impl,",
        "  build_setting = config.string(flag=True),",
        ")",
        "def _transition_impl(settings, attr):",
        "  return {'//test:formation': 'mesa'}",
        "formation_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//test:formation'],",
        "  outputs = ['@//test:formation', '//test:formation'],", // duplicates here
        ")",
        "def _impl(ctx):",
        "  return []",
        "state = rule(",
        "  implementation = _impl,",
        "  cfg = formation_transition,",
        "  attrs = {",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'state', 'string_flag')",
        "state(name = 'arizona')",
        "string_flag(name = 'formation', build_setting_default = 'canyon')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "Transition declares duplicate build setting '@//test:formation' in OUTPUTS");
  }

  @Test
  public void testDifferentFormsOfFlagInInputsAndOutputs() throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        "def _setting_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _setting_impl,",
        "  build_setting = config.string(flag=True),",
        ")",
        "def _transition_impl(settings, attr):",
        "  return {",
        "    '//test:formation': settings['@//test:formation']+'-transitioned',",
        "  }",
        "formation_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['@//test:formation'],",
        "  outputs = ['//test:formation'],",
        ")",
        "def _impl(ctx):",
        "  return []",
        "state = rule(",
        "  implementation = _impl,",
        "  cfg = formation_transition,",
        "  attrs = {",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'state', 'string_flag')",
        "state(name = 'arizona')",
        "string_flag(name = 'formation', build_setting_default = 'canyon')");

    Map<Label, Object> starlarkOptions =
        getConfiguration(getConfiguredTarget("//test:arizona")).getOptions().getStarlarkOptions();
    assertThat(starlarkOptions).hasSize(1);
    assertThat(starlarkOptions.get(Label.parseAbsoluteUnchecked("//test:formation")))
        .isEqualTo("canyon-transitioned");
  }
}
