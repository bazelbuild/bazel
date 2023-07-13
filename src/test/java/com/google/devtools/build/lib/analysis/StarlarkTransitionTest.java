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
        "Transition declares duplicate build setting '@@//test:formation' in INPUTS");
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
        "Transition declares duplicate build setting '@@//test:formation' in OUTPUTS");
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
        "  formation = settings['@//test:formation']",
        "  if formation.endswith('-transitioned'):",
        "    new_value = formation",
        "  else:",
        "    new_value = formation + '-transitioned'",
        "  return {",
        "    '//test:formation': new_value,",
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
    assertThat(starlarkOptions.get(Label.parseCanonicalUnchecked("//test:formation")))
        .isEqualTo("canyon-transitioned");
  }

  private void writeDefBzlWithStringFlagAndEaterRule() throws Exception {
    scratch.file(
        "test/defs.bzl",
        "def _setting_impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _setting_impl,",
        "  build_setting = config.string(flag=True),",
        ")",
        "def _transition_impl(settings, attr):",
        "  if settings['@//options:fruit'].endswith('-eaten'):",
        "    return {'//options:fruit': settings['@//options:fruit']}",
        "  return {'//options:fruit': settings['@//options:fruit'] + '-eaten'}",
        "eating_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['@//options:fruit'],",
        "  outputs = ['//options:fruit'],",
        ")",
        "def _impl(ctx):",
        "  return []",
        "eater = rule(",
        "  implementation = _impl,",
        "  cfg = eating_transition,",
        "  attrs = {",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  })");
  }

  @Test
  public void testDifferentDefaultsRerunsTransitionTest() throws Exception {
    writeAllowlistFile(scratch);
    writeDefBzlWithStringFlagAndEaterRule();
    scratch.file(
        "options/BUILD",
        "load('//test:defs.bzl', 'string_flag')",
        "string_flag(name = 'fruit', build_setting_default = 'apple')");
    scratch.file("test/BUILD", "load('//test:defs.bzl', 'eater')", "eater(name = 'foo')");

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo"))
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//options:fruit")))
        .isEqualTo("apple-eaten");

    scratch.overwriteFile(
        "options/BUILD",
        "load('//test:defs.bzl', 'string_flag')",
        "string_flag(name = 'fruit', build_setting_default = 'orange')");
    invalidatePackages();
    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo"))
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//options:fruit")))
        .isEqualTo("orange-eaten");
  }

  @Test
  public void testAliasChangeRerunsTransitionTest() throws Exception {
    writeAllowlistFile(scratch);
    writeDefBzlWithStringFlagAndEaterRule();
    scratch.file(
        "options/BUILD",
        "load('//test:defs.bzl', 'string_flag')",
        "string_flag(name = 'usually_apple', build_setting_default = 'apple')",
        "string_flag(name = 'usually_orange', build_setting_default = 'orange')",
        "alias(name = 'fruit', actual = ':usually_apple')");
    scratch.file("test/BUILD", "load('//test:defs.bzl', 'eater')", "eater(name = 'foo')");

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo")).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//options:usually_apple"), "apple-eaten");

    scratch.overwriteFile(
        "options/BUILD",
        "load('//test:defs.bzl', 'string_flag')",
        "string_flag(name = 'usually_apple', build_setting_default = 'apple')",
        "string_flag(name = 'usually_orange', build_setting_default = 'orange')",
        "alias(name = 'fruit', actual = ':usually_orange')");
    invalidatePackages();

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo")).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//options:usually_orange"), "orange-eaten");
  }
}
