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

package com.google.devtools.build.lib.rules;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LabelBuildSettings} rules. */
@RunWith(JUnit4.class)
public class LabelBuildSettingTest extends BuildViewTestCase {

  private void writeRulesBzl(String type) throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");

    scratch.file(
        "test/rules.bzl",
        "def _my_rule_impl(ctx):",
        "    return struct(value = ctx.attr._label_setting[SimpleRuleInfo].value)",
        "",
        "my_rule = rule(",
        "    implementation = _my_rule_impl,",
        "    attrs = {",
        "        '_label_setting': attr.label(default = Label('//test:my_label_" + type + "')),",
        "    },",
        ")",
        "",
        "SimpleRuleInfo = provider(fields = ['value'])",
        "",
        "def _simple_rule_impl(ctx):",
        "    return [SimpleRuleInfo(value = ctx.attr.value)]",
        "",
        "simple_rule = rule(",
        "    implementation = _simple_rule_impl,",
        "    attrs = {",
        "        'value':attr.string(),",
        "    },",
        ")");
  }

  @Test
  public void testLabelSetting() throws Exception {
    String testing = "setting";
    writeRulesBzl(testing);
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "",
        "my_rule(name = 'my_rule')",
        "simple_rule(name = 'default', value = 'default_value')",
        "simple_rule(name = 'command_line', value = 'command_line_value')",
        "label_setting(name = 'my_label_" + testing + "', build_setting_default = ':default')");

    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])", "alias(name='b', actual='a')");

    ConfiguredTarget b = getConfiguredTarget("//test:my_rule");
    assertThat(b.get("value")).isEqualTo("default_value");
  }

  @Test
  public void testLabelFlag_default() throws Exception {
    String testing = "flag";
    writeRulesBzl(testing);
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "",
        "my_rule(name = 'my_rule')",
        "simple_rule(name = 'default', value = 'default_value')",
        "simple_rule(name = 'command_line', value = 'command_line_value')",
        "label_flag(name = 'my_label_" + testing + "', build_setting_default = ':default')");

    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])", "alias(name='b', actual='a')");

    ConfiguredTarget b = getConfiguredTarget("//test:my_rule");
    assertThat(b.get("value")).isEqualTo("default_value");
  }

  @Test
  public void testLabelFlag_set() throws Exception {
    String testing = "flag";
    writeRulesBzl(testing);
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "",
        "my_rule(name = 'my_rule')",
        "simple_rule(name = 'default', value = 'default_value')",
        "simple_rule(name = 'command_line', value = 'command_line_value')",
        "label_flag(name = 'my_label_" + testing + "', build_setting_default = ':default')");

    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])", "alias(name='b', actual='a')");

    useConfiguration(
        ImmutableMap.of(
            "//test:my_label_flag", Label.parseAbsoluteUnchecked("//test:command_line")));

    ConfiguredTarget b = getConfiguredTarget("//test:my_rule");
    assertThat(b.get("value")).isEqualTo("command_line_value");
  }
}
