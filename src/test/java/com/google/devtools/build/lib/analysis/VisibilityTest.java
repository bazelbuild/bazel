// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for visibility of targets. */
@RunWith(JUnit4.class)
public class VisibilityTest extends AnalysisTestCase {

  void setupArgsScenario() throws Exception {
    scratch.file("tool/tool.sh", "#!/bin/sh", "echo Hello > $2", "cat $1 >> $2");
    scratch.file("rule/BUILD");
    scratch.file(
        "rule/rule.bzl",
        "def _impl(ctx):",
        "  output = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(",
        "    inputs = ctx.files._tool + ctx.files.data,",
        "    executable = ctx.files._tool[0].path,",
        "    arguments =  [f.path for f in ctx.files.data] + [output.path],",
        "    outputs = [output],",
        "  )",
        "",
        "greet = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'data' : attr.label(allow_files=True),",
        "    '_tool' : attr.label(cfg='host', allow_files=True,",
        "                         default = Label('//tool:tool.sh')),",
        "  },",
        "  outputs = {'out' : '%{name}.out'},",
        ")");
    scratch.file("data/data.txt", "World");
    scratch.file(
        "use/BUILD",
        "load('//rule:rule.bzl', 'greet')",
        "",
        "greet(",
        "  name = 'world',",
        "  data = '//data:data.txt',",
        ")");
  }

  @Test
  public void testToolVisibilityRuleCheckAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//rule:__pkg__'])");
    useConfiguration("--incompatible_visibility_private_attributes_at_definition");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testToolVisibilityRuleCheckAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//rule:__pkg__'])");
    useConfiguration("--noincompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testToolVisibilityUseCheckAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//use:__pkg__'])");
    useConfiguration("--noincompatible_visibility_private_attributes_at_definition");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testToolVisibilityUseCheckAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//use:__pkg__'])");
    useConfiguration("--incompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testToolVisibilityPrivateCheckAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:private'])");
    useConfiguration("--noincompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testToolVisibilityPrivateCheckAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:public'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:private'])");
    useConfiguration("--incompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testDataVisibilityUseCheckPrivateAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//use:__pkg__'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");
    useConfiguration("--noincompatible_visibility_private_attributes_at_definition");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testDataVisibilityUseCheckPrivateAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//use:__pkg__'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");
    useConfiguration("--incompatible_visibility_private_attributes_at_definition");
    update("//use:world");
    assertThat(hasErrors(getConfiguredTarget("//use:world"))).isFalse();
  }

  @Test
  public void testDataVisibilityPrivateCheckPrivateAtRule() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:private'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");
    useConfiguration("--incompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }

  @Test
  public void testDataVisibilityPrivateCheckPrivateAtUse() throws Exception {
    setupArgsScenario();
    scratch.file("data/BUILD", "exports_files(['data.txt'], visibility=['//visibility:private'])");
    scratch.file("tool/BUILD", "exports_files(['tool.sh'], visibility=['//visibility:public'])");
    useConfiguration("--noincompatible_visibility_private_attributes_at_definition");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//use:world"));
  }
}
