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

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaAction;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaBuildRule;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link NinjaBuild} configured target factory. */
@RunWith(JUnit4.class)
public class NinjaBuildTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(new NinjaGraphRule());
    builder.addRuleDefinition(new NinjaBuildRule());
    return builder.build();
  }

  @Before
  public void setUp() throws Exception {
    setSkylarkSemanticsOptions("--experimental_ninja_actions");
  }

  @Test
  public void testNinjaBuildRule() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build build_config/hello.txt: echo build_config/input.txt");

    // Working directory is workspace root.
    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " output_groups= {'main': ['build_config/hello.txt']})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(1);
    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);
    assertThat(action).isInstanceOf(NinjaAction.class);
    NinjaAction ninjaAction = (NinjaAction) action;
    List<CommandLineAndParamFileInfo> commandLines =
        ninjaAction.getCommandLines().getCommandLines();
    assertThat(commandLines).hasSize(1);
    assertThat(commandLines.get(0).commandLine.toString())
        .endsWith("echo \"Hello $(cat build_config/input.txt)!\" > build_config/hello.txt");
    assertThat(ninjaAction.getPrimaryInput().getExecPathString())
        .isEqualTo("build_config/input.txt");
    assertThat(ninjaAction.getPrimaryOutput().getExecPathString())
        .isEqualTo("build_config/hello.txt");
  }

  @Test
  public void testNinjaGraphRuleWithPhonyTarget() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt",
        "build alias: phony hello.txt");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " output_groups= {'main': ['alias']})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);

    assertThat(actions).hasSize(1);
    assertThat(action).isInstanceOf(NinjaAction.class);
    NinjaAction ninjaAction = (NinjaAction) action;
    List<CommandLineAndParamFileInfo> commandLines =
        ninjaAction.getCommandLines().getCommandLines();
    assertThat(commandLines).hasSize(1);
    assertThat(commandLines.get(0).commandLine.toString())
        .endsWith("cd build_config && echo \"Hello $(cat input.txt)!\" > hello.txt");
    assertThat(ninjaAction.getPrimaryInput().getExecPathString())
        .isEqualTo("build_config/input.txt");
    assertThat(ninjaAction.getPrimaryOutput().getExecPathString())
        .isEqualTo("build_config/hello.txt");
  }

  @Test
  public void testNinjaGraphRuleWithPhonyTree() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/a.txt", "A");
    scratch.file("build_config/b.txt", "B");
    scratch.file("build_config/c.txt", "C");
    scratch.file("build_config/d.txt", "D");
    scratch.file("build_config/e.txt", "E");

    scratch.file(
        "build_config/build.ninja",
        "rule cat",
        "  command = cat ${in} > ${out}",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in} | tr '\\r\\n' ' ')!\" > ${out}",
        "build a: cat a.txt",
        "build b: cat b.txt",
        "build c: cat c.txt",
        "build d: cat d.txt",
        // e should be executed unconditionally as it depends on always-dirty phony action
        "build e: cat e.txt always_dirty",
        "build always_dirty: phony",
        "build group1: phony a b c",
        "build group2: phony d e",
        "build inputs_alias: phony group1 group2",
        "build hello.txt: echo inputs_alias",
        "build alias: phony hello.txt");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt'])",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " output_groups= {'main': ['alias']})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(6);
    List<String> outputs = Lists.newArrayList();
    actions.forEach(a -> outputs.add(Iterables.getOnlyElement(a.getOutputs()).getExecPathString()));
    assertThat(outputs)
        .containsExactlyElementsIn(
            new String[] {
              "build_config/hello.txt",
              "build_config/a",
              "build_config/b",
              "build_config/c",
              "build_config/d",
              "build_config/e"
            });

    for (ActionAnalysisMetadata action : actions) {
      Artifact artifact = action.getPrimaryOutput();
      if ("hello.txt".equals(artifact.getFilename())) {
        assertThat(action).isInstanceOf(NinjaAction.class);
        NinjaAction ninjaAction = (NinjaAction) action;
        List<CommandLineAndParamFileInfo> commandLines =
            ninjaAction.getCommandLines().getCommandLines();
        assertThat(commandLines).hasSize(1);
        assertThat(commandLines.get(0).commandLine.toString())
            .contains(
                "cd build_config && echo \"Hello $(cat inputs_alias | tr '\\r\\n' ' ')!\""
                    + " > hello.txt");
        List<String> inputPaths =
            ninjaAction.getInputs().toList().stream()
                .map(Artifact::getExecPathString)
                .collect(Collectors.toList());
        assertThat(inputPaths)
            .containsExactly(
                "build_config/a",
                "build_config/b",
                "build_config/c",
                "build_config/d",
                "build_config/e");
        assertThat(ninjaAction.getPrimaryOutput().getExecPathString())
            .isEqualTo("build_config/hello.txt");
      } else if ("e".equals(artifact.getFilename())) {
        assertThat(action).isInstanceOf(NinjaAction.class);
        NinjaAction ninjaAction = (NinjaAction) action;
        List<CommandLineAndParamFileInfo> commandLines =
            ninjaAction.getCommandLines().getCommandLines();
        assertThat(commandLines).hasSize(1);
        assertThat(commandLines.get(0).commandLine.toString())
            .endsWith("cd build_config && cat e.txt always_dirty > e");
        assertThat(ninjaAction.executeUnconditionally()).isTrue();
      }
    }
  }

  @Test
  public void testDepsMapping() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("input.txt", "World");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo placeholder");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja')",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " output_groups= {'main': ['hello.txt']},",
            " deps_mapping = {'placeholder': ':input.txt'})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(1);

    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);
    assertThat(action).isInstanceOf(NinjaAction.class);
    NinjaAction ninjaAction = (NinjaAction) action;
    List<CommandLineAndParamFileInfo> commandLines =
        ninjaAction.getCommandLines().getCommandLines();
    assertThat(commandLines).hasSize(1);
    assertThat(commandLines.get(0).commandLine.toString())
        .endsWith("cd build_config && echo \"Hello $(cat /workspace/input.txt)!\" > hello.txt");
    assertThat(ninjaAction.getPrimaryInput().getExecPathString()).isEqualTo("input.txt");
    assertThat(ninjaAction.getPrimaryOutput().getExecPathString())
        .isEqualTo("build_config/hello.txt");
  }

  @Test
  public void testOnlySubGraphIsCreated() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/a.txt", "A");
    scratch.file("build_config/b.txt", "B");
    scratch.file("build_config/c.txt", "C");
    scratch.file("build_config/d.txt", "D");
    scratch.file("build_config/e.txt", "E");

    scratch.file(
        "build_config/build.ninja",
        "rule cat",
        "  command = cat ${in} > ${out}",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in} | tr '\\r\\n' ' ')!\" > ${out}",
        "build a: cat a.txt",
        "build b: cat b.txt",
        "build c: cat c.txt",
        "build d: cat d.txt",
        "build e: cat e.txt",
        "build group1: phony a b c",
        "build group2: phony d e",
        "build inputs_alias: phony group1 group2",
        "build hello.txt: echo inputs_alias",
        "build alias: phony hello.txt");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt'])",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " output_groups= {'main': ['group1']})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(3);
    List<String> outputs = Lists.newArrayList();
    actions.forEach(a -> outputs.add(Iterables.getOnlyElement(a.getOutputs()).getExecPathString()));
    assertThat(outputs)
        .containsExactlyElementsIn(
            new String[] {
              "build_config/a", "build_config/b", "build_config/c",
            });
  }

  @Test
  public void testRuleWithDepfileVariable() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("input");
    scratch.file(
        "build_config/build.ninja",
        "rule rule123",
        "  command = executable -d ${depfile} ${in} > ${out}",
        "  depfile = ${out}.d",
        "  deps = gcc",
        "build out_file: rule123 ../input");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja')",
            "ninja_build(name = 'ninja_target', ninja_graph = 'graph',",
            " srcs = ['input'],",
            " output_groups= {'main': ['out_file']})");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(1);

    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);
    assertThat(action).isInstanceOf(NinjaAction.class);
    List<CommandLineAndParamFileInfo> commandLines =
        ((NinjaAction) action).getCommandLines().getCommandLines();
    assertThat(commandLines).hasSize(1);
    assertThat(commandLines.get(0).commandLine.toString())
        .endsWith("cd build_config && executable -d out_file.d ../input > out_file");
  }
}
