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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaAction;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaBuildRule;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaBuildRule} */
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
  public void testOneTarget() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt");

    ConfiguredTarget buildTarget =
        scratchConfiguredTarget(
            "",
            "build_ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])",
            "ninja_build(name = 'build_ninja_target', ninja_graph = ':graph',",
            " targets = ['hello.txt'])");
    Artifact helloArtifact = Iterables.getOnlyElement(getFilesToBuild(buildTarget).toList());
    assertThat(helloArtifact.getExecPathString()).isEqualTo("build_config/hello.txt");
    assertThat(getGeneratingAction(helloArtifact)).isInstanceOf(NinjaAction.class);
  }

  @Test
  public void testPhonyTarget() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file("build_config/variant.txt", "Sun");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt",
        "build sun.txt: echo variant.txt",
        "build alias: phony hello.txt sun.txt");

    ConfiguredTarget buildTarget =
        scratchConfiguredTarget(
            "",
            "build_ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt', 'variant.txt'])",
            "ninja_build(name = 'build_ninja_target', ninja_graph = ':graph',",
            " targets = ['alias'])");
    NestedSet<Artifact> filesToBuild = getFilesToBuild(buildTarget);
    List<String> execPaths = artifactsToExecPaths(filesToBuild);
    assertThat(execPaths).containsExactlyElementsIn(new String[]{"build_config/hello.txt",
        "build_config/sun.txt"});
  }

  @Test
  public void testPhonyWithGroups() throws Exception {
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

    ConfiguredTarget buildTarget =
        scratchConfiguredTarget(
            "",
            "build_ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt'])",
            "ninja_build(name = 'build_ninja_target', ninja_graph = ':graph',",
            " targets = ['alias'], output_groups = {'out_group1': ['a', 'b'], "
                + "'out_group2': ['c', 'd', 'e'] })");
    assertThat(artifactsToExecPaths(getFilesToBuild(buildTarget)))
        .containsExactlyElementsIn(new String[]{
        "build_config/hello.txt",
        "build_config/a",
        "build_config/b",
        "build_config/c",
        "build_config/d",
        "build_config/e"
    });
    assertThat(artifactsToExecPaths(getOutputGroup(buildTarget, "out_group1")))
        .containsExactlyElementsIn(new String[]{
        "build_config/a",
        "build_config/b",
    });
    assertThat(artifactsToExecPaths(getOutputGroup(buildTarget, "out_group2")))
        .containsExactlyElementsIn(new String[]{
        "build_config/c",
        "build_config/d",
        "build_config/e"
    });
  }

  @Test
  public void testOnlyPartIsBuilt() throws Exception {
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

    ConfiguredTarget buildTarget =
        scratchConfiguredTarget(
            "",
            "build_ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt'])",
            "ninja_build(name = 'build_ninja_target', ninja_graph = ':graph',",
            " targets = ['a', 'b'])");
    assertThat(artifactsToExecPaths(getFilesToBuild(buildTarget)))
        .containsExactlyElementsIn(new String[]{
            "build_config/a",
            "build_config/b",
        });
  }

  @Test
  public void testErrorWhenUnknownTarget() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt");

    useConfiguration("--allow_analysis_failures=true");
    ConfiguredTarget buildTarget =
        scratchConfiguredTarget(
            "",
            "build_ninja_target",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])",
            "ninja_build(name = 'build_ninja_target', ninja_graph = ':graph',",
            " targets = ['not_existing'])");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) buildTarget.get(AnalysisFailureInfo.SKYLARK_CONSTRUCTOR.getKey());
    assertThat(info).isNotNull();
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("Required target 'not_existing' is not created in ninja_graph.");
    assertThat(failure.getLabel()).isEqualTo(buildTarget.getLabel());

    assertThat(getFilesToBuild(buildTarget).toList()).isEmpty();
  }

  private static List<String> artifactsToExecPaths(NestedSet<Artifact> filesToBuild) {
    return filesToBuild.toList().stream()
        .map(Artifact::getExecPathString)
        .collect(Collectors.toList());
  }
}
