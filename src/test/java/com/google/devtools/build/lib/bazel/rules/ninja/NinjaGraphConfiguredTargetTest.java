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

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGenericAction;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Injectable;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule} */
@RunWith(JUnit4.class)
public class NinjaGraphConfiguredTargetTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(new NinjaGraphRule());
    return builder.build();
  }

  @Before
  public void setUp() throws Exception {
    setSkylarkSemanticsOptions("--experimental_ninja_actions");
  }

  @Test
  public void testNinjaGraphRule() throws Exception {
    rewriteWorkspace("workspace(name = 'test')",
        "dont_symlink_directories_in_execroot(paths = ['build_config'])");

    scratch.file("build_config/input.txt", "World");
    scratch.file("build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt");

    ConfiguredTarget configuredTarget = scratchConfiguredTarget("", "graph",
        "ninja_graph(name = 'graph', output_root = 'build_config',",
        " working_directory = 'build_config',",
        " main = 'build_config/build.ninja',",
        " output_root_inputs = ['input.txt'])");
    NinjaGraphProvider provider = configuredTarget.getProvider(NinjaGraphProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getOutputRoot()).isEqualTo("build_config");
    assertThat(provider.getWorkingDirectory()).isEqualTo("build_config");

    NestedSet<Artifact> filesToBuild = getFilesToBuild(configuredTarget);
    assertThat(artifactsToStrings(filesToBuild)).containsExactly("/ build_config/input.txt",
        "/ build_config/hello.txt");

    ActionGraph actionGraph = getActionGraph();
    for (Artifact artifact : filesToBuild.toList()) {
      ActionAnalysisMetadata action = actionGraph.getGeneratingAction(artifact);
      if ("hello.txt".equals(artifact.getFilename())) {
        assertThat(action instanceof NinjaGenericAction).isTrue();
        NinjaGenericAction ninjaAction = (NinjaGenericAction) action;
        List<CommandLineAndParamFileInfo> commandLines = ninjaAction.getCommandLines()
            .getCommandLines();
        assertThat(commandLines).hasSize(1);
        assertThat(commandLines.get(0).commandLine.toString()).endsWith(
            "cd build_config && echo \"Hello $(cat input.txt)!\" > hello.txt");
        assertThat(ninjaAction.getPrimaryInput().getRootRelativePathString())
            .isEqualTo("build_config/input.txt");
        assertThat(ninjaAction.getPrimaryOutput().getRootRelativePathString())
            .isEqualTo("build_config/hello.txt");
      } else {
        assertThat(action instanceof SymlinkAction).isTrue();
        SymlinkAction symlinkAction = (SymlinkAction) action;
        assertThat(symlinkAction.executeUnconditionally()).isTrue();
        assertThat(symlinkAction.getInputPath()).isEqualTo(
            PathFragment.create("/workspace/build_config/input.txt"));
        assertThat(symlinkAction.getPrimaryOutput().getRootRelativePathString()).isEqualTo(
            "build_config/input.txt");
      }
    }

    // assertThat(provider.getScope()).isNotNull();
    // assertThat(provider.getSymlinkedUnderOutputRoot()).hasSize(1);
    // assertThat(provider.getSymlinkedUnderOutputRoot().asList().get(0).getExecPath())
    //     .isEqualTo(PathFragment.create("build_config/input.txt"));

    // PathFragment key = PathFragment.create("hello.txt");
    // assertThat(provider.getTargets()).hasSize(1);
    // assertThat(provider.getTargets()).containsKey(key);
    // assertThat(provider.getTargets().get(key).getRuleName()).isEqualTo("echo");
  }

  // todo: test actions and command lines
  // todo: test phony actions (?)
  // todo: test symlinks
  // todo: we can have genrule depending on this for the integration test


  // todo: blind the test from checking for serialization
}
