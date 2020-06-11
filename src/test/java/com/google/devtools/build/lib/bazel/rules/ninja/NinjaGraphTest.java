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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.PhonyTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule} */
@RunWith(JUnit4.class)
public class NinjaGraphTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(new NinjaGraphRule());
    return builder.build();
  }

  @Before
  public void setUp() throws Exception {
    setStarlarkSemanticsOptions("--experimental_ninja_actions");
  }

  @Test
  public void testNinjaGraphRule() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')", "toplevel_output_directories(paths = ['build_config'])");

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
            "graph",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(1);

    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);
    assertThat(action).isInstanceOf(SymlinkAction.class);
    SymlinkAction symlinkAction = (SymlinkAction) action;
    assertThat(symlinkAction.executeUnconditionally()).isTrue();
    assertThat(symlinkAction.getInputPath())
        .isEqualTo(PathFragment.create("/workspace/build_config/input.txt"));
    assertThat(symlinkAction.getPrimaryOutput().getExecPathString())
        .isEqualTo("build_config/input.txt");

    NinjaGraphProvider provider = configuredTarget.getProvider(NinjaGraphProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getOutputRoot()).isEqualTo(PathFragment.create("build_config"));
    assertThat(provider.getWorkingDirectory()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
    assertThat(provider.getPhonyTargetsMap()).isEmpty();
    assertThat(provider.getTargetsMap()).hasSize(1);

    NinjaTarget target = Iterables.getOnlyElement(provider.getTargetsMap().values());
    assertThat(target.getRuleName()).isEqualTo("echo");
    assertThat(target.getAllInputs())
        .containsExactly(PathFragment.create("build_config/input.txt"));
    assertThat(target.getAllOutputs())
        .containsExactly(PathFragment.create("build_config/hello.txt"));
  }

  @Test
  public void testNinjaGraphRuleWithPhonyTarget() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')", "toplevel_output_directories(paths = ['build_config'])");

    // We do not have to have the real files in place, the rule only reads
    // the contents of Ninja files.
    scratch.file(
        "build_config/build.ninja",
        "rule echo",
        "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
        "build hello.txt: echo input.txt",
        "build alias: phony hello.txt");

    ConfiguredTarget configuredTarget =
        scratchConfiguredTarget(
            "",
            "graph",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['input.txt'])");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);

    assertThat(actions).hasSize(1);
    ActionAnalysisMetadata action = Iterables.getOnlyElement(actions);
    assertThat(action).isInstanceOf(SymlinkAction.class);
    SymlinkAction symlinkAction = (SymlinkAction) action;
    assertThat(symlinkAction.executeUnconditionally()).isTrue();
    assertThat(symlinkAction.getInputPath())
        .isEqualTo(PathFragment.create("/workspace/build_config/input.txt"));
    assertThat(symlinkAction.getPrimaryOutput().getExecPathString())
        .isEqualTo("build_config/input.txt");

    NinjaGraphProvider provider = configuredTarget.getProvider(NinjaGraphProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getOutputRoot()).isEqualTo(PathFragment.create("build_config"));
    assertThat(provider.getWorkingDirectory()).isEqualTo(PathFragment.create("build_config"));
    assertThat(provider.getTargetsMap()).hasSize(1);

    NinjaTarget target = Iterables.getOnlyElement(provider.getTargetsMap().values());
    assertThat(target.getRuleName()).isEqualTo("echo");
    assertThat(target.getAllInputs()).containsExactly(PathFragment.create("input.txt"));
    assertThat(target.getAllOutputs()).containsExactly(PathFragment.create("hello.txt"));

    PathFragment alias = PathFragment.create("alias");
    assertThat(provider.getPhonyTargetsMap().keySet()).containsExactly(alias);
    PhonyTarget phonyTarget = provider.getPhonyTargetsMap().get(alias);
    assertThat(phonyTarget.isAlwaysDirty()).isFalse();
    assertThat(phonyTarget.getPhonyNames()).isEmpty();
    assertThat(phonyTarget.getDirectExplicitInputs())
        .containsExactly(PathFragment.create("hello.txt"));
  }

  @Test
  public void testNinjaGraphRuleWithPhonyTree() throws Exception {
    rewriteWorkspace(
        "workspace(name = 'test')", "toplevel_output_directories(paths = ['build_config'])");

    // We do not have to have the real files in place, the rule only reads
    // the contents of Ninja files.
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
            "graph",
            "ninja_graph(name = 'graph', output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_root_inputs = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt'])");
    assertThat(configuredTarget).isInstanceOf(RuleConfiguredTarget.class);
    RuleConfiguredTarget ninjaConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    ImmutableList<ActionAnalysisMetadata> actions = ninjaConfiguredTarget.getActions();
    assertThat(actions).hasSize(5);
    List<String> outputs = Lists.newArrayList();
    actions.forEach(a -> outputs.add(Iterables.getOnlyElement(a.getOutputs()).getExecPathString()));
    assertThat(outputs)
        .containsExactlyElementsIn(
            new String[] {
              "build_config/a.txt",
              "build_config/b.txt",
              "build_config/c.txt",
              "build_config/d.txt",
              "build_config/e.txt"
            });

    for (ActionAnalysisMetadata action : actions) {
      assertThat(action).isInstanceOf(SymlinkAction.class);
      SymlinkAction symlinkAction = (SymlinkAction) action;
      assertThat(symlinkAction.executeUnconditionally()).isTrue();
      assertThat(symlinkAction.getInputPath().getParentDirectory())
          .isEqualTo(PathFragment.create("/workspace/build_config"));
      assertThat(symlinkAction.getInputPath().getFileExtension()).isEqualTo("txt");
      PathFragment execRootPath = symlinkAction.getPrimaryOutput().getExecPath();
      assertThat(execRootPath.getParentDirectory()).isEqualTo(PathFragment.create("build_config"));
      assertThat(execRootPath.getFileExtension()).isEqualTo("txt");
    }
  }
}
