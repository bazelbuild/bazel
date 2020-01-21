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
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.actions.NinjaGraphRule;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Injectable;
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

    scratch.file("build_config/input.txt");
    scratch.file("build_config/build.ninja",
        "rule echo",
        "  command = echo 'Hello World!' > ${out}",
        "build hello.txt: echo");

    ConfiguredTarget configuredTarget = scratchConfiguredTarget("", "graph",
        "ninja_graph(name = 'graph', output_root = 'build_config',",
        " working_directory = 'build_config',",
        " main = 'build_config/build.ninja',",
        " output_root_inputs = ['input.txt'])");
    NinjaGraphProvider provider = configuredTarget.getProvider(NinjaGraphProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.getOutputRoot()).isEqualTo("build_config");
    assertThat(provider.getScope()).isNotNull();
    assertThat(provider.getSymlinkedUnderOutputRoot()).hasSize(1);
    assertThat(provider.getSymlinkedUnderOutputRoot().asList().get(0).getExecPath())
        .isEqualTo(PathFragment.create("build_config/input.txt"));
    assertThat(provider.getWorkingDirectory()).isEqualTo("build_config");

    PathFragment key = PathFragment.create("hello.txt");
    assertThat(provider.getTargets()).hasSize(1);
    assertThat(provider.getTargets()).containsKey(key);
    assertThat(provider.getTargets().get(key).getRuleName()).isEqualTo("echo");
  }
}
