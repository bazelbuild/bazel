// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link OutputFileConfiguredTarget}.
 */
@RunWith(JUnit4.class)
public class OutputFileConfiguredTargetTest extends BuildViewTestBase {
  @Test
  public void generatingRuleIsCorrect() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(",
        "    name='generating_rule', ",
        "    cmd = 'echo hi > $@',",
        "    srcs = [],",
        "    outs = ['generated.source'])");
    update("//foo:generating_rule");
    OutputFileConfiguredTarget generatedSource = (OutputFileConfiguredTarget)
        getConfiguredTarget("//foo:generated.source", getHostConfiguration());
    assertThat(generatedSource.getGeneratingRule())
        .isSameAs(getConfiguredTarget("//foo:generating_rule", getHostConfiguration()));
  }

  /**
   * This regression-tests the fix for a bug in which Bazel could crash with a NullPointerException
   * when two builds were invoked with different configurations because an output file ended up
   * unexpectedly having a null generating rule.
   *
   * <p>The reasons for this are subtle and complex. In short, the bug only happens when
   * --experimental_dynamic_configs=off and BuildConfiguration.equals provides value equality.
   *
   * <p>In that scenario, when we call {@code bazel build //foo:gen1 --copt a=b}, Bazel creates a
   * host configuration H1 for this build that attaches to {@code
   * //foo:host_generated_file_producer} (which is the generator of {@code host_src1.cc}).
   *
   * <p>When we then call {@code bazel build //foo:gen2 --copt a=c}, the options have changed so
   * Bazel clears all configurations and configured targets. This includes the host configuration,
   * even though - importantly - none of its options have changed. So Bazel uses a new host config
   * H2 that's value-equal to H1 (but not reference-equal) and assigns it to {@code host_src2.cc}.
   *
   * <p>Here's the problem: during configured target analysis, Bazel creates the configured target
   * <host_src2.cc, H2>, then requests from Skyframe the dependency with SkyKey
   * <host_generated_file_producer, H2>. But Skyframe uses an interner to reference-collapse
   * equivalent keys (SkyKey.SKY_KEY_INTERNER}). And that doesn't get cleared between builds. So
   * the actual SkyKey used is <host_generated_file_producer, H1>, which produces a ConfiguredTarget
   * with config=H1. This breaks {@link TargetContext#findDirectPrerequisite}, which output files
   * use to find their generating rules. That method looks for
   * {@code label.equals("host_generated_file_producer")} and {@code config == H2}, which finds
   * no match so returns a null reference.
   *
   * <p>When configs only use reference equality, the problem goes away because the interner can no
   * longer safely merge H1 and H2. An alternative fix might be to change
   * {@link TargetContext#findDirectPrerequisite} to check {@code config.equals(H2)} instead of
   * {@code config == H2}. But this isn't really safe because static configs embed references to
   * other configs in their transition table. So returning H1 when you expect H2 creates the
   * possibility of "leaking out" to the wrong configuration during a transition, even if H1 and H2
   * have the same exact build options.
   *
   * <p>All of these problems go away with dynamic configurations. This is for two reasons: 1)
   * dynamic configs don't embed transition tables, so "leaking out" is no longer possible. And
   * 2) a dynamic config is evaluated through Skyframe keyed on its BuildOptions, so semantic
   * equality implies reference equality anyway, 2 may change in the future, but if/when it does
   * this test should reliably fail so appropriate measures can be taken.
   */
  @Test
  public void regressionTestForStaticConfigsWithValueEquality() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'host_generated_file_producer',",
        "    srcs = [],",
        "    outs = [",
        "        'host_src1.cc',",
        "        'host_src2.cc',",
        "    ],",
        "    cmd = 'echo hi > $(location host_src1.cc); echo hi > $(location host_src2.cc)')",
        "",
        "cc_binary(",
        "    name = 'host_generated_file_consumer1',",
        "    srcs = ['host_src1.cc'])",
        "",
        "cc_binary(",
        "    name = 'host_generated_file_consumer2',",
        "    srcs = ['host_src2.cc'])",
        "",
        "genrule(",
        "    name = 'gen1',",
        "    srcs = [],",
        "    outs = ['gen1.out'],",
        "    cmd = 'echo hi > $@',",
        "    tools = [':host_generated_file_consumer1'])",
        "",
        "genrule(",
        "    name = 'gen2',",
        "    srcs = [],",
        "    outs = ['gen2.out'],",
        "    cmd = 'echo hi > $@',",
        "    tools = [':host_generated_file_consumer2'])"
    );
    useConfiguration("--copt", "a=b");
    update("//foo:gen1");
    useConfiguration("--copt", "a=c");
    update("//foo:gen2");
    OutputFileConfiguredTarget hostSrc2 = (OutputFileConfiguredTarget)
        getConfiguredTarget("//foo:host_src2.cc", getHostConfiguration());
    assertThat(hostSrc2.getGeneratingRule()).isNotNull();
  }
}
