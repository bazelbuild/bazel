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

import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
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
        .isSameInstanceAs(getConfiguredTarget("//foo:generating_rule", getHostConfiguration()));
  }

  /**
   * Bazel maintains a host configuration cache that stores configurations instantiated outside of
   * Skyframe. We shouldn't, in general, instantiate configurations outside of
   * {@link BuildConfigurationValue}. But in this specific case Bazel performance suffers through
   * the Skyframe interface and maintaining a local cache is much faster.
   * See {@link SkyframeBuildView} for details.
   *
   * <p>One consequence of this is that requesting a config that would be a Skyframe cache hit
   * can still produce a distinct instance. Meaning you can get cases where {@code
   * config1.equals(config2) && config1 != config2}.
   *
   * <p>This test checks for such a case: perform three consecutive Bazel builds. The first builds
   * with default options, producing top-level host config H1 and configured target
   * <host_generated_file_producer, H1>. The second builds with {@code --host_copt=a=b},
   * producing host config H2 (and clearing the top-level host config cache since the host config
   * changed). The third builds back with default options. This once again clears the host config
   * cache, since the host config changed again. So that cache creates a new config H3 where
   * H3.equals(H1) and instantiates new configured target <host_src3.cc, H3>. It then requests
   * dependency <host_generated_file_producer, H3> from Skyframe, but the Skyframe SkyKey interner
   * reduces this to the previously seen <host_generated_file_producer, H1> and returns that
   * instead.
   *
   * <p>This produces the expected scenario where the output file's config is value-equal but not
   * reference-equal to its generating rule's config.
   */
  @Test
  public void hostConfigSwitch() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'host_generated_file_producer',",
        "    srcs = [],",
        "    outs = [",
        "        'host_src1.cc',",
        "        'host_src2.cc',",
        "        'host_src3.cc',",
        "    ],",
        "    cmd = 'echo hi > $(location host_src1.cc); echo hi > $(location host_src2.cc); "
            + "echo hi > $(location host_src3.cc)')",
        "",
        "cc_binary(name = 'host_generated_file_consumer1', srcs = ['host_src1.cc'])",
        "cc_binary(name = 'host_generated_file_consumer2', srcs = ['host_src2.cc'])",
        "cc_binary(name = 'host_generated_file_consumer3', srcs = ['host_src3.cc'])",
        "",
        "genrule(name = 'gen1', srcs = [], outs = ['gen1.out'], cmd = 'echo hi > $@',",
        "    tools = [':host_generated_file_consumer1'])",
        "genrule(name = 'gen2', srcs = [], outs = ['gen2.out'], cmd = 'echo hi > $@',",
        "    tools = [':host_generated_file_consumer2'])",
        "genrule(name = 'gen3', srcs = [], outs = ['gen3.out'], cmd = 'echo hi > $@',",
        "    tools = [':host_generated_file_consumer3'])");

    useConfiguration();
    update("//foo:gen1");
    useConfiguration("--host_copt", "a=b");
    update("//foo:gen2");
    useConfiguration();
    update("//foo:gen3");

    ConfiguredTargetAndData hostSrc3 =
        getConfiguredTargetAndData("//foo:host_src3.cc", getHostConfiguration());
    ConfiguredTarget hostGeneratedFileConsumer3 =
        ((OutputFileConfiguredTarget) hostSrc3.getConfiguredTarget()).getGeneratingRule();
    assertThat(hostSrc3.getConfiguration()).isEqualTo(getConfiguration(hostGeneratedFileConsumer3));
    // TODO(gregce): enable below for Bazel tests, which for some reason realize the same instance
//    assertThat(hostSrc3.getConfiguration())
//        .isNotSameAs(hostGeneratedFileConsumer3.getConfiguration());
  }
}
