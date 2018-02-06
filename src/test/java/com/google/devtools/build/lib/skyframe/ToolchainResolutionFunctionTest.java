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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ToolchainResolutionValue} and {@link ToolchainResolutionFunction}. */
@RunWith(JUnit4.class)
public class ToolchainResolutionFunctionTest extends ToolchainTestCase {

  private EvaluationResult<ToolchainResolutionValue> invokeToolchainResolution(SkyKey key)
      throws InterruptedException {

    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @Test
  public void testResolution_singleExecutionPlatform() throws Exception {
    SkyKey key =
        ToolchainResolutionValue.key(
            targetConfig, testToolchainType, linuxPlatform, ImmutableList.of(macPlatform));
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    ToolchainResolutionValue toolchainResolutionValue = result.get(key);
    assertThat(toolchainResolutionValue.availableToolchainLabels())
        .containsExactly(macPlatform, makeLabel("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testResolution_multipleExecutionPlatforms() throws Exception {
    addToolchain(
        "extra",
        "extra_toolchain",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace(
        "register_toolchains(",
        "'//toolchain:toolchain_1',",
        "'//toolchain:toolchain_2',",
        "'//extra:extra_toolchain')");

    SkyKey key =
        ToolchainResolutionValue.key(
            targetConfig,
            testToolchainType,
            linuxPlatform,
            ImmutableList.of(linuxPlatform, macPlatform));
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    ToolchainResolutionValue toolchainResolutionValue = result.get(key);
    assertThat(toolchainResolutionValue.availableToolchainLabels())
        .containsExactly(
            linuxPlatform,
            makeLabel("//extra:extra_toolchain_impl"),
            macPlatform,
            makeLabel("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testResolution_noneFound() throws Exception {
    // Clear the toolchains.
    rewriteWorkspace();

    SkyKey key =
        ToolchainResolutionValue.key(
            targetConfig, testToolchainType, linuxPlatform, ImmutableList.of(macPlatform));
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no matching toolchain found for //toolchain:test_toolchain");
  }

  @Test
  public void testToolchainResolutionValue_equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(linuxPlatform, makeLabel("//test:toolchain_impl_1"))
                    .build()),
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(linuxPlatform, makeLabel("//test:toolchain_impl_1"))
                    .build()))
        // Different execution platform, same label.
        .addEqualityGroup(
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(macPlatform, makeLabel("//test:toolchain_impl_1"))
                    .build()))
        // Same execution platform, different label.
        .addEqualityGroup(
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(linuxPlatform, makeLabel("//test:toolchain_impl_2"))
                    .build()))
        // Different execution platform, different label.
        .addEqualityGroup(
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(macPlatform, makeLabel("//test:toolchain_impl_2"))
                    .build()))
        // Multiple execution platforms.
        .addEqualityGroup(
            ToolchainResolutionValue.create(
                ImmutableMap.<PlatformInfo, Label>builder()
                    .put(linuxPlatform, makeLabel("//test:toolchain_impl_1"))
                    .put(macPlatform, makeLabel("//test:toolchain_impl_1"))
                    .build()));
  }
}
