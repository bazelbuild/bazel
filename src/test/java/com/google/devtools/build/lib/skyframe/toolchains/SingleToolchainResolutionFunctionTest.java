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

package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SingleToolchainResolutionValue} and {@link SingleToolchainResolutionFunction}.
 */
@RunWith(JUnit4.class)
public class SingleToolchainResolutionFunctionTest extends ToolchainTestCase {
  ConfiguredTargetKey linuxCtkey;
  ConfiguredTargetKey macCtkey;

  @Before
  public void setUpKeys() {
    // This has to happen here so that targetConfiguration is populated.
    linuxCtkey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:linux"))
            .setConfiguration(getTargetConfiguration())
            .build();
    macCtkey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:mac"))
            .setConfiguration(getTargetConfiguration())
            .build();
  }

  private EvaluationResult<SingleToolchainResolutionValue> invokeToolchainResolution(SkyKey key)
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
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels())
        .containsExactly(macCtkey, Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"));
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
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(linuxCtkey, macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels())
        .containsExactly(
            linuxCtkey,
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_impl"),
            macCtkey,
            Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testResolution_noneFound() throws Exception {
    // Clear the toolchains.
    rewriteWorkspace();

    SkyKey key =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels()).isEmpty();
  }

  @Test
  public void testToolchainResolutionValue_equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))),
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))))
        // Different execution platform, same label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))))
        // Same execution platform, different label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_2"))))
        // Different execution platform, different label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_2"))))
        // Multiple execution platforms.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.<ConfiguredTargetKey, Label>builder()
                    .put(linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))
                    .put(macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))
                    .buildOrThrow()))
        .testEquals();
  }
}
