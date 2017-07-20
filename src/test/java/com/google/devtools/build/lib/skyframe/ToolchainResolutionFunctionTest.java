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
import com.google.common.testing.EqualsTester;
import com.google.common.truth.DefaultSubject;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
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
  public void testResolution() throws Exception {
    SkyKey key =
        ToolchainResolutionValue.key(targetConfig, testToolchainType, linuxPlatform, macPlatform);
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    ToolchainResolutionValue toolchainResolutionValue = result.get(key);
    assertThat(toolchainResolutionValue.toolchainLabel())
        .isEqualTo(makeLabel("//toolchain:test_toolchain_2"));
  }

  @Test
  public void testResolution_flagOverride() throws Exception {
    // Add extra toolchain.
    scratch.file(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "test_toolchain(",
        "  name='extra_toolchain_impl',",
        "  data = 'extra')");

    useConfiguration(
        "--toolchain_resolution_override=" + testToolchainType + "=//extra:extra_toolchain_impl");

    SkyKey key =
        ToolchainResolutionValue.key(targetConfig, testToolchainType, linuxPlatform, macPlatform);
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    ToolchainResolutionValue toolchainResolutionValue = result.get(key);
    assertThat(toolchainResolutionValue.toolchainLabel())
        .isEqualTo(makeLabel("//extra:extra_toolchain_impl"));
  }

  @Test
  public void testResolution_noneFound() throws Exception {
    // Clear the toolchains.
    rewriteWorkspace();

    SkyKey key =
        ToolchainResolutionValue.key(targetConfig, testToolchainType, linuxPlatform, macPlatform);
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no matching toolchain found for //toolchain:test_toolchain");
  }

  @Test
  public void testResolveConstraints() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(makeLabel("//constraint:setting1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(makeLabel("//constraint:setting2"));
    ConstraintValueInfo constraint1a =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:value1a"));
    ConstraintValueInfo constraint1b =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:value1b"));
    ConstraintValueInfo constraint2a =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value2a"));
    ConstraintValueInfo constraint2b =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value2b"));

    Label toolchainType1 = makeLabel("//toolchain:type1");
    Label toolchainType2 = makeLabel("//toolchain:type2");

    DeclaredToolchainInfo toolchain1a =
        DeclaredToolchainInfo.create(
            toolchainType1,
            ImmutableList.of(constraint1a, constraint2a),
            ImmutableList.of(constraint1a, constraint2a),
            makeLabel("//toolchain:toolchain1a"));
    DeclaredToolchainInfo toolchain1b =
        DeclaredToolchainInfo.create(
            toolchainType1,
            ImmutableList.of(constraint1a, constraint2b),
            ImmutableList.of(constraint1a, constraint2b),
            makeLabel("//toolchain:toolchain1b"));
    DeclaredToolchainInfo toolchain2a =
        DeclaredToolchainInfo.create(
            toolchainType2,
            ImmutableList.of(constraint1b, constraint2a),
            ImmutableList.of(constraint1b, constraint2a),
            makeLabel("//toolchain:toolchain2a"));
    DeclaredToolchainInfo toolchain2b =
        DeclaredToolchainInfo.create(
            toolchainType2,
            ImmutableList.of(constraint1b, constraint2b),
            ImmutableList.of(constraint1b, constraint2b),
            makeLabel("//toolchain:toolchain2b"));

    ImmutableList<DeclaredToolchainInfo> allToolchains =
        ImmutableList.of(toolchain1a, toolchain1b, toolchain2a, toolchain2b);

    assertToolchainResolution(
            toolchainType1,
            ImmutableList.of(constraint1a, constraint2a),
            ImmutableList.of(constraint1a, constraint2a),
            allToolchains)
        .isEqualTo(toolchain1a);
    assertToolchainResolution(
            toolchainType1,
            ImmutableList.of(constraint1a, constraint2b),
            ImmutableList.of(constraint1a, constraint2b),
            allToolchains)
        .isEqualTo(toolchain1b);
    assertToolchainResolution(
            toolchainType2,
            ImmutableList.of(constraint1b, constraint2a),
            ImmutableList.of(constraint1b, constraint2a),
            allToolchains)
        .isEqualTo(toolchain2a);
    assertToolchainResolution(
            toolchainType2,
            ImmutableList.of(constraint1b, constraint2b),
            ImmutableList.of(constraint1b, constraint2b),
            allToolchains)
        .isEqualTo(toolchain2b);

    // No toolchains of type.
    assertToolchainResolution(
            makeLabel("//toolchain:type3"),
            ImmutableList.of(constraint1a, constraint2a),
            ImmutableList.of(constraint1a, constraint2a),
            allToolchains)
        .isNull();
  }

  private Subject<DefaultSubject, Object> assertToolchainResolution(
      Label toolchainType,
      Iterable<ConstraintValueInfo> targetConstraints,
      Iterable<ConstraintValueInfo> execConstraints,
      ImmutableList<DeclaredToolchainInfo> toolchains)
      throws Exception {

    PlatformInfo execPlatform =
        PlatformInfo.builder()
            .setLabel(makeLabel("//platform:exec"))
            .addConstraints(execConstraints)
            .build();
    PlatformInfo targetPlatform =
        PlatformInfo.builder()
            .setLabel(makeLabel("//platform:target"))
            .addConstraints(targetConstraints)
            .build();

    DeclaredToolchainInfo resolvedToolchain =
        ToolchainResolutionFunction.resolveConstraints(
            toolchainType, execPlatform, targetPlatform, toolchains);
    return assertThat(resolvedToolchain);
  }

  @Test
  public void testToolchainResolutionValue_equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_1")),
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_1")))
        .addEqualityGroup(
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_2")),
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_2")));
  }
}
