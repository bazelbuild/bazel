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
import com.google.common.truth.IterableSubject;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RegisteredToolchainsFunction} and {@link RegisteredToolchainsValue}. */
@RunWith(JUnit4.class)
public class RegisteredToolchainsFunctionTest extends ToolchainTestCase {

  private EvaluationResult<RegisteredToolchainsValue> requestToolchainsFromSkyframe(
      SkyKey toolchainsKey) throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), toolchainsKey, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @Test
  public void testRegisteredToolchains() throws Exception {
    // Request the toolchains.
    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfig);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(toolchainsKey).isNotNull();

    RegisteredToolchainsValue value = result.get(toolchainsKey);
    assertThat(value.registeredToolchains()).hasSize(2);

    DeclaredToolchainInfo registeredToolchain1 = value.registeredToolchains().get(0);
    assertThat(registeredToolchain1).isNotNull();

    assertThat(registeredToolchain1.toolchainType()).isEqualTo(testToolchainType);
    assertThat(registeredToolchain1.execConstraints()).containsExactly(linuxConstraint);
    assertThat(registeredToolchain1.targetConstraints()).containsExactly(macConstraint);
    assertThat(registeredToolchain1.toolchainLabel())
        .isEqualTo(makeLabel("//toolchain:test_toolchain_1"));

    DeclaredToolchainInfo registeredToolchain2 = value.registeredToolchains().get(1);
    assertThat(registeredToolchain2).isNotNull();

    assertThat(registeredToolchain2.toolchainType()).isEqualTo(testToolchainType);
    assertThat(registeredToolchain2.execConstraints()).containsExactly(macConstraint);
    assertThat(registeredToolchain2.targetConstraints()).containsExactly(linuxConstraint);
    assertThat(registeredToolchain2.toolchainLabel())
        .isEqualTo(makeLabel("//toolchain:test_toolchain_2"));
  }

  @Test
  public void testRegisteredToolchains_flagOverride() throws Exception {

    // Add an extra toolchain.
    scratch.file(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain(",
        "    name = 'extra_toolchain',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = ['//constraint:linux'],",
        "    target_compatible_with = ['//constraint:linux'],",
        "    toolchain = ':extra_toolchain_impl')",
        "test_toolchain(",
        "  name='extra_toolchain_impl',",
        "  data = 'extra')");

    rewriteWorkspace("register_toolchains('//toolchain:toolchain_1')");
    useConfiguration("--extra_toolchains=//extra:extra_toolchain");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfig);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_toolchains flag is first in the list.
    assertToolchainLabels(result.get(toolchainsKey))
        .containsExactly(
            makeLabel("//extra:extra_toolchain_impl"), makeLabel("//toolchain:test_toolchain_1"))
        .inOrder();
  }

  @Test
  public void testRegisteredToolchains_notToolchain() throws Exception {
    rewriteWorkspace("register_toolchains(", "    '//error:not_a_toolchain')");
    scratch.file("error/BUILD", "filegroup(name = 'not_a_toolchain')");

    // Request the toolchains.
    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfig);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(toolchainsKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("target '//error:not_a_toolchain' does not provide a toolchain");
  }

  @Test
  public void testRegisteredToolchains_reload() throws Exception {
    rewriteWorkspace("register_toolchains('//toolchain:toolchain_1')");

    SkyKey toolchainsKey = RegisteredToolchainsValue.key(targetConfig);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey))
        .containsExactly(makeLabel("//toolchain:test_toolchain_1"));

    // Re-write the WORKSPACE.
    rewriteWorkspace("register_toolchains('//toolchain:toolchain_2')");

    toolchainsKey = RegisteredToolchainsValue.key(targetConfig);
    result = requestToolchainsFromSkyframe(toolchainsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertToolchainLabels(result.get(toolchainsKey))
        .containsExactly(makeLabel("//toolchain:test_toolchain_2"));
  }

  @Test
  public void testRegisteredToolchainsValue_equalsAndHashCode() {
    DeclaredToolchainInfo toolchain1 =
        DeclaredToolchainInfo.create(
            makeLabel("//test:toolchain"),
            ImmutableList.of(),
            ImmutableList.of(),
            makeLabel("//test/toolchain_impl_1"));
    DeclaredToolchainInfo toolchain2 =
        DeclaredToolchainInfo.create(
            makeLabel("//test:toolchain"),
            ImmutableList.of(),
            ImmutableList.of(),
            makeLabel("//test/toolchain_impl_2"));

    new EqualsTester()
        .addEqualityGroup(
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain1, toolchain2)),
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain1, toolchain2)))
        .addEqualityGroup(
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain1)),
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain2)),
            RegisteredToolchainsValue.create(ImmutableList.of(toolchain2, toolchain1)));
  }

  private static IterableSubject assertToolchainLabels(
      RegisteredToolchainsValue registeredToolchainsValue) {
    assertThat(registeredToolchainsValue).isNotNull();
    ImmutableList<DeclaredToolchainInfo> declaredToolchains =
        registeredToolchainsValue.registeredToolchains();
    List<Label> labels = collectToolchainLabels(declaredToolchains);
    return assertThat(labels);
  }

  private static List<Label> collectToolchainLabels(List<DeclaredToolchainInfo> toolchains) {
    return toolchains
        .stream()
        .map((toolchain -> toolchain.toolchainLabel()))
        .collect(Collectors.toList());
  }
}
