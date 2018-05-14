// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.platform.PlatformInfo.DuplicateConstraintException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RegisteredExecutionPlatformsFunction} and {@link
 * RegisteredExecutionPlatformsValue}.
 */
@RunWith(JUnit4.class)
public class RegisteredExecutionPlatformsFunctionTest extends ToolchainTestCase {

  protected EvaluationResult<RegisteredExecutionPlatformsValue>
      requestExecutionPlatformsFromSkyframe(SkyKey executionPlatformsKey)
          throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), executionPlatformsKey, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  protected static IterableSubject assertExecutionPlatformLabels(
      RegisteredExecutionPlatformsValue registeredExecutionPlatformsValue) {
    assertThat(registeredExecutionPlatformsValue).isNotNull();
    ImmutableList<ConfiguredTargetKey> declaredExecutionPlatformKeys =
        registeredExecutionPlatformsValue.registeredExecutionPlatformKeys();
    List<Label> labels = collectExecutionPlatformLabels(declaredExecutionPlatformKeys);
    return assertThat(labels);
  }

  protected static List<Label> collectExecutionPlatformLabels(
      List<ConfiguredTargetKey> executionPlatformKeys) {
    return executionPlatformKeys
        .stream()
        .map(ConfiguredTargetKey::getLabel)
        .collect(Collectors.toList());
  }

  @Test
  public void testRegisteredExecutionPlatforms() throws Exception {
    // Request the executionPlatforms.
    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(executionPlatformsKey).isNotNull();

    RegisteredExecutionPlatformsValue value = result.get(executionPlatformsKey);
    assertThat(value.registeredExecutionPlatformKeys()).isEmpty();
  }

  @Test
  public void testRegisteredExecutionPlatforms_flagOverride() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//extra:execution_platform_2')");
    useConfiguration("--extra_execution_platforms=//extra:execution_platform_1");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAllOf(
            makeLabel("//extra:execution_platform_1"), makeLabel("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_workspace() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//extra/...')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAllOf(
            makeLabel("//extra:execution_platform_1"), makeLabel("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_targetPattern_flagOverride() throws Exception {

    // Add an extra execution platform.
    scratch.file(
        "extra/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    useConfiguration("--extra_execution_platforms=//extra/...");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();

    // Verify that the target registered with the extra_execution_platforms flag is first in the
    // list.
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .containsAllOf(
            makeLabel("//extra:execution_platform_1"), makeLabel("//extra:execution_platform_2"))
        .inOrder();
  }

  @Test
  public void testRegisteredExecutionPlatforms_notExecutionPlatform() throws Exception {
    rewriteWorkspace("register_execution_platforms(", "    '//error:not_an_execution_platform')");
    scratch.file("error/BUILD", "filegroup(name = 'not_an_execution_platform')");

    // Request the executionPlatforms.
    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(executionPlatformsKey)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(executionPlatformsKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//error:not_an_execution_platform");
  }

  @Test
  public void testRegisteredExecutionPlatforms_reload() throws Exception {
    scratch.file(
        "platform/BUILD",
        "platform(name = 'execution_platform_1')",
        "platform(name = 'execution_platform_2')");

    rewriteWorkspace("register_execution_platforms('//platform:execution_platform_1')");

    SkyKey executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    EvaluationResult<RegisteredExecutionPlatformsValue> result =
        requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .contains(makeLabel("//platform:execution_platform_1"));

    // Re-write the WORKSPACE.
    rewriteWorkspace("register_execution_platforms('//platform:execution_platform_2')");

    executionPlatformsKey = RegisteredExecutionPlatformsValue.key(targetConfigKey);
    result = requestExecutionPlatformsFromSkyframe(executionPlatformsKey);
    assertThatEvaluationResult(result).hasNoError();
    assertExecutionPlatformLabels(result.get(executionPlatformsKey))
        .contains(makeLabel("//platform:execution_platform_2"));
  }

  @Test
  public void testRegisteredExecutionPlatformsValue_equalsAndHashCode()
      throws DuplicateConstraintException {
    ConfiguredTargetKey executionPlatformKey1 =
        ConfiguredTargetKey.of(makeLabel("//test:executionPlatform1"), null, false);
    ConfiguredTargetKey executionPlatformKey2 =
        ConfiguredTargetKey.of(makeLabel("//test:executionPlatform2"), null, false);

    new EqualsTester()
        .addEqualityGroup(
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey1, executionPlatformKey2)),
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey1, executionPlatformKey2)))
        .addEqualityGroup(
            RegisteredExecutionPlatformsValue.create(ImmutableList.of(executionPlatformKey1)),
            RegisteredExecutionPlatformsValue.create(ImmutableList.of(executionPlatformKey2)),
            RegisteredExecutionPlatformsValue.create(
                ImmutableList.of(executionPlatformKey2, executionPlatformKey1)));
  }
}
