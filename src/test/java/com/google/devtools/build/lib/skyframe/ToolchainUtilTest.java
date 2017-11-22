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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.ToolchainContextException;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.UnresolvedToolchainsException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ToolchainUtil}. */
@RunWith(JUnit4.class)
public class ToolchainUtilTest extends ToolchainTestCase {

  /**
   * An {@link AnalysisMock} that injects {@link CreateToolchainContextFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithCreateToolchainContextFunction
      extends AnalysisMock.Delegate {
    AnalysisMockWithCreateToolchainContextFunction() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(CREATE_TOOLCHAIN_CONTEXT_FUNCTION, new CreateToolchainContextFunction())
          .build();
    }
  };

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithCreateToolchainContextFunction();
  }

  @Test
  public void createToolchainContext() throws Exception {
    useConfiguration(
        "--experimental_host_platform=//platforms:linux",
        "--experimental_platforms=//platforms:mac");
    CreateToolchainContextKey key =
        CreateToolchainContextKey.create("test", ImmutableSet.of(testToolchainType), targetConfig);

    EvaluationResult<CreateToolchainContextValue> result = createToolchainContext(key);

    assertThatEvaluationResult(result).hasNoError();
    ToolchainContext toolchainContext = result.get(key).toolchainContext();
    assertThat(toolchainContext).isNotNull();

    assertThat(toolchainContext.getRequiredToolchains()).containsExactly(testToolchainType);
    assertThat(toolchainContext.getResolvedToolchainLabels())
        .containsExactly(Label.parseAbsoluteUnchecked("//toolchain:test_toolchain_1"));

    assertThat(toolchainContext.getExecutionPlatform()).isNotNull();
    assertThat(toolchainContext.getExecutionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));

    assertThat(toolchainContext.getTargetPlatform()).isNotNull();
    assertThat(toolchainContext.getTargetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:mac"));
  }

  @Test
  public void createToolchainContext_unavailableToolchainType_single() throws Exception {
    useConfiguration(
        "--experimental_host_platform=//platforms:linux",
        "--experimental_platforms=//platforms:mac");
    CreateToolchainContextKey key =
        CreateToolchainContextKey.create(
            "test",
            ImmutableSet.of(
                testToolchainType, Label.parseAbsoluteUnchecked("//fake/toolchain:type_1")),
            targetConfig);

    EvaluationResult<CreateToolchainContextValue> result = createToolchainContext(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(ToolchainContextException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasCauseThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasCauseThat()
        .hasMessageThat()
        .contains("no matching toolchains found for types //fake/toolchain:type_1");
  }

  @Test
  public void createToolchainContext_unavailableToolchainType_multiple() throws Exception {
    useConfiguration(
        "--experimental_host_platform=//platforms:linux",
        "--experimental_platforms=//platforms:mac");
    CreateToolchainContextKey key =
        CreateToolchainContextKey.create(
            "test",
            ImmutableSet.of(
                testToolchainType,
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_1"),
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_2")),
            targetConfig);

    EvaluationResult<CreateToolchainContextValue> result = createToolchainContext(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(ToolchainContextException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasCauseThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    // Only one of the missing types will be reported, so do not check the specific error message.
  }

  // Calls ToolchainUtil.createToolchainContext.
  private static final SkyFunctionName CREATE_TOOLCHAIN_CONTEXT_FUNCTION =
      SkyFunctionName.create("CREATE_TOOLCHAIN_CONTEXT_FUNCTION");

  @AutoValue
  abstract static class CreateToolchainContextKey implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return CREATE_TOOLCHAIN_CONTEXT_FUNCTION;
    }

    abstract String targetDescription();

    abstract Set<Label> requiredToolchains();

    abstract BuildConfiguration configuration();

    public static CreateToolchainContextKey create(
        String targetDescription, Set<Label> requiredToolchains, BuildConfiguration configuration) {
      return new AutoValue_ToolchainUtilTest_CreateToolchainContextKey(
          targetDescription, requiredToolchains, configuration);
    }
  }

  EvaluationResult<CreateToolchainContextValue> createToolchainContext(
      CreateToolchainContextKey key) throws InterruptedException {
    try {
      // Must re-enable analysis for Skyframe functions that create configured targets.
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    } finally {
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @AutoValue
  abstract static class CreateToolchainContextValue implements SkyValue {
    abstract ToolchainContext toolchainContext();

    static CreateToolchainContextValue create(ToolchainContext toolchainContext) {
      return new AutoValue_ToolchainUtilTest_CreateToolchainContextValue(toolchainContext);
    }
  }

  private static final class CreateToolchainContextFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      CreateToolchainContextKey key = (CreateToolchainContextKey) skyKey;
      ToolchainContext toolchainContext = null;
      try {
        toolchainContext =
            ToolchainUtil.createToolchainContext(
                env, key.targetDescription(), key.requiredToolchains(), key.configuration());
        if (toolchainContext == null) {
          return null;
        }
        return CreateToolchainContextValue.create(toolchainContext);
      } catch (ToolchainContextException e) {
        throw new CreateToolchainContextFunctionException(e);
      }
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static class CreateToolchainContextFunctionException extends SkyFunctionException {
    public CreateToolchainContextFunctionException(ToolchainContextException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
