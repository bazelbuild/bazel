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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ToolchainTypeLookupUtil.InvalidToolchainTypeException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ToolchainTypeLookupUtil}. */
@RunWith(JUnit4.class)
public class ToolchainTypeLookupUtilTest extends ToolchainTestCase {

  /**
   * An {@link AnalysisMock} that injects {@link GetToolchainTypeInfoFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithGetToolchainTypeInfoFunction
      extends AnalysisMock.Delegate {
    AnalysisMockWithGetToolchainTypeInfoFunction() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(GET_TOOLCHAIN_TYPE_INFO_FUNCTION, new GetToolchainTypeInfoFunction())
          .build();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithGetToolchainTypeInfoFunction();
  }

  @Test
  public void testToolchainTypeLookup() throws Exception {
    ConfiguredTargetKey testToolchainTypeKey =
        ConfiguredTargetKey.builder()
            .setLabel(testToolchainTypeLabel)
            .setConfigurationKey(targetConfigKey)
            .build();
    GetToolchainTypeInfoKey key =
        GetToolchainTypeInfoKey.create(ImmutableList.of(testToolchainTypeKey));

    EvaluationResult<GetToolchainTypeInfoValue> result = getToolchainTypeInfo(key);

    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(key).isNotNull();

    Map<Label, ToolchainTypeInfo> toolchainTypes = result.get(key).toolchainTypes();
    assertThat(toolchainTypes)
        .containsExactlyEntriesIn(ImmutableMap.of(testToolchainTypeLabel, testToolchainType));
  }

  @Test
  public void testToolchainTypeLookup_toolchainAlias() throws Exception {
    scratch.file(
        "alias/BUILD", "alias(name = 'toolchain_type', actual = '" + testToolchainTypeLabel + "')");
    Label aliasToolchainTypeLabel = makeLabel("//alias:toolchain_type");
    ConfiguredTargetKey testToolchainTypeKey =
        ConfiguredTargetKey.builder()
            .setLabel(aliasToolchainTypeLabel)
            .setConfigurationKey(targetConfigKey)
            .build();
    GetToolchainTypeInfoKey key =
        GetToolchainTypeInfoKey.create(ImmutableList.of(testToolchainTypeKey));

    EvaluationResult<GetToolchainTypeInfoValue> result = getToolchainTypeInfo(key);

    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(key).isNotNull();

    Map<Label, ToolchainTypeInfo> toolchainTypes = result.get(key).toolchainTypes();
    assertThat(toolchainTypes)
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                testToolchainTypeLabel,
                testToolchainType,
                aliasToolchainTypeLabel,
                testToolchainType));
  }

  @Test
  public void testToolchainTypeLookup_targetNotToolchainType() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_toolchain_type')");

    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder()
            .setLabel(makeLabel("//invalid:not_a_toolchain_type"))
            .setConfigurationKey(targetConfigKey)
            .build();
    GetToolchainTypeInfoKey key = GetToolchainTypeInfoKey.create(ImmutableList.of(targetKey));

    EvaluationResult<GetToolchainTypeInfoValue> result = getToolchainTypeInfo(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidToolchainTypeException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_toolchain_type");
  }

  @Test
  public void testToolchainTypeLookup_targetDoesNotExist() throws Exception {
    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder()
            .setLabel(makeLabel("//fake:missing"))
            .setConfigurationKey(targetConfigKey)
            .build();
    GetToolchainTypeInfoKey key = GetToolchainTypeInfoKey.create(ImmutableList.of(targetKey));

    EvaluationResult<GetToolchainTypeInfoValue> result = getToolchainTypeInfo(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidToolchainTypeException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no such package 'fake': BUILD file not found");
  }

  // Calls ToolchainTypeLookupUtil.getToolchainTypeInfo.
  private static final SkyFunctionName GET_TOOLCHAIN_TYPE_INFO_FUNCTION =
      SkyFunctionName.createHermetic("GET_TOOLCHAIN_TYPE_INFO_FUNCTION");

  @AutoValue
  abstract static class GetToolchainTypeInfoKey implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return GET_TOOLCHAIN_TYPE_INFO_FUNCTION;
    }

    abstract Iterable<ConfiguredTargetKey> toolchainTypeKeys();

    public static GetToolchainTypeInfoKey create(Iterable<ConfiguredTargetKey> toolchainTypeKeys) {
      return new AutoValue_ToolchainTypeLookupUtilTest_GetToolchainTypeInfoKey(toolchainTypeKeys);
    }
  }

  EvaluationResult<GetToolchainTypeInfoValue> getToolchainTypeInfo(GetToolchainTypeInfoKey key)
      throws InterruptedException {
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
  abstract static class GetToolchainTypeInfoValue implements SkyValue {
    abstract Map<Label, ToolchainTypeInfo> toolchainTypes();

    static GetToolchainTypeInfoValue create(Map<Label, ToolchainTypeInfo> toolchainTypes) {
      return new AutoValue_ToolchainTypeLookupUtilTest_GetToolchainTypeInfoValue(toolchainTypes);
    }
  }

  private static final class GetToolchainTypeInfoFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      GetToolchainTypeInfoKey key = (GetToolchainTypeInfoKey) skyKey;
      try {
        Map<Label, ToolchainTypeInfo> toolchainTypes =
            ToolchainTypeLookupUtil.resolveToolchainTypes(env, key.toolchainTypeKeys(), false);
        if (env.valuesMissing()) {
          return null;
        }
        return GetToolchainTypeInfoValue.create(toolchainTypes);
      } catch (InvalidToolchainTypeException e) {
        throw new GetToolchainTypeInfoFunctionException(e);
      }
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static class GetToolchainTypeInfoFunctionException extends SkyFunctionException {
    public GetToolchainTypeInfoFunctionException(InvalidToolchainTypeException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
