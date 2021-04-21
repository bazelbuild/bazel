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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConstraintValueLookupUtil}. */
@RunWith(JUnit4.class)
public class ConstraintValueLookupUtilTest extends ToolchainTestCase {

  /**
   * An {@link AnalysisMock} that injects {@link GetConstraintValueInfoFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithGetPlatformInfoFunction extends AnalysisMock.Delegate {
    AnalysisMockWithGetPlatformInfoFunction() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(GET_CONSTRAINT_VALUE_INFO_FUNCTION, new GetConstraintValueInfoFunction())
          .build();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithGetPlatformInfoFunction();
  }

  @Test
  public void testConstraintValueLookup() throws Exception {
    ConfiguredTargetKey linuxKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//constraints:linux"))
            .setConfigurationKey(targetConfigKey)
            .build();
    ConfiguredTargetKey macKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//constraints:mac"))
            .setConfigurationKey(targetConfigKey)
            .build();
    GetConstraintValueInfoKey key =
        GetConstraintValueInfoKey.create(ImmutableList.of(linuxKey, macKey));

    EvaluationResult<GetConstraintValueInfoValue> result = getConstraintValueInfo(key);

    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(key).isNotNull();

    List<ConstraintValueInfo> constraintValues = result.get(key).constraintValues();
    assertThat(constraintValues).contains(linuxConstraint);
    assertThat(constraintValues).contains(macConstraint);
    assertThat(constraintValues).hasSize(2);
  }

  @Test
  public void testConstraintValueLookup_targetNotConstraintValue() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_constraint')");

    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//invalid:not_a_constraint"))
            .setConfigurationKey(targetConfigKey)
            .build();
    GetConstraintValueInfoKey key = GetConstraintValueInfoKey.create(ImmutableList.of(targetKey));

    EvaluationResult<GetConstraintValueInfoValue> result = getConstraintValueInfo(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidConstraintValueException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_constraint");
  }

  @Test
  public void testConstraintValueLookup_targetDoesNotExist() throws Exception {
    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseAbsoluteUnchecked("//fake:missing"))
            .setConfigurationKey(targetConfigKey)
            .build();
    GetConstraintValueInfoKey key = GetConstraintValueInfoKey.create(ImmutableList.of(targetKey));

    EvaluationResult<GetConstraintValueInfoValue> result = getConstraintValueInfo(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidConstraintValueException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no such package 'fake': BUILD file not found");
  }

  // Calls ConstraintValueLookupUtil.getConstraintValueInfo.
  private static final SkyFunctionName GET_CONSTRAINT_VALUE_INFO_FUNCTION =
      SkyFunctionName.createHermetic("GET_CONSTRAINT_VALUE_INFO_FUNCTION");

  @AutoValue
  abstract static class GetConstraintValueInfoKey implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return GET_CONSTRAINT_VALUE_INFO_FUNCTION;
    }

    abstract Iterable<ConfiguredTargetKey> constraintValueKeys();

    public static GetConstraintValueInfoKey create(
        Iterable<ConfiguredTargetKey> constraintValueKeys) {
      return new AutoValue_ConstraintValueLookupUtilTest_GetConstraintValueInfoKey(
          constraintValueKeys);
    }
  }

  EvaluationResult<GetConstraintValueInfoValue> getConstraintValueInfo(
      GetConstraintValueInfoKey key) throws InterruptedException {
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
  abstract static class GetConstraintValueInfoValue implements SkyValue {
    abstract List<ConstraintValueInfo> constraintValues();

    static GetConstraintValueInfoValue create(List<ConstraintValueInfo> constraintValues) {
      return new AutoValue_ConstraintValueLookupUtilTest_GetConstraintValueInfoValue(
          constraintValues);
    }
  }

  private static final class GetConstraintValueInfoFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      GetConstraintValueInfoKey key = (GetConstraintValueInfoKey) skyKey;
      try {
        List<ConstraintValueInfo> constraintValues =
            ConstraintValueLookupUtil.getConstraintValueInfo(key.constraintValueKeys(), env);
        if (env.valuesMissing()) {
          return null;
        }
        return GetConstraintValueInfoValue.create(constraintValues);
      } catch (InvalidConstraintValueException e) {
        throw new GetConstraintValueInfoFunctionException(e);
      }
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static class GetConstraintValueInfoFunctionException extends SkyFunctionException {
    public GetConstraintValueInfoFunctionException(InvalidConstraintValueException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
