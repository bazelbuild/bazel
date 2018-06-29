package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupFunction.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConstraintValueLookupFunction}. */
@RunWith(JUnit4.class)
public class ConstraintValueLookupFunctionTest extends ToolchainTestCase {

  @Test
  public void testConstraintValueLookup() throws Exception {
    ConfiguredTargetKey linuxKey =
        ConfiguredTargetKey.of(makeLabel("//constraints:linux"), targetConfigKey, false);
    ConfiguredTargetKey macKey =
        ConfiguredTargetKey.of(makeLabel("//constraints:mac"), targetConfigKey, false);
    ConstraintValueLookupValue.Key key = ConstraintValueLookupValue.key(ImmutableList.of(linuxKey, macKey));

    EvaluationResult<ConstraintValueLookupValue> result = requestConstraintValuesFromSkyframe(key);

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
        ConfiguredTargetKey.of(makeLabel("//invalid:not_a_constraint"), targetConfigKey, false);
    ConstraintValueLookupValue.Key key = ConstraintValueLookupValue.key(ImmutableList.of(targetKey));

    EvaluationResult<ConstraintValueLookupValue> result = requestConstraintValuesFromSkyframe(key);

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
        ConfiguredTargetKey.of(makeLabel("//fake:missing"), targetConfigKey, false);
    ConstraintValueLookupValue.Key key = ConstraintValueLookupValue.key(ImmutableList.of(targetKey));

    EvaluationResult<ConstraintValueLookupValue> result = requestConstraintValuesFromSkyframe(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidConstraintValueException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//fake:missing");
  }

  protected EvaluationResult<ConstraintValueLookupValue> requestConstraintValuesFromSkyframe(SkyKey key)
      throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }
}
