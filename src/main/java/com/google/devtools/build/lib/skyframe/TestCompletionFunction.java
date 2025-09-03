// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.test.TestAttempt;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.List;
import javax.annotation.Nullable;

/**
 * TestCompletionFunction builds all relevant test artifacts of a {@link
 * com.google.devtools.build.lib.analysis.ConfiguredTarget}. This includes test shards and repeated
 * runs.
 */
public final class TestCompletionFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    TestCompletionValue.TestCompletionKey key =
        (TestCompletionValue.TestCompletionKey) skyKey.argument();
    ConfiguredTargetKey ctKey = key.configuredTargetKey();
    TopLevelArtifactContext ctx = key.topLevelArtifactContext();
    if (env.getValue(TargetCompletionValue.key(ctKey, ctx, /* willTest= */ true)) == null) {
      return null;
    }

    ConfiguredTargetValue ctValue = (ConfiguredTargetValue) env.getValue(ctKey);
    if (ctValue == null) {
      return null;
    }

    ConfiguredTarget ct = ctValue.getConfiguredTarget();
    if (key.exclusiveTesting()) {
      // Request test execution iteratively if testing exclusively.
      for (Artifact.DerivedArtifact testArtifact : TestProvider.getTestStatusArtifacts(ct)) {
        env.getValue(testArtifact.getGeneratingActionKey());
        if (env.valuesMissing()) {
          return null;
        }
      }
    } else {
      List<SkyKey> skyKeys = Artifact.keys(TestProvider.getTestStatusArtifacts(ct));
      SkyframeLookupResult result = env.getValuesAndExceptions(skyKeys);
      if (env.valuesMissing()) {
        return null;
      }
      for (SkyKey actionKey : skyKeys) {
        try {
          if (result.getOrThrow(actionKey, ActionExecutionException.class) == null) {
            return null;
          }
        } catch (ActionExecutionException e) {
          DetailedExitCode detailedExitCode = e.getDetailedExitCode();
          if (detailedExitCode.getExitCode().equals(ExitCode.BUILD_FAILURE)
              && ctValue instanceof ActionLookupValue actionLookupValue) {
            postTestResultEventsForBuiltTestThatCouldNotBeRun(
                env, (ActionLookupData) actionKey, actionLookupValue, detailedExitCode);
          } else {
            return null;
          }
        }
      }
    }
    return TestCompletionValue.TEST_COMPLETION_MARKER;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }

  /**
   * Posts events for test actions that could not run despite the fact that the test target built
   * successfully.
   *
   * <p>When we run this SkyFunction we will have already built the test executable and its inputs,
   * but we might be unable to run the test. The currently known scenarios where this happens are:
   *
   * <ol>
   *   <li>A failure to build the exec-configured attributes providing inputs to the {@link
   *       TestRunnerAction} such as {@code $test_runtime}, {@code $test_wrapper}, {@code
   *       test_setup_script} and others.
   *   <li>The test strategy throws an {@link ExecException} prior to running the test, for example
   *       when some sort of validation fails.
   *   <li>The test action observes lost input(s) and initiates action rewinding, but the lost
   *       input(s) fail to build. Note that this implies action nondeterminism, since the lost
   *       input(s) were previously built successfully.
   * </ol>
   *
   * <p>In these scenarios, we do not get to use any {@code TestStrategy} that is responsible for
   * posting {@link TestAttempt} and {@link TestResult} events. We need to post minimal events here
   * indicating the test {@link BlazeTestStatus#FAILED_TO_BUILD FAILED_TO_BUILD}.
   */
  private static void postTestResultEventsForBuiltTestThatCouldNotBeRun(
      Environment env,
      ActionLookupData actionKey,
      ActionLookupValue actionLookupValue,
      DetailedExitCode detailedExitCode) {
    BlazeTestStatus status = BlazeTestStatus.FAILED_TO_BUILD;
    if (detailedExitCode
        .getFailureDetail()
        .getExecution()
        .getCode()
        .equals(Code.ACTION_NOT_UP_TO_DATE)) {
      status = BlazeTestStatus.NO_STATUS;
    }
    TestRunnerAction testRunnerAction =
        (TestRunnerAction) actionLookupValue.getAction(actionKey.getActionIndex());
    TestResultData testData = TestResultData.newBuilder().setStatus(status).build();
    env.getListener().post(TestAttempt.forUnstartableTestResult(testRunnerAction, testData));
    env.getListener()
        .post(
            new TestResult(
                testRunnerAction,
                testData,
                ImmutableMultimap.of(),
                /* cached= */ false,
                detailedExitCode));
  }
}
