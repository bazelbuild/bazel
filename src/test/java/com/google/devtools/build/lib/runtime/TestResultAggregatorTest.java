// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider.TestParams;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.runtime.TestResultAggregator.AggregationPolicy;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.util.stream.Stream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestResultAggregator}. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public final class TestResultAggregatorTest {

  private final TestParams mockParams = mock(TestParams.class);

  @Before
  public void configureMockParams() {
    when(mockParams.runsDetectsFlakes()).thenReturn(false);
    when(mockParams.getTimeout()).thenReturn(TestTimeout.LONG);
  }

  @Test
  public void nonCachedResult_setsActionRanTrue() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(1);

    underTest.testEvent(
        testResult(TestResultData.newBuilder().setRemotelyCached(false), /*locallyCached=*/ false));

    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isTrue();
  }

  @Test
  public void locallyCachedTest_setsActionRanFalse() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(1);

    underTest.testEvent(
        testResult(TestResultData.newBuilder().setRemotelyCached(false), /*locallyCached=*/ true));

    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isFalse();
  }

  @Test
  public void remotelyCachedTest_setsActionRanFalse() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(1);

    underTest.testEvent(
        testResult(TestResultData.newBuilder().setRemotelyCached(true), /*locallyCached=*/ false));

    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isFalse();
  }

  @Test
  public void newCachedResult_keepsActionRanTrueWhenAlreadyTrue() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(2);

    underTest.testEvent(
        testResult(TestResultData.newBuilder().setRemotelyCached(false), /*locallyCached=*/ false));
    underTest.testEvent(
        testResult(TestResultData.newBuilder().setRemotelyCached(true), /*locallyCached=*/ true));

    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isTrue();
  }

  @Test
  public void timingAggregation() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(2);

    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStartTimeMillisEpoch(7).setRunDurationMillis(10),
            /*locallyCached=*/ true));
    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStartTimeMillisEpoch(12).setRunDurationMillis(1),
            /*locallyCached=*/ true));

    TestSummary summary = underTest.aggregateAndReportSummary(false);
    assertThat(summary.getFirstStartTimeMillis()).isEqualTo(7);
    assertThat(summary.getLastStopTimeMillis()).isEqualTo(17);
    assertThat(summary.getTotalRunDurationMillis()).isEqualTo(11);
  }

  @Test
  public void cancelConcurrentTests_cancellationAfterPassIgnored() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(2);
    when(mockParams.runsDetectsFlakes()).thenReturn(true);

    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED),
            /*locallyCached=*/ true));
    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStatus(BlazeTestStatus.INCOMPLETE),
            /*locallyCached=*/ true));

    assertThat(underTest.aggregateAndReportSummary(false).getStatus())
        .isEqualTo(BlazeTestStatus.PASSED);
  }

  @Test
  public void notAllTestRunsReported_skipTargetsOnFailure_noStatus() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(2);

    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED),
            /*locallyCached=*/ false));

    assertThat(underTest.aggregateAndReportSummary(/*skipTargetsOnFailure=*/ true).getStatus())
        .isEqualTo(BlazeTestStatus.NO_STATUS);
  }

  @Test
  public void notAllTestRunsReported_noSkipTargetsOnFailure_incomplete() {
    TestResultAggregator underTest = createAggregatorWithTestRuns(2);

    underTest.testEvent(
        testResult(
            TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED),
            /*locallyCached=*/ false));

    assertThat(underTest.aggregateAndReportSummary(/*skipTargetsOnFailure=*/ false).getStatus())
        .isEqualTo(BlazeTestStatus.INCOMPLETE);
  }

  private TestResultAggregator createAggregatorWithTestRuns(int testRuns) {
    when(mockParams.getTestStatusArtifacts())
        .thenReturn(
            Stream.generate(() -> mock(DerivedArtifact.class))
                .limit(testRuns)
                .collect(toImmutableList()));
    when(mockParams.getRuns()).thenReturn(testRuns);

    ConfiguredTarget mockTarget = mock(ConfiguredTarget.class);
    when(mockTarget.getProvider(TestProvider.class)).thenReturn(new TestProvider(mockParams));

    return new TestResultAggregator(
        mockTarget,
        mock(BuildConfiguration.class),
        new AggregationPolicy(
            new EventBus(), /*testCheckUpToDate=*/ false, /*testVerboseTimeoutWarnings=*/ false),
        /*skippedThisTest=*/ false);
  }

  private static TestResult testResult(TestResultData.Builder data, boolean locallyCached) {
    TestRunnerAction mockTestAction = mock(TestRunnerAction.class);
    when(mockTestAction.getTestOutputsMapping(any(), any())).thenReturn(ImmutableList.of());
    return new TestResult(mockTestAction, data.build(), locallyCached, /*systemFailure=*/ null);
  }
}
