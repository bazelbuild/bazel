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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
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
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestResultAggregator}. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public final class TestResultAggregatorTest {

  private final TestParams mockParams = mock(TestParams.class);
  private TestResultAggregator underTest;

  @Before
  public void createAggregator() {
    when(mockParams.runsDetectsFlakes()).thenReturn(false);
    when(mockParams.getTimeout()).thenReturn(TestTimeout.LONG);
    when(mockParams.getTestStatusArtifacts()).thenReturn(ImmutableList.of());

    ConfiguredTarget mockTarget = mock(ConfiguredTarget.class);
    when(mockTarget.getProvider(TestProvider.class)).thenReturn(new TestProvider(mockParams));

    underTest =
        new TestResultAggregator(
            mockTarget,
            /*configuration=*/ null,
            new AggregationPolicy(
                new EventBus(),
                /*testCheckUpToDate=*/ false,
                /*testVerboseTimeoutWarnings=*/ false),
            /*skippedThisTest=*/ false);
  }

  @Test
  public void incrementalAnalyze_nonCachedResult_setsActionRanTrue() {
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setRemotelyCached(false).build(),
            /*cached=*/ false,
            /*systemFailure=*/ null));
    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isTrue();
  }

  @Test
  public void incrementalAnalyze_locallyCachedTest_setsActionRanFalse() {
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setRemotelyCached(false).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));
    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isFalse();
  }

  @Test
  public void incrementalAnalyze_remotelyCachedTest_setsActionRanFalse() {
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setRemotelyCached(true).build(),
            /*cached=*/ false,
            /*systemFailure=*/ null));
    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isFalse();
  }

  @Test
  public void incrementalAnalyze_newCachedResult_keepsActionRanTrueWhenAlreadyTrue() {
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setRemotelyCached(false).build(),
            /*cached=*/ false,
            /*systemFailure=*/ null));
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setRemotelyCached(true).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));
    assertThat(underTest.aggregateAndReportSummary(false).actionRan()).isTrue();
  }

  @Test
  public void timingAggregation() {
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setStartTimeMillisEpoch(7).setRunDurationMillis(10).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setStartTimeMillisEpoch(12).setRunDurationMillis(1).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));

    TestSummary summary = underTest.aggregateAndReportSummary(false);
    assertThat(summary.getFirstStartTimeMillis()).isEqualTo(7);
    assertThat(summary.getLastStopTimeMillis()).isEqualTo(17);
    assertThat(summary.getTotalRunDurationMillis()).isEqualTo(11);
  }

  @Test
  public void cancelConcurrentTests_cancellationAfterPassIgnored() {
    when(mockParams.runsDetectsFlakes()).thenReturn(true);
    when(mockParams.getRuns()).thenReturn(2);

    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));
    underTest.incrementalAnalyze(
        new TestResult(
            mock(TestRunnerAction.class),
            TestResultData.newBuilder().setStatus(BlazeTestStatus.INCOMPLETE).build(),
            /*cached=*/ true,
            /*systemFailure=*/ null));

    assertThat(underTest.aggregateAndReportSummary(false).getStatus())
        .isEqualTo(BlazeTestStatus.PASSED);
  }
}
