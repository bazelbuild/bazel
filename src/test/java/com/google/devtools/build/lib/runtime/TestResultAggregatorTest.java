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
import com.google.devtools.build.lib.actions.Artifact;
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
public class TestResultAggregatorTest {
  private TestParams mockParams;
  private ConfiguredTarget configuredTarget;
  private TestResultAggregator underTest;

  @Before
  public final void createMocks() throws Exception {
    mockParams = mock(TestParams.class);
    when(mockParams.runsDetectsFlakes()).thenReturn(false);
    when(mockParams.getTimeout()).thenReturn(TestTimeout.LONG);
    when(mockParams.getTestStatusArtifacts())
        .thenReturn(ImmutableList.<Artifact.DerivedArtifact>of());
    TestProvider testProvider = new TestProvider(mockParams, ImmutableList.<String>of());

    ConfiguredTarget mockTarget = mock(ConfiguredTarget.class);
    when(mockTarget.getProvider(TestProvider.class)).thenReturn(testProvider);
    this.configuredTarget = mockTarget;

    underTest =
        new TestResultAggregator(
            configuredTarget,
            null,
            new AggregationPolicy(
                new EventBus(),
                /*testCheckUpToDate=*/ false,
                /*testVerboseTimeoutWarnings=*/ false));
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanTrueWhenThereAreNonCachedResults() {
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(false).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ false));
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isTrue();
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanFalseForLocallyCachedTests() {
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(false).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isFalse();
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanFalseForRemotelyCachedTests() {
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(true).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ false));
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isFalse();
  }

  @Test
  public void testIncrementalAnalyzeKeepsActionRanTrueWhenAlreadyTrueAndNewCachedResults() {
    underTest.getCurrentSummaryForTesting().setActionRan(true);

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(true).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));
    assertThat(underTest.getCurrentSummaryForTesting().peek().actionRan()).isTrue();
  }

  @Test
  public void testTimingAggregation() {
    underTest.getCurrentSummaryForTesting().setActionRan(true);

    TestResultData testResultData =
        TestResultData.newBuilder().setStartTimeMillisEpoch(7).setRunDurationMillis(10).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));

    testResultData =
        TestResultData.newBuilder().setStartTimeMillisEpoch(12).setRunDurationMillis(1).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));
    TestSummary summary = underTest.getCurrentSummaryForTesting().build();

    assertThat(summary.actionRan()).isTrue();
    assertThat(summary.getFirstStartTimeMillis()).isEqualTo(7);
    assertThat(summary.getLastStopTimeMillis()).isEqualTo(17);
    assertThat(summary.getTotalRunDurationMillis()).isEqualTo(11);
  }

  @Test
  public void testCancelledTest() {
    when(mockParams.runsDetectsFlakes()).thenReturn(true);
    when(mockParams.getRuns()).thenReturn(2);
    underTest.getCurrentSummaryForTesting().setActionRan(true);

    TestResultData testResultData =
        TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));

    testResultData = TestResultData.newBuilder().setStatus(BlazeTestStatus.INCOMPLETE).build();
    underTest.incrementalAnalyze(
        new TestResult(mock(TestRunnerAction.class), testResultData, /*cached=*/ true));
    TestSummary summary = underTest.getCurrentSummaryForTesting().build();

    assertThat(summary.getStatus()).isEqualTo(BlazeTestStatus.PASSED);
  }
}
