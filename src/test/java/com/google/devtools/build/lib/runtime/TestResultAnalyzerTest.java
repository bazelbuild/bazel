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
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class TestResultAnalyzerTest {

  private TestResultAnalyzer underTest;
  
  @Before
  public final void createMocks() throws Exception  {
    OptionsParser testSpecificOptions =
        OptionsParser.builder()
            .optionsClasses(TestSummaryOptions.class, ExecutionOptions.class)
            .build();
    EventBus mockBus = mock(EventBus.class);
    underTest = new TestResultAnalyzer(
        testSpecificOptions.getOptions(TestSummaryOptions.class),
        testSpecificOptions.getOptions(ExecutionOptions.class),
        mockBus);
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanTrueWhenThereAreNonCachedResults() {
    TestSummary.Builder summaryBuilder = makeTestSummaryBuilder();
    assertThat(summaryBuilder.peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(false).build();
    TestResult result = new TestResult(
        mock(TestRunnerAction.class),
        testResultData,
        /*cached=*/false);

    TestSummary.Builder newSummaryBuilder = underTest.incrementalAnalyze(summaryBuilder, result);
    assertThat(newSummaryBuilder.peek().actionRan()).isTrue();
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanFalseForLocallyCachedTests() {
    TestSummary.Builder summaryBuilder = makeTestSummaryBuilder();
    assertThat(summaryBuilder.peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(false).build();
    TestResult result = new TestResult(
        mock(TestRunnerAction.class),
        testResultData,
        /*cached=*/true);
    
    TestSummary.Builder newSummaryBuilder = underTest.incrementalAnalyze(summaryBuilder, result);
    assertThat(newSummaryBuilder.peek().actionRan()).isFalse();
  }

  @Test
  public void testIncrementalAnalyzeSetsActionRanFalseForRemotelyCachedTests() {
    TestSummary.Builder summaryBuilder = makeTestSummaryBuilder();
    assertThat(summaryBuilder.peek().actionRan()).isFalse();

    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(true).build();
    TestResult result = new TestResult(
        mock(TestRunnerAction.class),
        testResultData,
        /*cached=*/false);

    TestSummary.Builder newSummaryBuilder = underTest.incrementalAnalyze(summaryBuilder, result);
    assertThat(newSummaryBuilder.peek().actionRan()).isFalse();
  }

  @Test
  public void testIncrementalAnalyzeKeepsActionRanTrueWhenAlreadyTrueAndNewCachedResults() {
    TestSummary.Builder summaryBuilder = makeTestSummaryBuilder().setActionRan(true);
    
    TestResultData testResultData = TestResultData.newBuilder().setRemotelyCached(true).build();
    TestResult result = new TestResult(
        mock(TestRunnerAction.class),
        testResultData,
        /*cached=*/true);

    TestSummary.Builder newSummaryBuilder = underTest.incrementalAnalyze(summaryBuilder, result);
    assertThat(newSummaryBuilder.peek().actionRan()).isTrue();
  }

  private TestSummary.Builder makeTestSummaryBuilder() {
    // a lot of mocks to mock out fetching the TestTimeout configuration needed by
    //  {@link TestResultAnalyzer#shouldEmitTestSizeWarningInSummary(...)
    TestParams mockParams = mock(TestParams.class);
    when(mockParams.getTimeout()).thenReturn(TestTimeout.LONG);
    TestProvider testProvider = new TestProvider(mockParams, ImmutableList.<String>of());
    
    ConfiguredTarget mockTarget = mock(ConfiguredTarget.class);
    when(mockTarget.getProvider(TestProvider.class)).thenReturn(testProvider);
    
    return TestSummary.newBuilder()
        .setStatus(BlazeTestStatus.PASSED)
        .setTarget(mockTarget);
    
  }
  
}
