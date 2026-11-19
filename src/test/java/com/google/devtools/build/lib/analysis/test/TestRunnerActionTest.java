// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.test;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptResult.Result.FAILED;
import static com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptResult.Result.FAILED_CAN_RETRY;
import static com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptResult.Result.PASSED;
import static com.google.devtools.build.lib.analysis.test.TestRunnerAction.computeNextRunnerAndMaxAttempts;
import static com.google.devtools.build.lib.analysis.test.TestRunnerAction.executeUnconditionally;
import static com.google.devtools.build.lib.analysis.test.TestRunnerAction.shouldAcceptCachedResult;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit.ActionCachedContext;
import com.google.devtools.build.lib.analysis.test.TestActionContext.TestRunnerSpawn;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction.TestRunnerSpawnAndMaxAttempts;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.TriState;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Optional;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestRunnerAction}. */
@RunWith(JUnit4.class)
public final class TestRunnerActionTest {

  private final TestRunnerAction testRunnerActionMock = mock(TestRunnerAction.class);

  @SuppressWarnings("InlineLambdaConstant")
  private static final Supplier<Optional<TestResultData>> EMPTY_SUPPLIER = Optional::empty;

  @SuppressWarnings("UnnecessaryLambda")
  private static final Supplier<Optional<TestResultData>> PASSED_SUPPLIER =
      () -> Optional.of(TestResultData.newBuilder().setTestPassed(true).setCachable(true).build());

  @SuppressWarnings("UnnecessaryLambda")
  private static final Supplier<Optional<TestResultData>> FAILED_SUPPLIER =
      () -> Optional.of(TestResultData.newBuilder().setTestPassed(false).setCachable(true).build());

  @SuppressWarnings("UnnecessaryLambda")
  private static final Supplier<Optional<TestResultData>> UNCACHABLE_SUPPLIER =
      () -> Optional.of(TestResultData.newBuilder().setTestPassed(true).setCachable(false).build());

  @Before
  public void configureMock() {
    when(testRunnerActionMock.getMnemonic()).thenCallRealMethod();
  }

  /** This is depended on by remote execution backend requirements. */
  @Test
  public void testMnenomic() {
    assertThat(testRunnerActionMock.getMnemonic()).isEqualTo("TestRunner");
  }

  @Test
  public void testNeedsUnconditionalExecution() {
    // --cache_test_results=0 always executes unconditionally.
    assertThat(executeUnconditionally(TriState.NO, EMPTY_SUPPLIER, false, 1)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, EMPTY_SUPPLIER, true, 1)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, EMPTY_SUPPLIER, false, 2)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, EMPTY_SUPPLIER, true, 2)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, UNCACHABLE_SUPPLIER, false, 1)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, FAILED_SUPPLIER, false, 1)).isTrue();
    assertThat(executeUnconditionally(TriState.NO, PASSED_SUPPLIER, false, 1)).isTrue();

    // --cache_test_results=1 executes tests unconditionally if one of the following is true:
    // (1) the test is marked as external
    // (2) the previous result is missing
    // (3) the previous result is uncacheable
    assertThat(executeUnconditionally(TriState.YES, EMPTY_SUPPLIER, true, 1)).isTrue(); // (1)
    assertThat(executeUnconditionally(TriState.YES, EMPTY_SUPPLIER, true, 2)).isTrue(); // (1)
    assertThat(executeUnconditionally(TriState.YES, EMPTY_SUPPLIER, false, 1)).isTrue(); // (2)
    assertThat(executeUnconditionally(TriState.YES, UNCACHABLE_SUPPLIER, false, 1)).isTrue(); // (3)
    assertThat(executeUnconditionally(TriState.YES, FAILED_SUPPLIER, false, 1)).isFalse();
    assertThat(executeUnconditionally(TriState.YES, PASSED_SUPPLIER, false, 1)).isFalse();

    // --cache_test_results=auto executes tests unconditionally if one of the following is true:
    // (1) the test is marked as external
    // (2) multiple test runs were requested
    // (3) the previous result is missing
    // (4) the previous result is uncacheable
    // (5) the previous result is a failure
    assertThat(executeUnconditionally(TriState.AUTO, EMPTY_SUPPLIER, true, 1)).isTrue(); // 1
    assertThat(executeUnconditionally(TriState.AUTO, EMPTY_SUPPLIER, false, 2)).isTrue(); // 2
    assertThat(executeUnconditionally(TriState.AUTO, EMPTY_SUPPLIER, false, 1)).isTrue(); // (3)
    assertThat(executeUnconditionally(TriState.AUTO, UNCACHABLE_SUPPLIER, false, 1))
        .isTrue(); // (4)
    assertThat(executeUnconditionally(TriState.AUTO, FAILED_SUPPLIER, false, 1)).isTrue(); // (5)
    assertThat(executeUnconditionally(TriState.AUTO, PASSED_SUPPLIER, false, 1)).isFalse();
  }

  @Test
  public void testIsObviouslyUncacheable_noAttemptToGetPreviousTestResult() {
    Supplier<Optional<TestResultData>> throwingSupplier =
        () -> {
          throw new AssertionError("Should not be called");
        };
    // --nocache_test_results
    assertThat(executeUnconditionally(TriState.NO, throwingSupplier, false, 1)).isTrue();
    // Test marked as external
    assertThat(executeUnconditionally(TriState.YES, throwingSupplier, true, 1)).isTrue();
    // --runs_per_test=2
    assertThat(executeUnconditionally(TriState.AUTO, throwingSupplier, false, 2)).isTrue();
  }

  @Test
  public void testShouldAcceptCachedResult() {
    // --cache_test_results=0 always rejects a cached result.
    assertThat(shouldAcceptCachedResult(TriState.NO, false, 1)).isFalse();
    assertThat(shouldAcceptCachedResult(TriState.NO, true, 1)).isFalse();
    assertThat(shouldAcceptCachedResult(TriState.NO, false, 2)).isFalse();
    assertThat(shouldAcceptCachedResult(TriState.NO, true, 2)).isFalse();

    // --cache_test_results=1 rejects a cached result only for external tests.
    assertThat(shouldAcceptCachedResult(TriState.YES, true, 1)).isFalse();
    assertThat(shouldAcceptCachedResult(TriState.YES, true, 2)).isFalse();
    assertThat(shouldAcceptCachedResult(TriState.YES, false, 1)).isTrue();
    assertThat(shouldAcceptCachedResult(TriState.YES, false, 2)).isTrue();

    // --cache_test_results=auto rejects a cached result if one of the following is true:
    // (1) the test is marked as external
    // (2) multiple test runs were requested
    assertThat(shouldAcceptCachedResult(TriState.AUTO, true, 1)).isFalse(); // (1)
    assertThat(shouldAcceptCachedResult(TriState.AUTO, false, 2)).isFalse(); // (2)
    assertThat(shouldAcceptCachedResult(TriState.AUTO, true, 2)).isFalse(); // (1)+(2)
    assertThat(shouldAcceptCachedResult(TriState.AUTO, false, 1)).isTrue();
  }

  @Test
  public void actionCacheHit_noEventOnReadFailure_newCachedTestResultData() throws IOException {
    when(testRunnerActionMock.readCacheStatus()).thenReturn(TestResultData.getDefaultInstance());
    String ioExceptionMsg = "IO exception message.";
    when(testRunnerActionMock.getErrorMessageOnNewCachedTestResultError(ioExceptionMsg))
        .thenReturn(ioExceptionMsg);
    doCallRealMethod().when(testRunnerActionMock).actionCacheHit(any());
    doCallRealMethod().when(testRunnerActionMock).maybeReadCacheStatus();

    ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
    doAnswer(
            invocation -> {
              throw new AssertionError("Unexpected event posted: " + invocation.getArgument(0));
            })
        .when(eventHandler)
        .post(any());

    TestActionContext testActionContext = mock(TestActionContext.class);
    when(testActionContext.newCachedTestResult(any(), any(), any(), any()))
        .thenThrow(new IOException(ioExceptionMsg));

    ActionCachedContext cachedContext = mock(ActionCachedContext.class);
    when(cachedContext.getEventHandler()).thenReturn(eventHandler);
    when(cachedContext.getContext(TestActionContext.class)).thenReturn(testActionContext);

    assertThat(testRunnerActionMock.actionCacheHit(cachedContext)).isFalse();
  }

  @Test
  public void actionCacheHit_noEventOnReadFailureIoException_cachedTestResultData()
      throws IOException {
    ActionCachedContext cachedContext = mock(ActionCachedContext.class);
    doCallRealMethod().when(testRunnerActionMock).maybeReadCacheStatus();
    when(testRunnerActionMock.readCacheStatus()).thenThrow(new IOException());
    when(testRunnerActionMock.getErrorMessageOnCachedTestResultError()).thenReturn("error msg");

    ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
    when(cachedContext.getEventHandler()).thenReturn(eventHandler);
    doAnswer(
            invocation -> {
              throw new AssertionError("Unexpected event posted: " + invocation.getArgument(0));
            })
        .when(eventHandler)
        .post(any());

    doCallRealMethod().when(testRunnerActionMock).actionCacheHit(any());
    assertThat(testRunnerActionMock.actionCacheHit(cachedContext)).isFalse();
  }

  @Test
  public void actionCacheHit_noEventOnReadFailureFileNotFoundException_cachedTestResultData()
      throws IOException {
    ActionCachedContext cachedContext = mock(ActionCachedContext.class);
    doCallRealMethod().when(testRunnerActionMock).maybeReadCacheStatus();
    when(testRunnerActionMock.readCacheStatus()).thenThrow(new FileNotFoundException());
    when(testRunnerActionMock.getErrorMessageOnCachedTestResultError()).thenReturn("error msg");

    ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
    when(cachedContext.getEventHandler()).thenReturn(eventHandler);
    doAnswer(
            invocation -> {
              throw new AssertionError("Unexpected event posted: " + invocation.getArgument(0));
            })
        .when(eventHandler)
        .post(any());
    doCallRealMethod().when(testRunnerActionMock).actionCacheHit(any());

    assertThat(testRunnerActionMock.actionCacheHit(cachedContext)).isFalse();
    assertThat(testRunnerActionMock.actionCacheHit(cachedContext)).isFalse();
    verify(testRunnerActionMock, times(1)).readCacheStatus();
  }

  @Test
  public void oneDiskAccess_multipleCallsToMaybeReadCachedStatus() throws Exception {
    doCallRealMethod().when(testRunnerActionMock).maybeReadCacheStatus();
    when(testRunnerActionMock.readCacheStatus()).thenReturn(TestResultData.getDefaultInstance());
    Optional<TestResultData> testResultData = Optional.of(TestResultData.getDefaultInstance());
    assertThat(testRunnerActionMock.maybeReadCacheStatus()).isEqualTo(testResultData);
    assertThat(testRunnerActionMock.maybeReadCacheStatus()).isEqualTo(testResultData);
    assertThat(testRunnerActionMock.maybeReadCacheStatus()).isEqualTo(testResultData);

    verify(testRunnerActionMock, times(1)).readCacheStatus();
  }

  @Test
  public void runAttemptContinuation_defaultRunner() throws Exception {
    TestRunnerSpawn testRunner = mock(TestRunnerSpawn.class);
    when(testRunner.getFlakyRetryRunner(any())).thenCallRealMethod();
    when(testRunner.getFallbackRunner()).thenCallRealMethod();

    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 1, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(testRunner, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 2, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(testRunner, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 3, 3, ImmutableList.of()))
        .isNull();

    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 1, 3, ImmutableList.of()))
        .isNull();
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 2, 3, ImmutableList.of()))
        .isNull();
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 3, 3, ImmutableList.of()))
        .isNull();

    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 1, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 2, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 3, 3, ImmutableList.of()));
  }

  @Test
  public void runAttemptContinuation_flakyRetryRunner() throws Exception {
    TestRunnerSpawn testRunner = mock(TestRunnerSpawn.class);
    TestRunnerSpawn flakyRetry = mock(TestRunnerSpawn.class);
    when(testRunner.getFlakyRetryRunner(any())).thenReturn(flakyRetry);
    when(testRunner.getFallbackRunner()).thenCallRealMethod();

    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 1, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(flakyRetry, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 2, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(flakyRetry, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 3, 3, ImmutableList.of()))
        .isNull();

    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 1, 3, ImmutableList.of()))
        .isNull();
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 2, 3, ImmutableList.of()))
        .isNull();
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 3, 3, ImmutableList.of()))
        .isNull();

    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 1, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 2, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 3, 3, ImmutableList.of()));
  }

  @Test
  public void runAttemptContinuation_fallbackRunner() throws Exception {
    TestRunnerSpawn testRunner = mock(TestRunnerSpawn.class);
    TestRunnerSpawn fallbackRunner = mock(TestRunnerSpawn.class);
    when(testRunner.getFlakyRetryRunner(any())).thenCallRealMethod();
    when(testRunner.getFallbackRunner()).thenReturn(fallbackRunner);

    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 1, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(testRunner, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 2, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(testRunner, 3));
    assertThat(
            computeNextRunnerAndMaxAttempts(FAILED_CAN_RETRY, testRunner, 3, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(fallbackRunner, 6));

    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 1, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(fallbackRunner, 4));
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 2, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(fallbackRunner, 5));
    assertThat(computeNextRunnerAndMaxAttempts(FAILED, testRunner, 3, 3, ImmutableList.of()))
        .isEqualTo(TestRunnerSpawnAndMaxAttempts.create(fallbackRunner, 6));

    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 1, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 2, 3, ImmutableList.of()));
    assertThrows(
        IllegalStateException.class,
        () -> computeNextRunnerAndMaxAttempts(PASSED, testRunner, 3, 3, ImmutableList.of()));
  }
}
