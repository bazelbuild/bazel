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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A context for the execution of test actions ({@link TestRunnerAction}).
 */
public interface TestActionContext extends ActionContext {

  /**
   * A group of attempts for a single test shard, ran either sequentially or in parallel.
   *
   * <p>When one attempt succeeds, threads running the other attempts get an {@link
   * InterruptedException} and {@link #cancelled()} will in the future return true. When a thread
   * joins an attempt group that is already cancelled, {@link InterruptedException} will be thrown
   * on the call to {@link #register()}.
   */
  interface AttemptGroup {

    /**
     * Registers a thread to the attempt group.
     *
     * <p>If the attempt group is already cancelled, throw {@link InterruptedException}.
     */
    void register() throws InterruptedException;

    /** Unregisters a thread from the attempt group. */
    void unregister();

    /** Signal that the attempt run by this thread has succeeded and cancel all the others. */
    void cancelOthers();

    /** Whether the attempt group has been cancelled. */
    boolean cancelled();

    /** A dummy attempt group used when no flaky test attempt cancellation is done. */
    AttemptGroup NOOP =
        new AttemptGroup() {
          @Override
          public void register() {}

          @Override
          public void unregister() {}

          @Override
          public void cancelOthers() {}

          @Override
          public boolean cancelled() {
            return false;
          }
        };
  }

  TestRunnerSpawn createTestRunnerSpawn(
      TestRunnerAction testRunnerAction, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

  /** Returns whether test_keep_going is enabled. */
  boolean isTestKeepGoing();

  /**
   * Returns {@code true} to indicate that exclusive tests should be treated as regular parallel
   * tests.
   *
   * <p>Returning {@code true} may make sense for certain forced remote test execution strategies
   * where running tests in sequence would be wasteful.
   */
  default boolean forceExclusiveTestsInParallel() {
    return false;
  }

  /**
   * Returns {@code true} to indicate that "exclusive-if-local" tests should be treated as regular
   * parallel tests.
   *
   * <p>Returning {@code true} may make sense for certain remote test execution strategies where
   * running tests in sequence would be wasteful.
   */
  default boolean forceExclusiveIfLocalTestsInParallel() {
    return false;
  }

  /** Creates a cached test result. */
  TestResult newCachedTestResult(Path execRoot, TestRunnerAction action, TestResultData cached)
      throws IOException;

  /** Returns the attempt group associaed with the given shard. */
  AttemptGroup getAttemptGroup(ActionOwner owner, int shardNum);

  /** An individual test attempt result. */
  interface TestAttemptResult {
    /** Test attempt result classification, splitting failures into permanent vs retriable. */
    enum Result {
      /** Test attempt successful. */
      PASSED,
      /** Test failed, potentially due to test flakiness, can be retried. */
      FAILED_CAN_RETRY,
      /** Permanent failure. */
      FAILED;

      boolean canRetry() {
        return this == FAILED_CAN_RETRY;
      }
    }

    /** Returns the overall test result. */
    Result result();

    /** Returns a list of spawn results for this test attempt. */
    ImmutableList<SpawnResult> spawnResults();

    /**
     * Returns a description of the system failure associated with the primary spawn result, if any.
     */
    @Nullable
    default DetailedExitCode primarySystemFailure() {
      if (spawnResults().isEmpty()) {
        return null;
      }
      SpawnResult primarySpawnResult = spawnResults().get(0);
      if (primarySpawnResult.status() == Status.SUCCESS) {
        return null;
      }
      if (primarySpawnResult.status().isConsideredUserError()) {
        return null;
      }
      return DetailedExitCode.of(primarySpawnResult.failureDetail());
    }
  }

  /**
   * An object representing a failed non-final attempt. This is only used for tests that are run
   * multiple times. At this time, Bazel retries tests until the first passed attempt, or until the
   * number of retries is exhausted - whichever comes first. This interface represents the result
   * from a previous attempt, but never the final attempt, even if unsuccessful.
   */
  interface FailedAttemptResult {}

  /** A delegate to run a test. This may include running multiple spawns, renaming outputs, etc. */
  interface TestRunnerSpawn {
    ActionExecutionContext getActionExecutionContext();

    /** Run the test attempt. Blocks until the attempt is complete. */
    TestAttemptResult execute() throws InterruptedException, IOException, ExecException;

    /**
     * After the first attempt has run, this method is called to determine the maximum number of
     * attempts for this test.
     */
    int getMaxAttempts(TestAttemptResult firstTestAttemptResult);

    /** Rename the output files if the test attempt failed, and post the test attempt result. */
    FailedAttemptResult finalizeFailedTestAttempt(TestAttemptResult testAttemptResult, int attempt)
        throws IOException, ExecException, InterruptedException;

    /** Post the final test result based on the last attempt and the list of failed attempts. */
    void finalizeTest(
        TestAttemptResult lastTestAttemptResult, List<FailedAttemptResult> failedAttempts)
        throws IOException, ExecException, InterruptedException;

    /** Post the final test result based on the last attempt and the list of failed attempts. */
    void finalizeCancelledTest(List<FailedAttemptResult> failedAttempts)
        throws IOException, ExecException, InterruptedException;

    /**
     * Return a {@link TestRunnerSpawn} object if test fallback is enabled, or {@code null}
     * otherwise. Test fallback is a feature to allow a test to run with one strategy until the max
     * attempts are exhausted and then run with another strategy for another set of attempts. This
     * is rarely used, and should ideally be removed.
     */
    @Nullable
    default TestRunnerSpawn getFallbackRunner() throws ExecException, InterruptedException {
      return null;
    }

    /**
     * Return a {@link TestRunnerSpawn} object that is used on flaky retries. Flaky retry runner
     * allows a test to run with a different strategy on flaky retries (for example, enabling test
     * fail-fast mode to save up resources).
     */
    default TestRunnerSpawn getFlakyRetryRunner() throws ExecException, InterruptedException {
      return this;
    }
  }
}
