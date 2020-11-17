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
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A context for the execution of test actions ({@link TestRunnerAction}).
 */
public interface TestActionContext extends ActionContext {
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
  default boolean forceParallelTestExecution() {
    return false;
  }

  /** Creates a cached test result. */
  TestResult newCachedTestResult(Path execRoot, TestRunnerAction action, TestResultData cached)
      throws IOException;

  /**
   * Returns a listenable future that is unique for any given combination of owner and shard number,
   * i.e., that is cached across different runs within the same shard of the same target. This is to
   * facilitate cross-action cancellation - if {@code runs_per_test} and {@code
   * runs_per_test_detects_flake} are both set, then it is sufficient to have a single passing
   * result per shard, and any concurrent actions can be cancelled.
   *
   * <p>Note that the output files of a test are named after the owner, which guarantees that there
   * are no two tests with the same owner.
   */
  @Nullable
  ListenableFuture<Void> getTestCancelFuture(ActionOwner owner, int shardNum);

  /**
   * An object representing an individual test attempt result. Note that {@link TestRunnerSpawn} is
   * generic in a subtype of this type; this interface only provide a tiny amount of generic
   * top-level functionality necessary to share code between the different {@link TestActionContext}
   * implementations.
   */
  interface TestAttemptResult {
    /** Test attempt result, splitting failures into permanent vs retriable. */
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
  }

  /**
   * An object representing a failed non-final attempt. This is only used for tests that are run
   * multiple times. At this time, Bazel retries tests until the first passed attempt, or until the
   * number of retries is exhausted - whichever comes first. This interface represents the result
   * from a previous attempt, but never the final attempt, even if unsuccessful.
   */
  interface FailedAttemptResult {}

  /** A continuation that represents a single test attempt. */
  abstract class TestAttemptContinuation {
    public static TestAttemptContinuation of(TestAttemptResult testAttemptResult) {
      return new Finished(testAttemptResult);
    }

    /** Returns true if this is a final result. */
    public boolean isDone() {
      return false;
    }

    /** Returns a future representing any ongoing concurrent work, or {@code null} otherwise. */
    @Nullable
    public abstract ListenableFuture<?> getFuture();

    /** Performs the next step in the process of executing the test attempt. */
    public abstract TestAttemptContinuation execute()
        throws InterruptedException, IOException, ExecException;

    /** Returns the final result. */
    public TestAttemptResult get() {
      throw new IllegalStateException();
    }

    /** Represents a finished test attempt. */
    private static final class Finished extends TestAttemptContinuation {
      private final TestAttemptResult testAttemptResult;

      private Finished(TestAttemptResult testAttemptResult) {
        this.testAttemptResult = testAttemptResult;
      }

      @Override
      public boolean isDone() {
        return true;
      }

      @Override
      public ListenableFuture<?> getFuture() {
        return Futures.immediateFuture(null);
      }

      @Override
      public TestAttemptContinuation execute() {
        return this;
      }

      @Override
      public TestAttemptResult get() {
        return testAttemptResult;
      }
    }
  }

  /** A delegate to run a test. This may include running multiple spawns, renaming outputs, etc. */
  interface TestRunnerSpawn {
    ActionExecutionContext getActionExecutionContext();

    /**
     * Begin the test attempt execution. This may block until the test attempt is complete and
     * return a completed result, or it may return a continuation with a non-null future
     * representing asynchronous execution.
     */
    TestAttemptContinuation beginExecution() throws InterruptedException, IOException;

    /**
     * After the first attempt has run, this method is called to determine the maximum number of
     * attempts for this test.
     */
    int getMaxAttempts(TestAttemptResult firstTestAttemptResult);

    /** Rename the output files if the test attempt failed, and post the test attempt result. */
    FailedAttemptResult finalizeFailedTestAttempt(TestAttemptResult testAttemptResult, int attempt)
        throws IOException;

    /** Post the final test result based on the last attempt and the list of failed attempts. */
    void finalizeTest(
        TestAttemptResult lastTestAttemptResult, List<FailedAttemptResult> failedAttempts)
        throws IOException;

    /** Post the final test result based on the last attempt and the list of failed attempts. */
    void finalizeCancelledTest(List<FailedAttemptResult> failedAttempts) throws IOException;

    /**
     * Return a {@link TestRunnerSpawn} object if test fallback is enabled, or {@code null}
     * otherwise. Test fallback is a feature to allow a test to run with one strategy until the max
     * attempts are exhausted and then run with another strategy for another set of attempts. This
     * is rarely used, and should ideally be removed.
     */
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
