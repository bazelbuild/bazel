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

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.IOException;
import java.util.List;

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
   * Creates a cached test result.
   */
  TestResult newCachedTestResult(Path execRoot, TestRunnerAction action, TestResultData cached)
      throws IOException;

  /**
   * An object representing an individual test attempt result. Note that {@link TestRunnerSpawn} is
   * generic in a subtype of this type; this interface only provide a tiny amount of generic
   * top-level functionality necessary to share code between the different {@link TestActionContext}
   * implementations.
   */
  interface TestAttemptResult {
    /** Returns {@code true} if the test attempt passed successfully. */
    boolean hasPassed();

    /** Returns a list of spawn results for this test attempt. */
    List<SpawnResult> spawnResults();
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
    /** Execute the test, and handle the test runner protocol. */
    TestAttemptResult execute() throws InterruptedException, IOException, ExecException;

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
        throws IOException, ExecException;

    /**
     * Return a {@link TestRunnerSpawn} object if test fallback is enabled, or {@code null}
     * otherwise. Test fallback is a feature to allow a test to run with one strategy until the max
     * attempts are exhausted and then run with another strategy for another set of attempts. This
     * is rarely used, and should ideally be removed.
     */
    default TestRunnerSpawn getFallbackRunner() throws ExecException, InterruptedException {
      return null;
    }
  }
}
