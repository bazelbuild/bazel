// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner;

import org.junit.internal.AssumptionViolatedException;
import org.junit.runner.Description;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunListener;

/**
 * A straightforward listener that prints to stdout/stderr whenever a test changes its state (e.g.
 * started, finished, failed).
 */
class TestListener extends RunListener {
    /**
     * Called before any tests have been run. Prints to stdout the number of test cases found.
     *
     * @param description describes the tests to be run
     */
    @Override
    public void testRunStarted(Description description) throws Exception {
      System.out.println("Found " + formatTestCaseCount(description.testCount()) + ".");
    }

    /**
     * Called when all tests have finished. Prints to stdout if the tests were successful or not. If
     * not, it also prints the number of failed test cases. Finally, it prints the number of
     * ignored test cases.
     *
     * @param result the summary of the test run, including all the tests that failed
     */
    @Override
    public void testRunFinished(Result result) throws Exception {
      if (result.wasSuccessful()) {
        System.out.println("Successfully finished running "
            + formatTestCaseCount(result.getRunCount()) + " in " + result.getRunTime() + " ms.");
      } else {
        System.out.println("Finished running " + formatTestCaseCount(result.getRunCount())
            + " in " + result.getRunTime() + " ms.");
        int failureCount = result.getFailureCount();
        if (failureCount == 1) {
          System.out.println("There was 1 failed test.");
        } else {
          System.out.println("There were " + failureCount + " failed tests.");
        }
      }
      int ignoredCount = result.getIgnoreCount();
      if (ignoredCount == 1) {
        System.out.println(result.getIgnoreCount() + " test case was ignored.");
      } else if (ignoredCount > 1) {
        System.out.println(result.getIgnoreCount() + " test cases were ignored.");
      }
    }

    /**
     * Called when an atomic test is about to be started. Prints to stdout the name of the test that
     * started with the corresponding information.
     *
     * @param description the description of the test that is about to be run
     * (generally a class and method name)
     */
    @Override
    public void testStarted(Description description) throws Exception {
      System.out.println("Test case started: " + description.getDisplayName());
    }

    /**
     * Called when an atomic test fails. Prints to stderr the name of the test that failed
     * (including its class) and the reason why, including the stack trace.
     *
     * @param failure describes the test that failed and the exception that was thrown
     */
    @Override
    public void testFailure(Failure failure) throws Exception {
      System.err.println("Failure in " + failure.getTestHeader() + ": " + failure.getMessage()
          + "\n" + failure.getTrace());
    }

    /**
     * Called when an atomic test flags that it assumes a condition that is false. Prints to stderr
     * that a test case assumed false condition, including the corresponding message containing
     * the context.
     *
     * @param failure describes the test that failed and the
     * {@link AssumptionViolatedException} that was thrown
     */
    @Override
    public void testAssumptionFailure(Failure failure) {
      System.err.println("Test case assumed false condition: " + failure.getMessage());
    }

    /**
     *  Called when a test will not be run, generally because a test method is annotated with
     *  Ignore. Prints to stderr that a test case was ignored, alongside with the test name.
     **/
    @Override
    public void testIgnored(Description description) throws java.lang.Exception {
      System.err.println("Test case " + description.getMethodName() + " ignored.");
    }

    private static String formatTestCaseCount(int count) {
      if (count == 1) {
        return "1 test case";
      }
      return count + " test cases";
    }
  }
