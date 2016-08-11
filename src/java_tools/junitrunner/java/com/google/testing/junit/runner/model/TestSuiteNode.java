// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.model;

import com.google.testing.junit.runner.model.TestResult.Status;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import org.joda.time.Interval;
import org.junit.runner.Description;

/**
 * A parent node in the test suite model.
 */
class TestSuiteNode extends TestNode {
  private final List<TestNode> children = new ArrayList<>();

  TestSuiteNode(Description description) {
    super(description);
  }

  // VisibleForTesting
  @Override
  public List<TestNode> getChildren() {
    return Collections.unmodifiableList(children);
  }

  @Override
  public boolean isTestCase() {
    return false;
  }

  @Override
  public void testFailure(Throwable throwable, long now) {
    for (TestNode child : getChildren()) {
      child.testFailure(throwable, now);
    }
  }

  @Override
  public void dynamicTestFailure(Description test, Throwable throwable, long now) {
    for (TestNode child : getChildren()) {
      child.dynamicTestFailure(test, throwable, now);
    }
  }

  @Override
  public void testInterrupted(long now) {
    for (TestNode child : getChildren()) {
      child.testInterrupted(now);
    }
  }

  @Override
  public void testSkipped(long now) {
    for (TestNode child : getChildren()) {
      child.testSkipped(now);
    }
  }

  @Override
  public void testSuppressed(long now) {
    for (TestNode child : getChildren()) {
      child.testSuppressed(now);
    }
  }

  void addTestSuite(TestSuiteNode suite) {
    children.add(suite);
  }

  void addTestCase(TestCaseNode testCase) {
    children.add(testCase);
  }

  @Override
  protected TestResult buildResult() {
    Interval runTime = null;
    int numTests = 0, numFailures = 0;
    LinkedList<TestResult> childResults = new LinkedList<>();

    for (TestNode child : children) {
      TestResult childResult = child.getResult();
      childResults.add(childResult);
      numTests += childResult.getNumTests();
      numFailures += childResult.getNumFailures();

      Interval childRunTime = childResult.getRunTimeInterval();
      if (childRunTime != null) {
        runTime = runTime == null ? childRunTime
            : new Interval(Math.min(runTime.getStartMillis(), childRunTime.getStartMillis()),
                         Math.max(runTime.getEndMillis(), childRunTime.getEndMillis()));
      }
    }

    return new TestResult.Builder()
        .name(getDescription().getDisplayName())
        .className("")
        .properties(Collections.<String, String>emptyMap())
        .failures(Collections.<Throwable>emptyList())
        .runTimeInterval(runTime)
        .status(Status.SKIPPED)
        .numTests(numTests)
        .numFailures(numFailures)
        .childResults(childResults)
        .build();
  }
}
