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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import org.joda.time.Interval;

import java.util.List;
import java.util.Map;

/**
 * Result of executing a test suite or test case.
 */
final class TestResult {

  /**
   * Possible result values to a test.
   */
  enum Status {
    /**
     * Test case was not run because the test decided that it should not be run 
     * (e.g.: due to a failed assumption in a JUnit4-style tests).
     */
    SKIPPED(false),

    /**
     * Test case was not run because the user specified that it should be filtered out of the
     * test run.
     */
    FILTERED(false),

    /**
     * Test case was not run because the test was labeled in the code as suppressed 
     * (e.g.: the test was annotated with {@code @Suppress} or {@code @Ignore}).
     */
    SUPPRESSED(false),

    /**
     * Test case was not started because the test harness run was interrupted by a 
     * signal or timed out.
     */
    CANCELLED(false),

    /**
     * Test case was started but not finished because the test harness run was interrupted by a 
     * signal or timed out.
     */
    INTERRUPTED(true),

    /**
     * Test case was run and completed (possibly failing or throwing an exception, but not 
     * interrupted).
     */
    COMPLETED(true);

    private final boolean wasRun;

    Status(boolean wasRun) {
      this.wasRun = wasRun;
    }

    /**
     * Equivalent semantic value to wasRun {@code status="run|notrun"} on 
     * the XML schema.
     */
    public boolean wasRun() {
      return wasRun;
    }
  }

  private final String name, className;
  private final ImmutableMap<String, String> properties;
  private final ImmutableList<Throwable> failures;
  private final Optional<Interval> runTime;
  private final Status status;
  private final int numTests, numFailures;
  private final ImmutableList<TestResult> childResults;

  private TestResult(Builder builder) {
    name = checkNotNull(builder.name, "name not set");
    className = checkNotNull(builder.className, "className not set");
    properties = checkNotNull(builder.properties, "properties not set");
    failures = checkNotNull(builder.failures, "failures not set");
    runTime = checkNotNull(builder.runTime, "runTime not set");
    status = checkNotNull(builder.status, "status not set");
    numTests = checkNotNull(builder.numTests, "numTests not set");
    numFailures = checkNotNull(builder.numFailures, "numFailures not set");
    childResults = checkNotNull(builder.childResults, "childResults not set");
  }

  String getName() {
    return name;
  }

  String getClassName() {
    return className;
  }

  ImmutableMap<String, String> getProperties() {
    return properties;
  }

  ImmutableList<Throwable> getFailures() {
    return failures;
  }

  Optional<Interval> getRunTimeInterval() {
    return runTime;
  }

  Status getStatus() {
    return status;
  }

  boolean wasRun() {
    return getStatus().wasRun();
  }

  int getNumTests() {
    return numTests;
  }

  int getNumFailures() {
    return numFailures;
  }

  ImmutableList<TestResult> getChildResults() {
    return childResults;
  }

  static final class Builder {
    private String name = null;
    private String className = null;
    private ImmutableMap<String, String> properties = null;
    private ImmutableList<Throwable> failures = null;
    private Optional<Interval> runTime = null;
    private Status status = null;
    private Integer numTests = null;
    private Integer numFailures = null;
    private ImmutableList<TestResult> childResults = null;

    Builder() {}

    Builder name(String name) {
      this.name = checkNullToNotNull(this.name, name, "name");
      return this;
    }

    Builder className(String className) {
      this.className = checkNullToNotNull(this.className, className, "className");
      return this;
    }

    Builder properties(Map<String, String> properties) {
      this.properties = ImmutableMap.copyOf(
          checkNullToNotNull(this.properties, properties, "properties"));
      return this;
    }

    Builder failures(List<Throwable> failures) {
      this.failures = ImmutableList.copyOf(
          checkNullToNotNull(this.failures, failures, "failures"));
      return this;
    }

    Builder runTimeInterval(Optional<Interval> runTime) {
      this.runTime = checkNullToNotNull(this.runTime, runTime, "runTime");
      return this;
    }

    Builder status(Status status) {
      this.status = checkNullToNotNull(this.status, status, "status");
      return this;
    }

    Builder numTests(int numTests) {
      this.numTests = checkNullToNotNull(this.numTests, numTests, "numTests");
      return this;
    }

    Builder numFailures(int numFailures) {
      this.numFailures = checkNullToNotNull(this.numFailures, numFailures, "numFailures");
      return this;
    }

    Builder childResults(List<TestResult> childResults) {
      this.childResults = ImmutableList.copyOf(
          checkNullToNotNull(this.childResults, childResults, "childResults"));
      return this;
    }

    TestResult build() {
      return new TestResult(this);
    }

    private static <T> T checkNullToNotNull(T currValue, T newValue, String desc) {
      checkState(currValue == null, desc + " already set");
      return checkNotNull(newValue, desc + " is null");
    }
  }
}
