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

import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.joda.time.Interval;

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
  private final Map<String, String> properties;
  private final List<Throwable> failures;
  @Nullable
  private final Interval runTime;
  private final Status status;
  private final int numTests, numFailures;
  private final List<TestResult> childResults;

  private TestResult(Builder builder) {
    name = checkNotNull(builder.name, "name not set");
    className = checkNotNull(builder.className, "className not set");
    properties = checkNotNull(builder.properties, "properties not set");
    failures = checkNotNull(builder.failures, "failures not set");
    runTime = builder.runTime;
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

  Map<String, String> getProperties() {
    return properties;
  }

  List<Throwable> getFailures() {
    return failures;
  }

  @Nullable
  Interval getRunTimeInterval() {
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

  List<TestResult> getChildResults() {
    return childResults;
  }

  private static <T> T checkNotNull(T reference, String errorMessage) {
    if (reference == null) {
      throw new NullPointerException(errorMessage);
    }
    return reference;
  }

  static final class Builder {
    private String name = null;
    private String className = null;
    private Map<String, String> properties = null;
    private List<Throwable> failures = null;
    @Nullable
    private Interval runTime = null;
    private Status status = null;
    private Integer numTests = null;
    private Integer numFailures = null;
    private List<TestResult> childResults = null;

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
      this.properties = checkNullToNotNull(this.properties, properties, "properties");
      return this;
    }

    Builder failures(List<Throwable> failures) {
      this.failures = checkNullToNotNull(this.failures, failures, "failures");
      return this;
    }

    Builder runTimeInterval(@Nullable Interval runTime) {
      if (this.runTime != null) {
        throw new IllegalStateException("runTime already set");
      }
      this.runTime = runTime;
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
      this.childResults = checkNullToNotNull(this.childResults, childResults, "childResults");
      return this;
    }

    TestResult build() {
      return new TestResult(this);
    }

    private static <T> T checkNullToNotNull(T currValue, T newValue, String desc) {
      if (currValue != null) {
        throw new IllegalStateException(desc + " already set");
      }
      return checkNotNull(newValue, desc + " is null");
    }
  }
}
