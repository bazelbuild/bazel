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

import com.google.testing.junit.runner.util.TestIntegration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** Result of executing a test suite or test case. */
public final class TestResult {

  /**
   * Possible result values to a test.
   */
  public enum Status {
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

  private final String name;
  private final String className;
  private final Map<String, String> properties;
  private final List<Throwable> failures;
  @Nullable private final TestInterval runTime;
  private final Set<TestIntegration> integrations;
  private final Status status;
  private final int numTests;
  private final int numFailures;
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
    integrations = checkNotNull(builder.integrations, "integrations not set");
  }

  public String getName() {
    return name;
  }

  public String getClassName() {
    return className;
  }

  public Map<String, String> getProperties() {
    return properties;
  }

  public List<Throwable> getFailures() {
    return failures;
  }

  public Set<TestIntegration> getIntegrations() {
    return integrations;
  }

  @Nullable
  public TestInterval getRunTimeInterval() {
    return runTime;
  }

  public Status getStatus() {
    return status;
  }

  public boolean wasRun() {
    return getStatus().wasRun();
  }

  public int getNumTests() {
    return numTests;
  }

  public int getNumFailures() {
    return numFailures;
  }

  public List<TestResult> getChildResults() {
    return childResults;
  }

  private static <T> T checkNotNull(T reference, String errorMessage) {
    if (reference == null) {
      throw new NullPointerException(errorMessage);
    }
    return reference;
  }

  public static final class Builder {
    private String name = null;
    private String className = null;
    private Map<String, String> properties = null;
    private List<Throwable> failures = null;
    @Nullable private TestInterval runTime = null;
    private Set<TestIntegration> integrations = null;
    private Status status = null;
    private Integer numTests = null;
    private Integer numFailures = null;
    private List<TestResult> childResults = null;

    public Builder() {}

    public Builder name(String name) {
      this.name = checkNullToNotNull(this.name, name, "name");
      return this;
    }

    public Builder className(String className) {
      this.className = checkNullToNotNull(this.className, className, "className");
      return this;
    }

    public Builder properties(Map<String, String> properties) {
      this.properties = checkNullToNotNull(this.properties, properties, "properties");
      return this;
    }

    public Builder integrations(Set<TestIntegration> integrations) {
      this.integrations = checkNullToNotNull(this.integrations, integrations, "integrations");
      return this;
    }

    public Builder failures(List<Throwable> failures) {
      this.failures = checkNullToNotNull(this.failures, failures, "failures");
      return this;
    }

    public Builder runTimeInterval(@Nullable TestInterval runTime) {
      if (this.runTime != null) {
        throw new IllegalStateException("runTime already set");
      }
      this.runTime = runTime;
      return this;
    }

    public Builder status(Status status) {
      this.status = checkNullToNotNull(this.status, status, "status");
      return this;
    }

    public Builder numTests(int numTests) {
      this.numTests = checkNullToNotNull(this.numTests, numTests, "numTests");
      return this;
    }

    public Builder numFailures(int numFailures) {
      this.numFailures = checkNullToNotNull(this.numFailures, numFailures, "numFailures");
      return this;
    }

    public Builder childResults(List<TestResult> childResults) {
      this.childResults = checkNullToNotNull(this.childResults, childResults, "childResults");
      return this;
    }

    public TestResult build() {
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
