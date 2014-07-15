// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

/**
 * This is the event passed from the various test strategies to the {@code RecordingTestListener}
 * upon test completion.
 */
@ThreadSafe
@Immutable
public final class TestResultData {
  // This class is very similar to TestSummary. It would be nice to merge them.

  /**
   * How much information is present in a TestCaseDetails.
   */
  public enum FailedTestCaseDetailsStatus {
    /** Information about every test case is available. */
    FULL,
    /** Information about some test cases may be missing. */
    PARTIAL,
    /** No information about individual test cases. */
    NOT_AVAILABLE,
    /** This is an empty object still without data. */
    EMPTY,
  }

  /**
   * The collection of test case details for a test run. Contains a list of test
   * case results and a status field to show how much information is available.
   *
   * <p>The test cases are sorted according to their name.
   */
  public static class FailedTestCaseDetails {
    private TreeSet<TestCaseDetail> details;
    private FailedTestCaseDetailsStatus status;

    public FailedTestCaseDetails(FailedTestCaseDetailsStatus status) {
      this.details = Sets.newTreeSet(new Comparator<TestCaseDetail>() {
        @Override
        public int compare(TestCaseDetail o1, TestCaseDetail o2) {
          return o1.getName().compareTo(o2.getName());
        }
      });
      this.status = status;
    }

    public FailedTestCaseDetailsStatus getStatus() {
      return status;
    }

    public ImmutableList<TestCaseDetail> getDetails() {
      return ImmutableList.copyOf(details);
    }

    public void add(TestCaseDetail newDetail) {
      details.add(newDetail);
    }

    public void addAll(Collection<TestCaseDetail> newDetails) {
      details.addAll(newDetails);
    }

    public void mergeFrom(FailedTestCaseDetails that) {
      this.details.addAll(that.details);

      if (this.status == FailedTestCaseDetailsStatus.EMPTY) {

        // If there was no data yet, just copy the status.
        this.status = that.status;
      } else if (that.status == FailedTestCaseDetailsStatus.EMPTY) {

        // If the other one was empty, keep our old status.
        return;
      } else if (this.status != that.status) {

        // Otherwise, if the statuses were different, change status to partial.
        this.status = FailedTestCaseDetailsStatus.PARTIAL;
      }
    }

    void setStatus(FailedTestCaseDetailsStatus status) {
      this.status = status;
    }
  }

  /**
   * The status of an individual test case. Other test results are not needed
   * here, because if we get back information from a test run, we have at least
   * tried to run it.
   */
  public enum TestCaseStatus { PASSED, FAILED, ERROR }

  /**
   * The summary of an individual test case
   */
  public static class TestCaseDetail {
    private final String name;
    private final TestCaseStatus status;
    private final Long runDurationMillis;       // so that it can be null

    public TestCaseDetail(
        String name, TestCaseStatus status, Long runDurationMillis) {
      this.name = name;
      this.status = status;
      this.runDurationMillis = runDurationMillis;
    }

    public String getName() {
      return name;
    }

    public TestCaseStatus getStatus() {
      return status;
    }

    public Long getRunDurationMillis() {
      return runDurationMillis;
    }
  }

  private boolean isCached;
  private BlazeTestStatus status = BlazeTestStatus.NO_STATUS;
  private List<Path> failedLogs = ImmutableList.of();
  private FailedTestCaseDetails failedTestCaseDetails;
  private List<String> warnings = ImmutableList.of();
  private boolean hasCoverage;
  private boolean remotelyCached;
  private boolean isRemoteStrategy;
  private List<Long> testTimes = ImmutableList.of();
  private List<Path> passedLogs = ImmutableList.of();
  private List<Long> testProcessTimes = ImmutableList.of();

  // Don't allow public instantiation; go through the Builder.
  private TestResultData() {
  }

  /** Create a TestResultData instance. */
  public static class Builder {
    TestResultData data;
    Builder() {
      data = new TestResultData();
    }

    TestResultData build() {
      TestResultData built = data;
      data = new TestResultData();
      return built;
    }

    public Builder setTestTimes(List<Long> times) {
      data.testTimes = ImmutableList.copyOf(times);
      return this;
    }
    public Builder setTestProcessTimes(List<Long> times) {
      data.testProcessTimes = ImmutableList.copyOf(times);
      return this;
    }
    public Builder setIsCached(boolean isCached) {
      data.isCached = isCached;
      return this;
    }
    public Builder setStatus(BlazeTestStatus status) {
      data.status = status;
      return this;
    }
    public Builder setFailedLogs(List<Path> failedLogs) {
      data.failedLogs = ImmutableList.copyOf(failedLogs);
      return this;
    }
    public Builder setFailedTestCaseDetails(FailedTestCaseDetails failedTestCases) {
      data.failedTestCaseDetails = failedTestCases;
      return this;
    }
    public Builder setWarnings(List<String> warnings) {
      data.warnings = ImmutableList.copyOf(warnings);
      return this;
    }
    public Builder setHasCoverage(boolean hasCoverage) {
      data.hasCoverage = hasCoverage;
      return this;
    }
    public Builder setRemotelyCached(boolean remotelyCached) {
      data.remotelyCached = remotelyCached;
      return this;
    }
    public Builder setIsRemoteStrategy(boolean remote) {
      data.isRemoteStrategy = remote;
      return this;
    }
    public Builder setPassedLogs(List<Path> logs) {
      data.passedLogs = ImmutableList.copyOf(logs);
      return this;
    }
  }

  /**
   * Creates a new Builder allowing construction of a new TestSummary object.
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * @return The test status.
   */
  public BlazeTestStatus getStatus() {
    return status;
  }

  /**
   * @return true iff result represents cached test result.
   */
  public boolean isCached() {
    return isCached;
  }

  /**
   * @return true iff result represents successful test.
   */
  public boolean isPassed() {
    return status == BlazeTestStatus.PASSED;
  }

  /**
   * @return immutable list of failed log paths.
   */
  public List<Path> getFailedLogs() {
    return failedLogs;
  }

  /**
   * @return immutable list of failed test case names.
   */
  public FailedTestCaseDetails getFailedTestCaseDetails() {
    return failedTestCaseDetails;
  }

  /**
   * @return immutable list of passed log paths (either 0 or 1 entries).
   */
  public List<Path> getPassedLogs() {
    return passedLogs;
  }

  /**
   * @return immutable list of warnings.
   */
  public List<String> getWarnings() {
    return warnings;
  }

  /**
   * @return immutable list of all associated test times (in ms).
   */
  public List<Long> getTestTimes() {
    return testTimes;
  }

  /**
   * @return immutable list of all associated test times (in ms).
    */
  public List<Long> getTestProcessTimes() {
    return testProcessTimes;
  }

  /**
   * @return Coverage data artifact, if available and null otherwise.
   */
  public boolean hasCoverage() {
    return hasCoverage;
  }

  /**
   * Returns if this was cached in remote execution.
   */
  public boolean isRemotelyCached() {
    return remotelyCached;
  }

  public boolean isRemoteStrategy() {
    return isRemoteStrategy;
  }
}
