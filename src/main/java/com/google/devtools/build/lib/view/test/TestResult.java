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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.proto.HierarchicalTestResult;
import com.google.testing.proto.TestStatus;
import com.google.testing.proto.TestStrategy;
import com.google.testing.proto.TestTargetResult;
import com.google.testing.proto.TestWarning;
import com.google.testing.proto.TimingBreakdown;

import java.io.IOException;
import java.util.ArrayList;
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
public class TestResult {

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

  private final TestRunnerAction testAction;
  private final TestTargetResult result;
  private final boolean isCached;

  /**
   * Construct a cached TestResult instance from the saved test result
   * protobuffer if possible.
   *
   * @return new TestResult instance or null if it cannot be created.
   * @throws IOException on underlying or protocol buffer parsing failure.
   */
  public static TestResult createCached(TestRunnerAction testAction) throws IOException {
    return new TestResult(testAction,
        TestTargetResult.parseFrom(testAction.getTestTargetResult().getPath().getInputStream()),
        true);
  }

  /**
   * Construct new TestResult instance for the given test / status.
   *
   * @param testAction The test that was run.
   * @param result test result protobuffer.
   */
  public static TestResult createNew(TestRunnerAction testAction, TestTargetResult result) {
    return new TestResult(testAction, result, false);
  }

  /**
   * Construct the TestResult for the given test / status.
   *
   * @param testAction The test that was run.
   * @param result test result protobuffer.
   * @param isCached true if this is a cached test result.
   */
  private TestResult(TestRunnerAction testAction, TestTargetResult result, boolean isCached) {
    this.testAction = Preconditions.checkNotNull(testAction);
    if (result.hasCombinedOut()) {
      Preconditions.checkState(testAction.getTestLog().getPath().getPathString().equals(
          result.getCombinedOut().getPathname()));
    }
    this.result = result;
    this.isCached = isCached;
  }

  /**
   * @return The test action.
   */
  public TestRunnerAction getTestAction() {
    return testAction;
  }

  /**
   * @return The test result protobuffer.
   */
  public TestTargetResult getResult() {
    // TODO(bazel-team): refactor so Bazel does not need to expose this.
    return result;
  }

  /**
   * @return The test status.
   */
  public BlazeTestStatus getStatus() {
    return BlazeTestStatus.getStatusFromTestTargetResult(result);
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
    return result.getStatus() == TestStatus.PASSED;
  }

  /**
   * @return immutable list of failed log paths.
   */
  public List<Path> getFailedLogs() {
    List<Path> list = new ArrayList<>();
    Path basePath = testAction.getTestLog().getPath();
    if (!isPassed() && result.hasCombinedOut()) {
      list.add(basePath.getRelative(result.getCombinedOut().getPathname()));
    }
    for (TestTargetResult attempt : result.getAttemptsList()) {
      if (attempt.hasCombinedOut()) {
        list.add(basePath.getRelative(attempt.getCombinedOut().getPathname()));
      }
    }
    return ImmutableList.copyOf(list);
  }

  /**
   * @return immutable list of failed test case names.
   */
  public FailedTestCaseDetails getFailedTestCaseDetails() {
    if (!result.hasHierarchicalTestResult()) {
      return new FailedTestCaseDetails(FailedTestCaseDetailsStatus.NOT_AVAILABLE);
    }

    FailedTestCaseDetails testCaseDetails =
      new FailedTestCaseDetails(FailedTestCaseDetailsStatus.FULL);

    collectNotPassedTestCases(result.getHierarchicalTestResult(), testCaseDetails);
    return testCaseDetails;
  }

  private void collectNotPassedTestCases(
      HierarchicalTestResult hierarchicalResult, FailedTestCaseDetails testCaseDetails) {
    if (hierarchicalResult.getChildCount() > 0) {
      // This is a non-leaf result. Traverse its children, but do not add its
      // name to the output list. It should not contain any 'failure' or
      // 'error' tags, but we want to be lax here, because the syntax of the
      // test.xml file is also lax.
      for (HierarchicalTestResult child : hierarchicalResult.getChildList()) {
        collectNotPassedTestCases(child, testCaseDetails);
      }
    } else {
      // This is a leaf result. If there was a failure or an error, return it.
      boolean passed = hierarchicalResult.getFailureCount() == 0
                    && hierarchicalResult.getErrorCount() == 0;
      if (passed) {
        return;
      }

      String name = hierarchicalResult.getName();
      String className = hierarchicalResult.getClassName();
      if (name == null || className == null) {
        // A test case detail is not really interesting if we cannot tell which
        // one it is.
        testCaseDetails.setStatus(FailedTestCaseDetailsStatus.PARTIAL);
        return;
      }

      TestCaseStatus status = hierarchicalResult.getErrorCount() > 0
          ? TestCaseStatus.ERROR : TestCaseStatus.FAILED;

      Long runDurationMillis = hierarchicalResult.hasRunDurationMillis()
          ? hierarchicalResult.getRunDurationMillis() : null;

      // TODO(bazel-team): The dot separator is only correct for Java.
      String testCaseName = className + "." + name;

      testCaseDetails.add(new TestCaseDetail(testCaseName, status, runDurationMillis));
    }
  }

  /**
   * @return immutable list of passed log paths (either 0 or 1 entries).
   */
  public List<Path> getPassedLogs() {
    return (isPassed() && result.hasCombinedOut())
        ? ImmutableList.<Path>of(testAction.getTestLog().getPath().getRelative(
            result.getCombinedOut().getPathname()))
        : ImmutableList.<Path>of();
  }

  /**
   * @return immutable list of warnings.
   */
  public List<String> getWarnings() {
    List<String> warnings = new ArrayList<>();

    for (TestTargetResult attempt : result.getAttemptsList()) {
      for (TestWarning warning : attempt.getWarningList()) {
        warnings.add(warning.getWarningMessage());
      }
    }

    for (TestWarning warning : result.getWarningList()) {
      warnings.add(warning.getWarningMessage());
    }
    return ImmutableList.copyOf(warnings);
  }

  /**
   * @return immutable list of all associated test times (in ms).
   */
  public List<Long> getTestTimes() {
    List<Long> list = Lists.newArrayList(result.getRunDurationMillis());
    for (TestTargetResult attempt : result.getAttemptsList()) {
      list.add(attempt.getRunDurationMillis());
    }
    return ImmutableList.copyOf(list);
  }

  /**
   * @return immutable list of all associated test times (in ms).
   * Unlike getTestTimes(), does not include remote execution overhead.
   */
  public List<Long> getTestProcessTimes() {
    List<Long> list = Lists.newArrayList(getTestProcessTime(result));
    for (TestTargetResult attempt : result.getAttemptsList()) {
      list.add(getTestProcessTime(attempt));
    }
    return ImmutableList.copyOf(list);
  }

  private static long getTestProcessTime(TestTargetResult result) {
    for (TimingBreakdown child : result.getTimingBreakdown().getChildList()) {
      if (child.getName().equals("test process")) {
        return child.getTimeMillis();
      }
    }
    throw new RuntimeException("Test does not have 'test process' timing breakdown field");
  }

  /**
   * @return The test log artifact. Note, that actual log file may no longer
   *         correspond to this artifact - use getActualLogPath() method if
   *         you need log location.
   */
  public Artifact getTestLogArtifact() {
    return testAction.getTestLog();
  }

  /**
   * @return Coverage data artifact, if available and null otherwise.
   */
  public PathFragment getCoverageData() {
    if (result.hasCoverage()) {
      Preconditions.checkState(
          result.getCoverage().hasLcov() && result.getCoverage().getLcov().hasPathname());
      Preconditions.checkState(
          result.getCoverage().getLcov().getPathname().endsWith(
              testAction.getCoverageData().getPathString()));
      return testAction.getCoverageData();
    }
    return null;
  }

  /**
   * @return The test status artifact.
   */
  public Artifact getTestStatusArtifact() {
    return testAction.getTestTargetResult();
  }

  /**
   * Gets the test name in a user-friendly format.
   * Will generally include the target name and shard number, if applicable.
   *
   * @return The test name.
   */
  public String getTestName() {
    return testAction.getTestName();
  }

  /**
   * @return The test label.
   */
  public String getLabel() {
    return Label.print(testAction.getOwner().getLabel());
  }

  /**
   * @return The test shard number.
   */
  public int getShardNum() {
    return testAction.getShardNum();
  }

  /**
   * @return Total number of test shards. 0 means
   *     no sharding, whereas 1 means degenerate sharding.
   */
  public int getTotalShards() {
    return testAction.getExecutionSettings().getTotalShards();
  }

  /**
   * Returns if this was cached in remote execution.
   */
  public boolean isRemotelyCached() {
    // TODO(bazel-team): see if this can be folded into isCached().
    return result.getRemoteCacheHit();
  }

  public TestStrategy getStrategy() {
    return result.getStrategy();
  }
}
