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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestResultData.FailedTestCaseDetails;
import com.google.devtools.build.lib.view.test.TestResultData.FailedTestCaseDetailsStatus;
import com.google.devtools.build.lib.view.test.TestResultData.TestCaseDetail;
import com.google.devtools.build.lib.view.test.TestResultData.TestCaseStatus;
import com.google.testing.proto.HierarchicalTestResult;
import com.google.testing.proto.TestStatus;
import com.google.testing.proto.TestStrategy;
import com.google.testing.proto.TestTargetResult;
import com.google.testing.proto.TestWarning;
import com.google.testing.proto.TimingBreakdown;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This is the event passed from the various test strategies to the {@code RecordingTestListener}
 * upon test completion.
 */
@ThreadSafe
@Immutable
public class TestResult {

  private final TestRunnerAction testAction;
  private final TestTargetResult result;
  private final TestResultData data;

  /**
   * Construct a cached TestResult instance from the saved test result
   * protobuffer if possible.
   *
   * @return new TestResult instance or null if it cannot be created.
   * @throws IOException on underlying or protocol buffer parsing failure.
   */
  public static TestResult createCached(TestRunnerAction testAction) throws IOException {
    TestTargetResult result = TestTargetResult.parseFrom(
        testAction.getTestTargetResult().getInputStream());

    return new TestResult(testAction, fromTestTargetResult(result, testAction, true), result);
  }

  /**
   * Construct new TestResult instance for the given test / status.
   *
   * @param testAction The test that was run.
   * @param result test result protobuffer.
   */
  public static TestResult createNew(TestRunnerAction testAction, TestTargetResult result) {
    return new TestResult(testAction, fromTestTargetResult(result, testAction, false), result);
  }

  /**
   * Construct the TestResult for the given test / status.
   *
   * @param testAction The test that was run.
   * @param result test result protobuffer.
   * @param isCached true if this is a cached test result.
   */
  private TestResult(TestRunnerAction testAction, TestResultData data, TestTargetResult result) {
    this.testAction = Preconditions.checkNotNull(testAction);
    this.result = result;
    this.data  = data;
  }

  private static TestResultData fromTestTargetResult(
      TestTargetResult result, TestRunnerAction action,
      boolean cached) {
    TestResultData.Builder data = TestResultData.newBuilder();
    data.setIsCached(cached);
    data.setStatus(BlazeTestStatus.getStatusFromTestTargetResult(result));
    data.setRemotelyCached(false);
    data.setIsRemoteStrategy(false);

    boolean isPassed = result.getStatus() == TestStatus.PASSED;
    if (result.hasCombinedOut()) {
      Preconditions.checkState(action.getTestLog().getPath().getPathString().equals(
          result.getCombinedOut().getPathname()));
    }

    {
      List<Path> list = new ArrayList<>();
      Path basePath = action.getTestLog().getPath();
      if (!isPassed && result.hasCombinedOut()) {
        list.add(basePath.getRelative(result.getCombinedOut().getPathname()));
      }
      for (TestTargetResult attempt : result.getAttemptsList()) {
        if (attempt.hasCombinedOut()) {
          list.add(basePath.getRelative(attempt.getCombinedOut().getPathname()));
        }
      }
      data.setFailedLogs(list);
    }

    {
      FailedTestCaseDetails details;
      if (!result.hasHierarchicalTestResult()) {
        details = new FailedTestCaseDetails(FailedTestCaseDetailsStatus.NOT_AVAILABLE);
      } else {
        details = new FailedTestCaseDetails(FailedTestCaseDetailsStatus.FULL);
        collectNotPassedTestCases(result.getHierarchicalTestResult(), details);
      }
      data.setFailedTestCaseDetails(details);
    }

    if (isPassed && result.hasCombinedOut()) {
      data.setPassedLogs(ImmutableList.<Path>of(action.getTestLog().getPath().getRelative(
          result.getCombinedOut().getPathname())));
    }


    {
      List<String> warnings = new ArrayList<>();

      for (TestTargetResult attempt : result.getAttemptsList()) {
        for (TestWarning warning : attempt.getWarningList()) {
          warnings.add(warning.getWarningMessage());
        }
      }

      for (TestWarning warning : result.getWarningList()) {
        warnings.add(warning.getWarningMessage());
      }
      data.setWarnings(warnings);
    }

    {
      List<Long> list = Lists.newArrayList(result.getRunDurationMillis());
      for (TestTargetResult attempt : result.getAttemptsList()) {
        list.add(attempt.getRunDurationMillis());
      }
      data.setTestTimes(list);
    }

    {
      List<Long> list = Lists.newArrayList(getTestProcessTime(result));
      for (TestTargetResult attempt : result.getAttemptsList()) {
        list.add(getTestProcessTime(attempt));
      }
      data.setTestProcessTimes(list);
    }

    if (result.hasCoverage()) {
      data.setHasCoverage(true);
      Preconditions.checkState(
          result.getCoverage().hasLcov() && result.getCoverage().getLcov().hasPathname());
      Preconditions.checkState(
          result.getCoverage().getLcov().getPathname().endsWith(
              action.getCoverageData().getPathString()));
    }
    return data.build();
  }

  private static long getTestProcessTime(TestTargetResult result) {
    for (TimingBreakdown child : result.getTimingBreakdown().getChildList()) {
      if (child.getName().equals("test process")) {
        return child.getTimeMillis();
      }
    }
    return 0;
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

  private static void collectNotPassedTestCases(
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
   * @return The test log path. Note, that actual log file may no longer
   *         correspond to this artifact - use getActualLogPath() method if
   *         you need log location.
   */
  public Path getTestLogPath() {
    return testAction.getTestLog().getPath();
  }

  /**
   * @return Coverage data artifact, if available and null otherwise.
   */
  public PathFragment getCoverageData() {
    if (data.hasCoverage()) {
      return testAction.getCoverageData();
    }
    return null;
  }

  /**
   * @return The test status artifact.
   */
  public Artifact getTestStatusArtifact() {
    // these artifacts are used to keep track of the number of pending and completed tests.
    return testAction.getCacheStatusArtifact();
  }

  public Path getTestTargetResult() {
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

  public TestResultData getData() {
    return data;
  }
}
