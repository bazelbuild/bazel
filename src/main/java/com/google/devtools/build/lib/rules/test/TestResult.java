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

package com.google.devtools.build.lib.rules.test;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;

/**
 * This is the event passed from the various test strategies to the {@code RecordingTestListener}
 * upon test completion.
 */
@ThreadSafe
@Immutable
public class TestResult {

  private final TestRunnerAction testAction;
  private final TestResultData data;
  private final boolean cached;

  /**
   * Construct the TestResult for the given test / status.
   *
   * @param testAction The test that was run.
   * @param data test result protobuffer.
   * @param cached true if this is a cached test result.
   */
  public TestResult(TestRunnerAction testAction, TestResultData data, boolean cached) {
    this.testAction = Preconditions.checkNotNull(testAction);
    this.data = data;
    this.cached = cached;
  }

  public static boolean isBlazeTestStatusPassed(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED || status == BlazeTestStatus.FLAKY;
  }

  /**
   * @return The test action.
   */
  public TestRunnerAction getTestAction() {
    return testAction;
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
   * Return if result was loaded from local action cache.
   */
  public final boolean isCached() {
    return cached;
  }

  /**
   * @return Coverage data artifact, if available and null otherwise.
   */
  public PathFragment getCoverageData() {
    if (data.getHasCoverage()) {
      return testAction.getCoverageData().getExecPath();
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
