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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.util.List;
import javax.annotation.Nullable;

/**
 * This is the event passed from the various test strategies to the {@code RecordingTestListener}
 * upon test completion.
 */
@ThreadSafe
@Immutable
public class TestResult implements ExtendedEventHandler.Postable {

  private final TestRunnerAction testAction;
  private final TestResultData data;
  private final ImmutableMultimap<String, Path> testOutputs;
  private final boolean cached;
  @Nullable protected final Path execRoot;
  @Nullable private final DetailedExitCode systemFailure;

  /**
   * Construct the TestResult for the given test / status.
   *
   * @param testAction The test that was run.
   * @param data test result protobuffer.
   * @param cached true if this is a locally cached test result.
   * @param execRoot The execution root in which the action was carried out; can be null, in which
   *     case everything depending on the execution root is ignored.
   * @param systemFailure Description of the system failure responsible for the test not succeeding;
   *     null if no such failure occurred
   */
  public TestResult(
      TestRunnerAction testAction,
      TestResultData data,
      ImmutableMultimap<String, Path> testOutputs,
      boolean cached,
      @Nullable Path execRoot,
      @Nullable DetailedExitCode systemFailure) {
    this.testAction = checkNotNull(testAction);
    this.data = checkNotNull(data);
    this.testOutputs = checkNotNull(testOutputs);
    this.cached = cached;
    this.execRoot = execRoot;
    this.systemFailure = systemFailure;
  }

  public TestResult(
      TestRunnerAction testAction,
      TestResultData data,
      ImmutableMultimap<String, Path> testOutputs,
      boolean cached,
      @Nullable DetailedExitCode systemFailure) {
    this(testAction, data, testOutputs, cached, null, systemFailure);
  }

  public static boolean isBlazeTestStatusPassed(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED || status == BlazeTestStatus.FLAKY;
  }

  /** Returns the test action. */
  public TestRunnerAction getTestAction() {
    return testAction;
  }

  /**
   * Returns the test log path. Note, that actual log file may no longer correspond to this
   * artifact. Use getActualLogPath() method if you need log location.
   */
  public Path getTestLogPath() {
    Path testLogPath = testAction.getTestLog().getPath();
    // If we have an exec root we'll use its fileSystem
    if (execRoot != null) {
      FileSystem fileSystem = execRoot.getFileSystem();
      return fileSystem.getPath(testLogPath.asFragment());
    }
    return testLogPath;
  }

  /** Returns whether the result was loaded from local action cache. */
  public final boolean isCached() {
    return cached;
  }

  /**
   * Returns the list of locally cached test attempts. This method must only be called if {@link
   * #isCached} returns <code>true</code>.
   */
  public List<TestAttempt> getCachedTestAttempts() {
    Preconditions.checkState(isCached());
    return ImmutableList.of(
        TestAttempt.fromCachedTestResult(
            testAction,
            data,
            1,
            testOutputs,
            BuildEventStreamProtos.TestResult.ExecutionInfo.getDefaultInstance(),
            /* lastAttempt= */ true));
  }

  /** Returns the coverage data artifact, if available, and null otherwise. */
  @Nullable
  public Path getCoverageData() {
    if (data.getHasCoverage()) {
      return testAction.getCoverageData().getPath();
    }
    return null;
  }

  /** Returns the test status artifact. */
  public Artifact getTestStatusArtifact() {
    // these artifacts are used to keep track of the number of pending and completed tests.
    return testAction.getCacheStatusArtifact();
  }

  /**
   * Returns the test name in a user-friendly format. Will generally include the target name and
   * shard number, if applicable.
   */
  public String getTestName() {
    return testAction.getTestName();
  }

  /** Returns the test label. */
  public String getLabel() {
    return Label.print(testAction.getOwner().getLabel());
  }

  /** Returns the test shard number. */
  public int getShardNum() {
    return testAction.getShardNum();
  }

  /**
   * Returns the total number of test shards. 0 means no sharding, whereas 1 means degenerate
   * sharding.
   */
  public int getTotalShards() {
    return testAction.getExecutionSettings().getTotalShards();
  }

  public TestResultData getData() {
    return data;
  }

  /**
   * Returns the description of the system failure responsible for the test not succeeding or {@code
   * null} if no such failure occurred.
   */
  @Nullable
  public DetailedExitCode getSystemFailure() {
    return systemFailure;
  }
}
