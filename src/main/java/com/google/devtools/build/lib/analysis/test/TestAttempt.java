// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationId;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.runtime.BuildEventStreamerUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.protobuf.util.Durations;
import com.google.protobuf.util.Timestamps;
import java.util.Collection;
import java.util.List;

/** This event is raised whenever an individual test attempt is completed. */
public class TestAttempt implements BuildEventWithOrderConstraint {

  private final TestRunnerAction testAction;
  private final TestStatus status;
  private final String statusDetails;
  private final boolean cachedLocally;
  private final int attempt;
  private final boolean lastAttempt;
  private final Collection<Pair<String, Path>> files;
  private final List<String> testWarnings;
  private final long durationMillis;
  private final long startTimeMillis;
  private final BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo;

  /**
   * Construct the event given the test action and attempt number.
   *
   * @param cachedLocally True if the reported attempt is taken from the tool's local cache.
   * @param testAction The test that was run.
   * @param attempt The number of the attempt for this action.
   */
  private TestAttempt(
      boolean cachedLocally,
      TestRunnerAction testAction,
      BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo,
      int attempt,
      BlazeTestStatus status,
      String statusDetails,
      long startTimeMillis,
      long durationMillis,
      Collection<Pair<String, Path>> files,
      List<String> testWarnings,
      boolean lastAttempt) {
    this.testAction = testAction;
    this.executionInfo = Preconditions.checkNotNull(executionInfo);
    this.attempt = attempt;
    this.status = BuildEventStreamerUtils.bepStatus(Preconditions.checkNotNull(status));
    this.statusDetails = statusDetails;
    this.cachedLocally = cachedLocally;
    this.startTimeMillis = startTimeMillis;
    this.durationMillis = durationMillis;
    this.files = Preconditions.checkNotNull(files);
    this.testWarnings = Preconditions.checkNotNull(testWarnings);
    this.lastAttempt = lastAttempt;
  }

  /**
   * Creates a test attempt result instance for a test that was not locally cached; it may have been
   * locally executed, remotely executed, or remotely cached.
   */
  public static TestAttempt forExecutedTestResult(
      TestRunnerAction testAction,
      TestResultData attemptData,
      int attempt,
      Collection<Pair<String, Path>> files,
      BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo,
      boolean lastAttempt) {
    return new TestAttempt(
        false,
        testAction,
        executionInfo,
        attempt,
        attemptData.getStatus(),
        attemptData.getStatusDetails(),
        attemptData.getStartTimeMillisEpoch(),
        attemptData.getRunDurationMillis(),
        files,
        attemptData.getWarningList(),
        lastAttempt);
  }

  public static TestAttempt fromCachedTestResult(
      TestRunnerAction testAction,
      TestResultData attemptData,
      int attempt,
      Collection<Pair<String, Path>> files,
      BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo,
      boolean lastAttempt) {
    return new TestAttempt(
        true,
        testAction,
        executionInfo,
        attempt,
        attemptData.getStatus(),
        attemptData.getStatusDetails(),
        attemptData.getStartTimeMillisEpoch(),
        attemptData.getRunDurationMillis(),
        files,
        attemptData.getWarningList(),
        lastAttempt);
  }

  @VisibleForTesting
  public Artifact getTestStatusArtifact() {
    return testAction.getCacheStatusArtifact();
  }

  @VisibleForTesting
  public Collection<Pair<String, Path>> getFiles() {
    return files;
  }

  @VisibleForTesting
  public BuildEventStreamProtos.TestResult.ExecutionInfo getExecutionInfo() {
    return executionInfo;
  }

  @VisibleForTesting
  public TestStatus getStatus() {
    return status;
  }

  @VisibleForTesting
  public boolean isCachedLocally() {
    return cachedLocally;
  }

  @VisibleForTesting
  public int getAttempt() {
    return attempt;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.testResult(
        testAction.getOwner().getLabel(),
        testAction.getRunNumber(),
        testAction.getShardNum(),
        attempt,
        configurationId(testAction.getConfiguration()));
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(
        BuildEventIdUtil.targetCompleted(
            testAction.getOwner().getLabel(), configurationId(testAction.getConfiguration())));
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    if (lastAttempt) {
      return ImmutableList.of();
    } else {
      return ImmutableList.of(
          BuildEventIdUtil.testResult(
              testAction.getOwner().getLabel(),
              testAction.getRunNumber(),
              testAction.getShardNum(),
              attempt + 1,
              configurationId(testAction.getConfiguration())));
    }
  }

  @Override
  public ImmutableList<LocalFile> referencedLocalFiles() {
    LocalFileType localFileType =
        status == TestStatus.PASSED
            ? LocalFileType.SUCCESSFUL_TEST_OUTPUT
            : LocalFileType.FAILED_TEST_OUTPUT;
    ImmutableList.Builder<LocalFile> localFiles = ImmutableList.builder();
    for (Pair<String, Path> file : files) {
      if (file.getSecond() != null) {
        // TODO(b/199940216): Can we populate metadata for these files?
        localFiles.add(
            new LocalFile(
                file.getSecond(), localFileType, /*artifact=*/ null, /*artifactMetadata=*/ null));
      }
    }
    return localFiles.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this).setTestResult(asTestResult(converters)).build();
  }

  @VisibleForTesting
  public BuildEventStreamProtos.TestResult asTestResult(BuildEventContext converters) {
    PathConverter pathConverter = converters.pathConverter();
    BuildEventStreamProtos.TestResult.Builder builder =
        BuildEventStreamProtos.TestResult.newBuilder();
    builder.setStatus(status);
    builder.setStatusDetails(statusDetails);
    builder.setExecutionInfo(executionInfo);
    builder.setCachedLocally(cachedLocally);
    builder.setTestAttemptStart(Timestamps.fromMillis(startTimeMillis));
    builder.setTestAttemptStartMillisEpoch(startTimeMillis);
    builder.setTestAttemptDuration(Durations.fromMillis(durationMillis));
    builder.setTestAttemptDurationMillis(durationMillis);
    builder.addAllWarning(testWarnings);
    for (Pair<String, Path> file : files) {
      String uri = pathConverter.apply(file.getSecond());
      if (uri != null) {
        builder.addTestActionOutput(
            BuildEventStreamProtos.File.newBuilder().setName(file.getFirst()).setUri(uri).build());
      }
    }
    return builder.build();
  }
}
