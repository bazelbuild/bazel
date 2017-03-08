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

package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.util.Collection;

/** This event is raised whenever a an individual test attempt is completed. */
public class TestAttempt implements BuildEvent {

  private final TestRunnerAction testAction;
  private final boolean success;
  private final int attempt;
  private final boolean lastAttempt;
  private final Collection<Pair<String, Path>> files;
  private final long durationMillis;

  /**
   * Construct the event given the test action and attempt number.
   *
   * @param testAction The test that was run.
   * @param attempt The number of the attempt for this action.
   */
  public TestAttempt(
      TestRunnerAction testAction,
      Integer attempt,
      boolean success,
      long durationMillis,
      Collection<Pair<String, Path>> files,
      boolean lastAttempt) {
    this.testAction = testAction;
    this.attempt = attempt;
    this.success = success;
    this.durationMillis = durationMillis;
    this.files = files;
    this.lastAttempt = lastAttempt;
  }

  public TestAttempt(
      TestRunnerAction testAction,
      Integer attempt,
      boolean success,
      Collection<Pair<String, Path>> files,
      boolean lastAttempt) {
    this(testAction, attempt, success, 0, files, lastAttempt);
  }

  public TestAttempt(
      TestRunnerAction testAction,
      Integer attempt,
      boolean success,
      Collection<Pair<String, Path>> files) {
    this(testAction, attempt, success, files, false);
  }

  public static TestAttempt fromCachedTestResult(TestResult result) {
    TestResultData data = result.getData();
    return new TestAttempt(
        result.getTestAction(), 1, data.getTestPassed(), data.getRunDurationMillis(),
        result.getFiles(), true);
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.testResult(
        testAction.getOwner().getLabel(),
        testAction.getRunNumber(),
        testAction.getShardNum(),
        attempt);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    if (lastAttempt) {
      return ImmutableList.of();
    } else {
      return ImmutableList.of(
          BuildEventId.testResult(
              testAction.getOwner().getLabel(),
              testAction.getRunNumber(),
              testAction.getShardNum(),
              attempt + 1));
    }
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(PathConverter pathConverter) {
    BuildEventStreamProtos.TestResult.Builder builder =
        BuildEventStreamProtos.TestResult.newBuilder();
    builder.setSuccess(success);
    builder.setTestAttemptDurationMillis(durationMillis);
    for (Pair<String, Path> file : files) {
      builder.addTestActionOutput(
          BuildEventStreamProtos.File.newBuilder()
              .setName(file.getFirst())
              .setUri(pathConverter.apply(file.getSecond()))
              .build());
    }
    return GenericBuildEvent.protoChaining(this).setTestResult(builder.build()).build();
  }
}
