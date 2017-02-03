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
import java.util.Collection;

/**
 * This event is raised whenever a an individual test attempt is completed that is not the final
 * attempt for the given test, shard, and run.
 */
public class TestAttempt implements BuildEvent {

  private final TestRunnerAction testAction;
  private final int attempt;

  /**
   * Construct the event given the test action and attempt number.
   *
   * @param testAction The test that was run.
   * @param attempt The number of the attempt for this action.
   */
  public TestAttempt(TestRunnerAction testAction, Integer attempt) {
    this.testAction = testAction;
    this.attempt = attempt;
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
    return ImmutableList.of(
        BuildEventId.testResult(
            testAction.getOwner().getLabel(),
            testAction.getRunNumber(),
            testAction.getShardNum(),
            attempt + 1));
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(PathConverter pathConverter) {
    BuildEventStreamProtos.TestResult.Builder resultBuilder =
        BuildEventStreamProtos.TestResult.newBuilder();
    return GenericBuildEvent.protoChaining(this).setTestResult(resultBuilder.build()).build();
  }
}
