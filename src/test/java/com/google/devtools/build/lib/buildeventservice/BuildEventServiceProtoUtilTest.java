// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.v1.BuildEvent;
import com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished;
import com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished.FinishType;
import com.google.devtools.build.v1.BuildEvent.BuildEnqueued;
import com.google.devtools.build.v1.BuildEvent.BuildFinished;
import com.google.devtools.build.v1.BuildEvent.InvocationAttemptFinished;
import com.google.devtools.build.v1.BuildEvent.InvocationAttemptStarted;
import com.google.devtools.build.v1.BuildStatus;
import com.google.devtools.build.v1.BuildStatus.Result;
import com.google.devtools.build.v1.OrderedBuildEvent;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.devtools.build.v1.PublishLifecycleEventRequest.ServiceLevel;
import com.google.devtools.build.v1.StreamId;
import com.google.devtools.build.v1.StreamId.BuildComponent;
import com.google.protobuf.Any;
import com.google.protobuf.Timestamp;
import com.google.protobuf.util.Timestamps;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BuildEventServiceProtoUtil}. * */
@RunWith(JUnit4.class)
public class BuildEventServiceProtoUtilTest {

  private static final String BUILD_REQUEST_ID = "feedbeef-dead-4321-beef-deaddeaddead";
  private static final String BUILD_INVOCATION_ID = "feedbeef-dead-4444-beef-deaddeaddead";
  private static final String PROJECT_ID = "my_project";
  private static final String COMMAND_NAME = "test";
  private static final String ADDITIONAL_KEYWORD = "keyword=foo";
  private static final ImmutableList<String> EXPECTED_KEYWORDS =
      ImmutableList.of("command_name=" + COMMAND_NAME, "protocol_name=BEP", ADDITIONAL_KEYWORD);
  private static final BuildEventServiceProtoUtil BES_PROTO_UTIL =
      new BuildEventServiceProtoUtil.Builder()
          .buildRequestId(BUILD_REQUEST_ID)
          .invocationId(BUILD_INVOCATION_ID)
          .projectId(PROJECT_ID)
          .commandName(COMMAND_NAME)
          .keywords(ImmutableSet.of(ADDITIONAL_KEYWORD))
          .attemptNumber(1)
          .build();
  private final ManualClock clock = new ManualClock();

  @Test
  public void testBuildEnqueued() {
    Timestamp expected = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.buildEnqueued(expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(EXPECTED_KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(1)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(expected)
                                .setBuildEnqueued(BuildEnqueued.getDefaultInstance())))
                .build());
  }

  @Test
  public void testInvocationAttemptStarted() {
    Timestamp expected = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.invocationStarted(expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(EXPECTED_KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(1)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(expected)
                                .setInvocationAttemptStarted(
                                    InvocationAttemptStarted.newBuilder().setAttemptNumber(1))))
                .build());
  }

  @Test
  public void invocationAttemptStarted_attemptNumber() {
    var besProtoUtil =
        new BuildEventServiceProtoUtil.Builder()
            .buildRequestId(BUILD_REQUEST_ID)
            .invocationId(BUILD_INVOCATION_ID)
            .projectId(PROJECT_ID)
            .commandName(COMMAND_NAME)
            .keywords(ImmutableSet.of(ADDITIONAL_KEYWORD))
            .attemptNumber(2)
            .build();
    Timestamp expected = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(besProtoUtil.invocationStarted(expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(EXPECTED_KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(1)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(expected)
                                .setInvocationAttemptStarted(
                                    InvocationAttemptStarted.newBuilder().setAttemptNumber(2))))
                .build());
  }

  @Test
  public void testInvocationAttemptFinished() {
    Timestamp expected = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.invocationFinished(expected, Result.COMMAND_SUCCEEDED))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(2)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(expected)
                                .setInvocationAttemptFinished(
                                    InvocationAttemptFinished.newBuilder()
                                        .setInvocationStatus(
                                            BuildStatus.newBuilder()
                                                .setResult(Result.COMMAND_SUCCEEDED)))))
                .build());
  }

  @Test
  public void testBuildFinished() {
    Timestamp expected = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.buildFinished(expected, Result.COMMAND_SUCCEEDED))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(EXPECTED_KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(2)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(expected)
                                .setBuildFinished(
                                    BuildFinished.newBuilder()
                                        .setStatus(
                                            BuildStatus.newBuilder()
                                                .setResult(Result.COMMAND_SUCCEEDED)))))
                .build());
  }

  @Test
  public void testStreamEvents() {
    Timestamp firstEventTimestamp = Timestamps.fromMillis(clock.advanceMillis(100));
    Any anything = Any.getDefaultInstance();
    assertThat(BES_PROTO_UTIL.bazelEvent(1, firstEventTimestamp, anything))
        .isEqualTo(
            PublishBuildToolEventStreamRequest.newBuilder()
                .addAllNotificationKeywords(EXPECTED_KEYWORDS)
                .setProjectId(PROJECT_ID)
                .setOrderedBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.TOOL))
                        .setSequenceNumber(1)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(firstEventTimestamp)
                                .setBazelEvent(anything))
                        .build())
                .setRetryAttemptNumber(1)
                .build());

    Timestamp secondEventTimestamp = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.bazelEvent(2, secondEventTimestamp, anything))
        .isEqualTo(
            PublishBuildToolEventStreamRequest.newBuilder()
                .setProjectId(PROJECT_ID)
                .setOrderedBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.TOOL))
                        .setSequenceNumber(2)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(secondEventTimestamp)
                                .setBazelEvent(anything))
                        .build())
                .setRetryAttemptNumber(1)
                .build());

    Timestamp thirdEventTimestamp = Timestamps.fromMillis(clock.advanceMillis(100));
    assertThat(BES_PROTO_UTIL.streamFinished(3, thirdEventTimestamp))
        .isEqualTo(
            PublishBuildToolEventStreamRequest.newBuilder()
                .setProjectId(PROJECT_ID)
                .setOrderedBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setInvocationId(BUILD_INVOCATION_ID)
                                .setComponent(BuildComponent.TOOL))
                        .setSequenceNumber(3)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(thirdEventTimestamp)
                                .setComponentStreamFinished(
                                    BuildComponentStreamFinished.newBuilder()
                                        .setType(FinishType.FINISHED)))
                        .build())
                .setRetryAttemptNumber(1)
                .build());
  }

  @Test
  public void testStreamEventsWithCheckPrecedingLifecycleEventsEnabled() {
    Any anything = Any.getDefaultInstance();
    BuildEventServiceProtoUtil besProtoUtil =
        new BuildEventServiceProtoUtil.Builder()
            .buildRequestId(BUILD_REQUEST_ID)
            .invocationId(BUILD_INVOCATION_ID)
            .commandName(COMMAND_NAME)
            .checkPrecedingLifecycleEvents(true)
            .keywords(ImmutableSet.of(ADDITIONAL_KEYWORD))
            .attemptNumber(1)
            .build();
    assertThat(
            besProtoUtil
                .bazelEvent(1, Timestamps.fromMillis(100), anything)
                .getCheckPrecedingLifecycleEventsPresent())
        .isTrue();
    // check_preceding_lifecycle_events_present is always false for events with sequence_number > 1.
    assertThat(
            besProtoUtil
                .bazelEvent(2, Timestamps.fromMillis(100), anything)
                .getCheckPrecedingLifecycleEventsPresent())
        .isFalse();
    assertThat(
            besProtoUtil
                .bazelEvent(3, Timestamps.fromMillis(100), anything)
                .getCheckPrecedingLifecycleEventsPresent())
        .isFalse();
  }
}
