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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.CommandContext;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.InvocationStatus;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceProtoUtil;
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
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import java.time.Instant;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BuildEventServiceProtoUtil}. * */
@RunWith(JUnit4.class)
public class BuildEventServiceProtoUtilTest {

  private static final String BUILD_REQUEST_ID = "feedbeef-dead-4321-beef-deaddeaddead";
  private static final String BUILD_INVOCATION_ID = "feedbeef-dead-4444-beef-deaddeaddead";
  private static final int ATTEMPT_NUMBER = 1;
  private static final String PROJECT_ID = "my_project";
  private static final ImmutableSet<String> KEYWORDS = ImmutableSet.of("foo=bar", "spam=eggs");
  private static final CommandContext COMMAND_CONTEXT =
      CommandContext.builder()
          .setBuildId(BUILD_REQUEST_ID)
          .setInvocationId(BUILD_INVOCATION_ID)
          .setAttemptNumber(ATTEMPT_NUMBER)
          .setKeywords(KEYWORDS)
          .setProjectId(PROJECT_ID)
          .setCheckPrecedingLifecycleEvents(false)
          .build();
  private final ManualClock clock = new ManualClock();

  @Test
  public void testBuildEnqueued() {
    clock.advanceMillis(100);
    Instant expected = clock.now();
    assertThat(BuildEventServiceProtoUtil.buildEnqueued(COMMAND_CONTEXT, expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(1)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(toProtoTimestamp(expected))
                                .setBuildEnqueued(BuildEnqueued.getDefaultInstance())))
                .build());
  }

  @Test
  public void testInvocationAttemptStarted() {
    clock.advanceMillis(100);
    Instant expected = clock.now();
    assertThat(BuildEventServiceProtoUtil.invocationStarted(COMMAND_CONTEXT, expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(KEYWORDS)
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
                                .setEventTime(toProtoTimestamp(expected))
                                .setInvocationAttemptStarted(
                                    InvocationAttemptStarted.newBuilder().setAttemptNumber(1))))
                .build());
  }

  @Test
  public void invocationAttemptStarted_attemptNumber() {
    var commandContext =
        CommandContext.builder()
            .setBuildId(BUILD_REQUEST_ID)
            .setInvocationId(BUILD_INVOCATION_ID)
            .setAttemptNumber(2)
            .setKeywords(KEYWORDS)
            .setProjectId(PROJECT_ID)
            .setCheckPrecedingLifecycleEvents(false)
            .build();
    clock.advanceMillis(100);
    Instant expected = clock.now();
    assertThat(BuildEventServiceProtoUtil.invocationStarted(commandContext, expected))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(KEYWORDS)
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
                                .setEventTime(toProtoTimestamp(expected))
                                .setInvocationAttemptStarted(
                                    InvocationAttemptStarted.newBuilder().setAttemptNumber(2))))
                .build());
  }

  @Test
  public void testInvocationAttemptFinished() {
    clock.advanceMillis(100);
    Instant expected = clock.now();
    assertThat(
            BuildEventServiceProtoUtil.invocationFinished(
                COMMAND_CONTEXT, expected, InvocationStatus.SUCCEEDED))
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
                                .setEventTime(toProtoTimestamp(expected))
                                .setInvocationAttemptFinished(
                                    InvocationAttemptFinished.newBuilder()
                                        .setInvocationStatus(
                                            BuildStatus.newBuilder()
                                                .setResult(Result.COMMAND_SUCCEEDED)))))
                .build());
  }

  @Test
  public void testBuildFinished() {
    clock.advanceMillis(100);
    Instant expected = clock.now();
    assertThat(
            BuildEventServiceProtoUtil.buildFinished(
                COMMAND_CONTEXT, expected, InvocationStatus.SUCCEEDED))
        .isEqualTo(
            PublishLifecycleEventRequest.newBuilder()
                .setServiceLevel(ServiceLevel.INTERACTIVE)
                .setProjectId(PROJECT_ID)
                .addAllNotificationKeywords(KEYWORDS)
                .setBuildEvent(
                    OrderedBuildEvent.newBuilder()
                        .setStreamId(
                            StreamId.newBuilder()
                                .setBuildId(BUILD_REQUEST_ID)
                                .setComponent(BuildComponent.CONTROLLER))
                        .setSequenceNumber(2)
                        .setEvent(
                            BuildEvent.newBuilder()
                                .setEventTime(toProtoTimestamp(expected))
                                .setBuildFinished(
                                    BuildFinished.newBuilder()
                                        .setStatus(
                                            BuildStatus.newBuilder()
                                                .setResult(Result.COMMAND_SUCCEEDED)))))
                .build());
  }

  @Test
  public void testStreamEvents() {
    clock.advanceMillis(100);
    Instant firstEventTimestamp = clock.now();
    ByteString payload = ByteString.fromHex("deadbeef");
    assertThat(
            BuildEventServiceProtoUtil.bazelEvent(COMMAND_CONTEXT, firstEventTimestamp, 1, payload))
        .isEqualTo(
            PublishBuildToolEventStreamRequest.newBuilder()
                .addAllNotificationKeywords(KEYWORDS)
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
                                .setEventTime(toProtoTimestamp(firstEventTimestamp))
                                .setBazelEvent(
                                    Any.newBuilder()
                                        .setTypeUrl(
                                            "type.googleapis.com/build_event_stream.BuildEvent")
                                        .setValue(payload)))
                        .build())
                .build());

    clock.advanceMillis(100);
    Instant secondEventTimestamp = clock.now();
    assertThat(
            BuildEventServiceProtoUtil.bazelEvent(
                COMMAND_CONTEXT, secondEventTimestamp, 2, payload))
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
                                .setEventTime(toProtoTimestamp(secondEventTimestamp))
                                .setBazelEvent(
                                    Any.newBuilder()
                                        .setTypeUrl(
                                            "type.googleapis.com/build_event_stream.BuildEvent")
                                        .setValue(payload)))
                        .build())
                .build());

    clock.advanceMillis(100);
    Instant thirdEventTimestamp = clock.now();
    assertThat(BuildEventServiceProtoUtil.streamFinished(COMMAND_CONTEXT, thirdEventTimestamp, 3))
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
                                .setEventTime(toProtoTimestamp(thirdEventTimestamp))
                                .setComponentStreamFinished(
                                    BuildComponentStreamFinished.newBuilder()
                                        .setType(FinishType.FINISHED)))
                        .build())
                .build());
  }

  @Test
  public void testStreamEventsWithCheckPrecedingLifecycleEventsEnabled() {
    ByteString payload = ByteString.fromHex("deadbeef");
    CommandContext commandContext =
        CommandContext.builder()
            .setBuildId(BUILD_REQUEST_ID)
            .setInvocationId(BUILD_INVOCATION_ID)
            .setAttemptNumber(ATTEMPT_NUMBER)
            .setKeywords(KEYWORDS)
            .setProjectId(PROJECT_ID)
            .setCheckPrecedingLifecycleEvents(true)
            .build();
    assertThat(
            BuildEventServiceProtoUtil.bazelEvent(
                    commandContext, Instant.ofEpochMilli(100), 1, payload)
                .getCheckPrecedingLifecycleEventsPresent())
        .isTrue();
    // check_preceding_lifecycle_events_present is always false for events with sequence_number > 1.
    assertThat(
            BuildEventServiceProtoUtil.bazelEvent(
                    commandContext, Instant.ofEpochMilli(100), 2, payload)
                .getCheckPrecedingLifecycleEventsPresent())
        .isFalse();
    assertThat(
            BuildEventServiceProtoUtil.bazelEvent(
                    commandContext, Instant.ofEpochMilli(100), 3, payload)
                .getCheckPrecedingLifecycleEventsPresent())
        .isFalse();
  }

  private static Timestamp toProtoTimestamp(Instant instant) {
    return Timestamp.newBuilder()
        .setSeconds(instant.getEpochSecond())
        .setNanos(instant.getNano())
        .build();
  }
}
