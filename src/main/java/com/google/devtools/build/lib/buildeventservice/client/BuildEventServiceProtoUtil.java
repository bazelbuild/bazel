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

package com.google.devtools.build.lib.buildeventservice.client;

import static com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished.FinishType.FINISHED;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.CommandContext;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.InvocationStatus;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.LifecycleEvent;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient.StreamEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.v1.BuildEvent;
import com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished;
import com.google.devtools.build.v1.BuildEvent.BuildEnqueued;
import com.google.devtools.build.v1.BuildEvent.BuildFinished;
import com.google.devtools.build.v1.BuildEvent.EventCase;
import com.google.devtools.build.v1.BuildEvent.InvocationAttemptFinished;
import com.google.devtools.build.v1.BuildEvent.InvocationAttemptStarted;
import com.google.devtools.build.v1.BuildStatus;
import com.google.devtools.build.v1.BuildStatus.Result;
import com.google.devtools.build.v1.OrderedBuildEvent;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.devtools.build.v1.StreamId;
import com.google.devtools.build.v1.StreamId.BuildComponent;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import java.time.Instant;

/** Utility methods to create BES proto messages. */
public final class BuildEventServiceProtoUtil {

  private BuildEventServiceProtoUtil() {}

  private static final String TYPE_URL =
      "type.googleapis.com/" + BuildEventStreamProtos.BuildEvent.getDescriptor().getFullName();

  /** Creates a {@link PublishLifecycleEventRequest} from a {@link LifecycleEvent}. */
  public static PublishLifecycleEventRequest publishLifecycleEventRequest(
      LifecycleEvent lifecycleEvent) {
    return switch (lifecycleEvent) {
      case LifecycleEvent.BuildEnqueued(CommandContext commandContext, Instant eventTime) ->
          buildEnqueued(commandContext, eventTime);
      case LifecycleEvent.InvocationStarted(CommandContext commandContext, Instant eventTime) ->
          invocationStarted(commandContext, eventTime);
      case LifecycleEvent.InvocationFinished(
              CommandContext commandContext,
              Instant eventTime,
              InvocationStatus status) ->
          invocationFinished(commandContext, eventTime, status);
      case LifecycleEvent.BuildFinished(
              CommandContext commandContext,
              Instant eventTime,
              InvocationStatus status) ->
          buildFinished(commandContext, eventTime, status);
    };
  }

  public static PublishLifecycleEventRequest buildEnqueued(
      CommandContext commandContext, Instant instant) {
    return lifecycleRequest(
            commandContext,
            1,
            BuildEvent.newBuilder()
                .setEventTime(toProtoTimestamp(instant))
                .setBuildEnqueued(BuildEnqueued.getDefaultInstance()))
        .build();
  }

  public static PublishLifecycleEventRequest buildFinished(
      CommandContext commandContext, Instant eventTime, InvocationStatus status) {
    return lifecycleRequest(
            commandContext,
            2,
            BuildEvent.newBuilder()
                .setEventTime(toProtoTimestamp(eventTime))
                .setBuildFinished(BuildFinished.newBuilder().setStatus(buildStatus(status))))
        .build();
  }

  public static PublishLifecycleEventRequest invocationStarted(
      CommandContext commandContext, Instant instant) {
    return lifecycleRequest(
            commandContext,
            1,
            BuildEvent.newBuilder()
                .setEventTime(toProtoTimestamp(instant))
                .setInvocationAttemptStarted(
                    InvocationAttemptStarted.newBuilder()
                        .setAttemptNumber(commandContext.attemptNumber())))
        .build();
  }

  public static PublishLifecycleEventRequest invocationFinished(
      CommandContext commandContext, Instant eventTime, InvocationStatus status) {
    return lifecycleRequest(
            commandContext,
            2,
            BuildEvent.newBuilder()
                .setEventTime(toProtoTimestamp(eventTime))
                .setInvocationAttemptFinished(
                    InvocationAttemptFinished.newBuilder()
                        .setInvocationStatus(buildStatus(status))))
        .build();
  }

  private static BuildStatus buildStatus(InvocationStatus status) {
    return switch (status) {
      case UNKNOWN -> BuildStatus.newBuilder().setResult(Result.UNKNOWN_STATUS).build();
      case SUCCEEDED -> BuildStatus.newBuilder().setResult(Result.COMMAND_SUCCEEDED).build();
      case FAILED -> BuildStatus.newBuilder().setResult(Result.COMMAND_FAILED).build();
    };
  }

  /** Creates a {@link PublishBuildToolEventStreamRequest} from a {@link StreamEvent}. */
  public static PublishBuildToolEventStreamRequest publishBuildToolEventStreamRequest(
      StreamEvent streamEvent) {
    return switch (streamEvent) {
      case StreamEvent.BazelEvent(
              CommandContext commandContext,
              Instant eventTime,
              long sequenceNumber,
              ByteString payload) ->
          bazelEvent(commandContext, eventTime, sequenceNumber, payload);
      case StreamEvent.StreamFinished(
              CommandContext commandContext,
              Instant eventTime,
              long sequenceNumber) ->
          streamFinished(commandContext, eventTime, sequenceNumber);
    };
  }

  public static PublishBuildToolEventStreamRequest bazelEvent(
      CommandContext commandContext, Instant eventTime, long sequenceNumber, ByteString payload) {
    // Any.pack() would require us to parse the payload into a Message, which is wasteful.
    // Implement it manually instead.
    Any packed = Any.newBuilder().setTypeUrl(TYPE_URL).setValue(payload).build();
    return streamRequest(
        commandContext,
        sequenceNumber,
        toProtoTimestamp(eventTime),
        BuildEvent.newBuilder().setBazelEvent(packed));
  }

  public static PublishBuildToolEventStreamRequest streamFinished(
      CommandContext commandContext, Instant eventTime, long sequenceNumber) {
    return streamRequest(
        commandContext,
        sequenceNumber,
        toProtoTimestamp(eventTime),
        BuildEvent.newBuilder()
            .setComponentStreamFinished(
                BuildComponentStreamFinished.newBuilder().setType(FINISHED)));
  }

  @VisibleForTesting
  public static PublishBuildToolEventStreamRequest streamRequest(
      CommandContext commandContext,
      long sequenceNumber,
      Timestamp timestamp,
      BuildEvent.Builder besEvent) {
    PublishBuildToolEventStreamRequest.Builder builder =
        PublishBuildToolEventStreamRequest.newBuilder()
            .setOrderedBuildEvent(
                OrderedBuildEvent.newBuilder()
                    .setSequenceNumber(sequenceNumber)
                    .setEvent(besEvent.setEventTime(timestamp))
                    .setStreamId(streamId(commandContext, besEvent.getEventCase())));
    if (sequenceNumber == 1) {
      builder
          .addAllNotificationKeywords(commandContext.keywords())
          .setCheckPrecedingLifecycleEventsPresent(commandContext.checkPrecedingLifecycleEvents());
    }
    if (commandContext.projectId() != null) {
      builder.setProjectId(commandContext.projectId());
    }
    return builder.build();
  }

  @VisibleForTesting
  public static PublishLifecycleEventRequest.Builder lifecycleRequest(
      CommandContext commandContext, int sequenceNumber, BuildEvent.Builder lifecycleEvent) {
    PublishLifecycleEventRequest.Builder builder =
        PublishLifecycleEventRequest.newBuilder()
            .setServiceLevel(PublishLifecycleEventRequest.ServiceLevel.INTERACTIVE)
            .setBuildEvent(
                OrderedBuildEvent.newBuilder()
                    .setSequenceNumber(sequenceNumber)
                    .setStreamId(streamId(commandContext, lifecycleEvent.getEventCase()))
                    .setEvent(lifecycleEvent));
    if (commandContext.projectId() != null) {
      builder.setProjectId(commandContext.projectId());
    }
    switch (lifecycleEvent.getEventCase()) {
      case BUILD_ENQUEUED, INVOCATION_ATTEMPT_STARTED, BUILD_FINISHED ->
          builder.addAllNotificationKeywords(commandContext.keywords());
      default -> {}
    }
    return builder;
  }

  @VisibleForTesting
  public static StreamId streamId(CommandContext commandContext, EventCase eventCase) {
    StreamId.Builder streamId = StreamId.newBuilder().setBuildId(commandContext.buildId());
    switch (eventCase) {
      case BUILD_ENQUEUED, BUILD_FINISHED -> streamId.setComponent(BuildComponent.CONTROLLER);
      case INVOCATION_ATTEMPT_STARTED, INVOCATION_ATTEMPT_FINISHED -> {
        streamId
            .setInvocationId(commandContext.invocationId())
            .setComponent(BuildComponent.CONTROLLER);
      }
      case BAZEL_EVENT, COMPONENT_STREAM_FINISHED -> {
        streamId.setInvocationId(commandContext.invocationId()).setComponent(BuildComponent.TOOL);
      }
      default -> throw new IllegalArgumentException("Illegal EventCase " + eventCase);
    }
    return streamId.build();
  }

  private static Timestamp toProtoTimestamp(Instant instant) {
    return Timestamp.newBuilder()
        .setSeconds(instant.getEpochSecond())
        .setNanos(instant.getNano())
        .build();
  }
}
