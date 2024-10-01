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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished.FinishType.FINISHED;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
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
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.Any;
import com.google.protobuf.Timestamp;
import java.util.Set;
import javax.annotation.Nullable;

/** Utility class with convenience methods for building a {@link BuildEventServiceTransport}. */
public final class BuildEventServiceProtoUtil {

  private final String buildRequestId;
  private final String buildInvocationId;
  private final String projectId;
  private final String commandName;
  private final Set<String> additionalKeywords;
  private final boolean checkPrecedingLifecycleEvents;
  private final int attemptNumber;

  private BuildEventServiceProtoUtil(
      String buildRequestId,
      String buildInvocationId,
      @Nullable String projectId,
      String commandName,
      Set<String> additionalKeywords,
      boolean checkPrecedingLifecycleEvents,
      int attemptNumber) {
    checkArgument(attemptNumber >= 1);

    this.buildRequestId = buildRequestId;
    this.buildInvocationId = buildInvocationId;
    this.projectId = projectId;
    this.commandName = commandName;
    this.additionalKeywords = ImmutableSet.copyOf(additionalKeywords);
    this.checkPrecedingLifecycleEvents = checkPrecedingLifecycleEvents;
    this.attemptNumber = attemptNumber;
  }

  public PublishLifecycleEventRequest buildEnqueued(Timestamp timestamp) {
    return lifecycleEvent(
            projectId,
            1,
            com.google.devtools.build.v1.BuildEvent.newBuilder()
                .setEventTime(timestamp)
                .setBuildEnqueued(BuildEnqueued.getDefaultInstance()))
        .build();
  }

  public PublishLifecycleEventRequest buildFinished(Timestamp timestamp, Result result) {
    return lifecycleEvent(
            projectId,
            2,
            com.google.devtools.build.v1.BuildEvent.newBuilder()
                .setEventTime(timestamp)
                .setBuildFinished(
                    BuildFinished.newBuilder()
                        .setStatus(BuildStatus.newBuilder().setResult(result))))
        .build();
  }

  public PublishLifecycleEventRequest invocationStarted(Timestamp timestamp) {
    return lifecycleEvent(
            projectId,
            1,
            com.google.devtools.build.v1.BuildEvent.newBuilder()
                .setEventTime(timestamp)
                .setInvocationAttemptStarted(
                    InvocationAttemptStarted.newBuilder().setAttemptNumber(attemptNumber)))
        .build();
  }

  public PublishLifecycleEventRequest invocationFinished(Timestamp timestamp, Result result) {
    return lifecycleEvent(
            projectId,
            2,
            com.google.devtools.build.v1.BuildEvent.newBuilder()
                .setEventTime(timestamp)
                .setInvocationAttemptFinished(
                    InvocationAttemptFinished.newBuilder()
                        .setInvocationStatus(BuildStatus.newBuilder().setResult(result))))
        .build();
  }

  /** Creates a PublishBuildToolEventStreamRequest from a packed bazel event. */
  public PublishBuildToolEventStreamRequest bazelEvent(
      long sequenceNumber, Timestamp timestamp, Any packedEvent) {
    return bazelEvent(sequenceNumber, timestamp, /* streamAttempt */ 1, packedEvent);
  }

  public PublishBuildToolEventStreamRequest bazelEvent(
      long sequenceNumber, Timestamp timestamp, int streamAttempt, Any packedEvent) {
    return publishBuildToolEventStreamRequest(
        projectId,
        sequenceNumber,
        timestamp,
        streamAttempt,
        com.google.devtools.build.v1.BuildEvent.newBuilder().setBazelEvent(packedEvent));
  }

  public PublishBuildToolEventStreamRequest streamFinished(
      long sequenceNumber, Timestamp timestamp) {
    return streamFinished(sequenceNumber, timestamp, /* streamAttempt */ 1);
  }

  public PublishBuildToolEventStreamRequest streamFinished(
      long sequenceNumber, Timestamp timestamp, int streamAttempt) {
    return publishBuildToolEventStreamRequest(
        projectId,
        sequenceNumber,
        timestamp,
        streamAttempt,
        BuildEvent.newBuilder()
            .setComponentStreamFinished(
                BuildComponentStreamFinished.newBuilder().setType(FINISHED)));
  }

  @VisibleForTesting
  public PublishBuildToolEventStreamRequest publishBuildToolEventStreamRequest(
      @Nullable String projectId,
      long sequenceNumber,
      Timestamp timestamp,
      int streamAttempt,
      BuildEvent.Builder besEvent) {
    PublishBuildToolEventStreamRequest.Builder builder =
        PublishBuildToolEventStreamRequest.newBuilder()
            .setRetryAttemptNumber(streamAttempt)
            .setOrderedBuildEvent(
                OrderedBuildEvent.newBuilder()
                    .setSequenceNumber(sequenceNumber)
                    .setEvent(besEvent.setEventTime(timestamp))
                    .setStreamId(streamId(besEvent.getEventCase())));
    if (sequenceNumber == 1) {
      builder.addAllNotificationKeywords(getKeywords());
      builder.setCheckPrecedingLifecycleEventsPresent(checkPrecedingLifecycleEvents);
    }
    if (projectId != null) {
      builder.setProjectId(projectId);
    }
    return builder.build();
  }

  @VisibleForTesting
  public PublishLifecycleEventRequest.Builder lifecycleEvent(@Nullable String projectId,
      int sequenceNumber, BuildEvent.Builder lifecycleEvent) {
    PublishLifecycleEventRequest.Builder builder = PublishLifecycleEventRequest.newBuilder()
        .setServiceLevel(PublishLifecycleEventRequest.ServiceLevel.INTERACTIVE)
        .setBuildEvent(
            OrderedBuildEvent.newBuilder()
                .setSequenceNumber(sequenceNumber)
                .setStreamId(streamId(lifecycleEvent.getEventCase()))
                .setEvent(lifecycleEvent));
    if (projectId != null) {
      builder.setProjectId(projectId);
    }
    switch (lifecycleEvent.getEventCase()) {
      case BUILD_ENQUEUED, INVOCATION_ATTEMPT_STARTED, BUILD_FINISHED ->
          builder.addAllNotificationKeywords(getKeywords());
      default -> {}
    }
    return builder;
  }

  @VisibleForTesting
  public StreamId streamId(EventCase eventCase) {
    StreamId.Builder streamId = StreamId.newBuilder().setBuildId(buildRequestId);
    switch (eventCase) {
      case BUILD_ENQUEUED, BUILD_FINISHED -> streamId.setComponent(BuildComponent.CONTROLLER);
      case INVOCATION_ATTEMPT_STARTED, INVOCATION_ATTEMPT_FINISHED -> {
        streamId.setInvocationId(buildInvocationId);
        streamId.setComponent(BuildComponent.CONTROLLER);
      }
      case BAZEL_EVENT, COMPONENT_STREAM_FINISHED -> {
        streamId.setInvocationId(buildInvocationId);
        streamId.setComponent(BuildComponent.TOOL);
      }
      default -> throw new IllegalArgumentException("Illegal EventCase " + eventCase);
    }
    return streamId.build();
  }

  /** Keywords used by BES subscribers to filter notifications */
  private ImmutableSet<String> getKeywords() {
    return ImmutableSet.<String>builder()
        .add("command_name=" + commandName)
        .add("protocol_name=BEP")
        .addAll(additionalKeywords)
        .build();
  }

  /** A builder for {@link BuildEventServiceProtoUtil}. */
  public static class Builder {
    private String invocationId;
    private String buildRequestId;
    private String commandName;
    private Set<String> keywords;
    @Nullable private String projectId;
    private boolean checkPrecedingLifecycleEvents;
    private int attemptNumber;

    @CanIgnoreReturnValue
    public Builder buildRequestId(String value) {
      this.buildRequestId = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder invocationId(String value) {
      this.invocationId = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder projectId(String value) {
      this.projectId = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder commandName(String value) {
      this.commandName = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder keywords(Set<String> value) {
      this.keywords = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder checkPrecedingLifecycleEvents(boolean value) {
      this.checkPrecedingLifecycleEvents = value;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder attemptNumber(int attemptNumber) {
      this.attemptNumber = attemptNumber;
      return this;
    }

    public BuildEventServiceProtoUtil build() {
      return new BuildEventServiceProtoUtil(
          checkNotNull(buildRequestId),
          checkNotNull(invocationId),
          projectId,
          checkNotNull(commandName),
          checkNotNull(keywords),
          checkPrecedingLifecycleEvents,
          attemptNumber);
    }
  }
}
