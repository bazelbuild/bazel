// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static java.util.Collections.emptyList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.v1.BuildEvent.EventCase;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.devtools.build.v1.StreamId;
import io.grpc.Status;
import java.util.Collection;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.junit.rules.ExternalResource;

public abstract class AbstractBuildEventRecorder extends ExternalResource {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * When processing a build event determines whether to return {@link #streamEventResponseStatus}.
   */
  private Predicate<PublishBuildToolEventStreamRequest> streamEventPredicate = (o) -> false;

  private Status streamEventResponseStatus;
  /**
   * When processing a lifecycle event determines whether to return {@link
   * #lifecycleEventResponseStatus}.
   */
  private Predicate<PublishLifecycleEventRequest> lifecycleEventPredicate = (o) -> false;

  private Status lifecycleEventResponseStatus;

  private Predicate<PublishBuildToolEventStreamRequest> sendResponsesOnRequestPredicate =
      (o) -> true;
  private ConcurrentLinkedQueue<PublishBuildToolEventStreamResponse> responseBuffer =
      new ConcurrentLinkedQueue<>();

  protected final ListMultimap<StreamId, PublishLifecycleEventRequest> lifecycleEvents =
      LinkedListMultimap.create();
  protected final ListMultimap<StreamId, PublishBuildToolEventStreamRequest> streamEvents =
      LinkedListMultimap.create();
  protected final ListMultimap<StreamId, PublishBuildToolEventStreamRequest>
      successfulStreamEvents = LinkedListMultimap.create();

  /** Tell the server to sends ACKs out of order or for the wrong events */
  protected volatile boolean sendOutOfOrderAcknowledgments;

  protected Status eventStreamError;

  /** Starts a server using the specified port. * */
  protected abstract void startRpcServer(int port);

  /** Starts a server using an arbitrary port * */
  final void startRpcServer() {
    int port = pickNewPort();
    logger.atInfo().log("Starting BES recorder server on port: %d", port);
    startRpcServer(port);
    logger.atInfo().log("Started BES recorder server on port: %d", port);
  }

  /** Stops a running server. * */
  protected abstract void stopRpcServer();

  /** Returns the port the port the server is running, -1 otherwise. * */
  protected abstract int getPort();

  /** Returns whether or not a {@code publishBuildToolEventStream} was observed on this server. */
  protected abstract boolean publishBuildToolEventStreamAccepted();

  synchronized ImmutableList<PublishLifecycleEventRequest> getLifecycleEvents(StreamId streamId) {
    return ImmutableList.copyOf(lifecycleEvents.get(streamId));
  }

  synchronized ImmutableList<PublishBuildToolEventStreamRequest> getStreamEvents(
      StreamId streamId) {
    return ImmutableList.copyOf(streamEvents.get(streamId));
  }

  synchronized ImmutableList<PublishBuildToolEventStreamRequest> getSuccessfulStreamEvents(
      StreamId streamId) {
    return ImmutableList.copyOf(successfulStreamEvents.get(streamId));
  }

  public void setStreamEventPredicateAndResponseStatus(
      Predicate<PublishBuildToolEventStreamRequest> predicate, Status responseStatus) {
    this.streamEventPredicate = predicate;
    this.streamEventResponseStatus = responseStatus;
  }

  public void setLifecycleEventPredicateAndResponseStatus(
      Predicate<PublishLifecycleEventRequest> predicate, Status responseStatus) {
    this.lifecycleEventPredicate = predicate;
    this.lifecycleEventResponseStatus = responseStatus;
  }

  public void setSendResponsesOnRequestPredicate(
      Predicate<PublishBuildToolEventStreamRequest> sendResponsesOnRequestPredicate) {
    this.sendResponsesOnRequestPredicate = sendResponsesOnRequestPredicate;
  }

  void sendOutOfOrderAcknowledgments() {
    sendOutOfOrderAcknowledgments = true;
  }

  synchronized Status eventStreamError() {
    return eventStreamError;
  }

  /** Picks a free port to use for a test using platform-specific logic. */
  protected abstract int pickNewPort();

  public Status computeLifecycleResponse(PublishLifecycleEventRequest request) {
    try {
      if (lifecycleEventPredicate.test(request)) {
        return lifecycleEventResponseStatus;
      } else {
        return statusFor(request);
      }
    } catch (Exception e) {
      return Status.INTERNAL.withDescription(e.getMessage());
    }
  }

  public Pair<Status, Collection<PublishBuildToolEventStreamResponse>> computeStreamResponse(
      PublishBuildToolEventStreamRequest request) {
    if (streamEventPredicate.test(request)) {
      return Pair.of(streamEventResponseStatus, emptyList());
    } else if (sendResponsesOnRequestPredicate.test(request)) {
      ImmutableList<PublishBuildToolEventStreamResponse> response =
          ImmutableList.<PublishBuildToolEventStreamResponse>builder()
              .addAll(responseBuffer)
              .add(responseFor(request))
              .build();
      responseBuffer = new ConcurrentLinkedQueue<>();
      return Pair.of(statusFor(request), response);
    } else {
      responseBuffer.add(responseFor(request));
      return Pair.of(statusFor(request), emptyList());
    }
  }

  private static PublishBuildToolEventStreamResponse responseFor(
      PublishBuildToolEventStreamRequest request) {
    return PublishBuildToolEventStreamResponse.newBuilder()
        .setStreamId(request.getOrderedBuildEvent().getStreamId())
        .setSequenceNumber(request.getOrderedBuildEvent().getSequenceNumber())
        .build();
  }

  private static Status statusFor(PublishLifecycleEventRequest request) {
    switch (request.getBuildEvent().getEvent().getEventCase()) {
      case INVOCATION_ATTEMPT_STARTED:
      case BUILD_ENQUEUED:
        if (request.getBuildEvent().getSequenceNumber() == 1) {
          return Status.OK;
        }
        break;
      case INVOCATION_ATTEMPT_FINISHED:
      case BUILD_FINISHED:
        if (request.getBuildEvent().getSequenceNumber() == 2) {
          return Status.OK;
        }
        break;
      default:
        break;
    }
    return Status.UNKNOWN;
  }

  @Nullable
  private static Status statusFor(PublishBuildToolEventStreamRequest request) {
    if (request.getOrderedBuildEvent().getEvent().getEventCase()
        == EventCase.COMPONENT_STREAM_FINISHED) {
      return Status.OK;
    }
    return null;
  }
}
