// Copyright 2025 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.v1.BuildEvent.BuildComponentStreamFinished.FinishType;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventImplBase;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.devtools.build.v1.StreamId;
import com.google.protobuf.Empty;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Trivial implementation of {@link PublishBuildEventImplBase} that can insert sleeps at critical
 * junctures.
 */
public class DelayingPublishBuildEventService extends PublishBuildEventImplBase {
  @GuardedBy("this")
  private Duration delayBeforeClosingStream = Duration.ZERO;

  @GuardedBy("this")
  private Duration delayBeforeHalfClosingStream = Duration.ZERO;

  @GuardedBy("this")
  @Nullable
  private String errorMessage = null;

  @GuardedBy("this")
  @Nullable
  private Status errorCode = null;

  private final AtomicInteger requestsReceived = new AtomicInteger(0);
  private boolean errorEarlyInStream = false;

  /**
   * Synchronizing this method can lead to deadlocks -- it calls into {@link
   * io.grpc.inprocess.InProcessTransport} which takes a locks on itself. Opposite order of locks
   * happens for {@link #publishBuildToolEventStream} called while holding the lock on {@link
   * io.grpc.inprocess.InProcessTransport}.
   */
  @Override
  public void publishLifecycleEvent(
      PublishLifecycleEventRequest request, StreamObserver<Empty> responseObserver) {
    RequestMetadata metadata = TracingMetadataUtils.fromCurrentContext();
    assertThat(metadata.getToolInvocationId()).isNotEmpty();
    assertThat(metadata.getCorrelatedInvocationsId()).isNotEmpty();
    assertThat(metadata.getActionId()).isEqualTo("publish_lifecycle_event");

    responseObserver.onNext(Empty.getDefaultInstance());
    responseObserver.onCompleted();
  }

  @Override
  public synchronized StreamObserver<PublishBuildToolEventStreamRequest>
      publishBuildToolEventStream(
          StreamObserver<PublishBuildToolEventStreamResponse> responseObserver) {
    requestsReceived.incrementAndGet();
    RequestMetadata metadata = TracingMetadataUtils.fromCurrentContext();
    assertThat(metadata.getToolInvocationId()).isNotEmpty();
    assertThat(metadata.getCorrelatedInvocationsId()).isNotEmpty();
    assertThat(metadata.getActionId()).isEqualTo("publish_build_tool_event_stream");

    if (errorMessage != null) {
      return new ErroringPublishBuildStreamObserver(
          responseObserver, errorMessage, errorCode, errorEarlyInStream);
    }
    DelayingPublishBuildStreamObserver observer =
        new DelayingPublishBuildStreamObserver(
            responseObserver, delayBeforeClosingStream, delayBeforeHalfClosingStream);
    observer.startAckingThread();
    return observer;
  }

  synchronized void setErrorMessage(String errorMessage) {
    setErrorMessageAndCode(errorMessage, Status.DATA_LOSS);
  }

  synchronized void setErrorMessageAndCode(String errorMessage, Status code) {
    this.errorMessage = errorMessage;
    this.errorCode = code;
  }

  synchronized void setErrorEarlyInStream(boolean errorEarlyInStream) {
    this.errorEarlyInStream = errorEarlyInStream;
  }

  synchronized void setDelayBeforeClosingStream(Duration delay) {
    this.delayBeforeClosingStream = delay;
  }

  synchronized void setDelayBeforeHalfClosingStream(Duration delay) {
    this.delayBeforeHalfClosingStream = delay;
  }

  int getRequestsReceivedCount() {
    return requestsReceived.get();
  }

  /**
   * A {@link StreamObserver} that simulates a server that terminates the stream with an error,
   * either immediately or when the client closes its end of the stream.
   */
  private static final class ErroringPublishBuildStreamObserver
      implements StreamObserver<PublishBuildToolEventStreamRequest> {

    private final StreamObserver<PublishBuildToolEventStreamResponse> responseObserver;
    private final String errorMessage;
    private final Status errorCode;
    private final boolean errorEarlyInStream;

    ErroringPublishBuildStreamObserver(
        StreamObserver<PublishBuildToolEventStreamResponse> responseObserver,
        String errorMessage,
        Status errorCode,
        boolean errorEarlyInStream) {
      this.responseObserver = responseObserver;
      this.errorMessage = errorMessage;
      this.errorCode = errorCode;
      this.errorEarlyInStream = errorEarlyInStream;
    }

    @Override
    public void onNext(PublishBuildToolEventStreamRequest value) {
      if (errorEarlyInStream) {
        responseObserver.onError(
            new StatusRuntimeException(errorCode.withDescription(errorMessage)));
      }
      responseObserver.onNext(
          PublishBuildToolEventStreamResponse.newBuilder()
              .setStreamId(value.getOrderedBuildEventOrBuilder().getStreamId())
              .setSequenceNumber(value.getOrderedBuildEvent().getSequenceNumber())
              .build());
    }

    @Override
    public void onError(Throwable t) {}

    @Override
    public void onCompleted() {
      responseObserver.onError(new StatusRuntimeException(errorCode.withDescription(errorMessage)));
    }
  }

  /**
   * Trivial, in-memory implementation of a PublishBuildToolEventStream handler that can have
   * pre-configured sleeps triggered at critical junctures.
   */
  private static class DelayingPublishBuildStreamObserver
      implements StreamObserver<PublishBuildToolEventStreamRequest> {

    private final StreamObserver<PublishBuildToolEventStreamResponse> responseObserver;
    private final Duration delayBeforeClosingStream;
    private final Duration delayBeforeHalfClosingStream;

    @GuardedBy("this")
    private final SortedSet<Long> unackedSequenceNumbers = Sets.newTreeSet();

    private final BlockingQueue<Long> ackQueue = new ArrayBlockingQueue<>(10);

    @GuardedBy("this")
    private Thread ackingThread = null;

    @GuardedBy("this")
    private StreamId streamId = null;

    @GuardedBy("this")
    private boolean finished = false;

    private DelayingPublishBuildStreamObserver(
        StreamObserver<PublishBuildToolEventStreamResponse> responseObserver,
        Duration delayBeforeClosingStream,
        Duration delayBeforeHalfClosingStream) {
      this.responseObserver = responseObserver;
      this.delayBeforeClosingStream = delayBeforeClosingStream;
      this.delayBeforeHalfClosingStream = delayBeforeHalfClosingStream;
    }

    /** Creates the acking thread, safely callable after the constructor finishes. */
    synchronized void startAckingThread() {
      Preconditions.checkState(ackingThread == null, "startAckingThread() called twice");
      ackingThread = new Thread(new AckingThread());
      ackingThread.start();
    }

    @Override
    public void onNext(PublishBuildToolEventStreamRequest req) {
      List<Long> longsToPut = new ArrayList<>();
      synchronized (this) {
        if (!unackedSequenceNumbers.add(req.getOrderedBuildEvent().getSequenceNumber())) {
          return; // dupe, ignore
        }
        streamId = MoreObjects.firstNonNull(streamId, req.getOrderedBuildEvent().getStreamId());
        if (req.getOrderedBuildEvent().getEvent().getComponentStreamFinished().getType()
            == FinishType.FINISH_TYPE_UNSPECIFIED) {
          // We did not get the final event. Ack the *previous* event, if there is a previous event.
          if (unackedSequenceNumbers.size() > 1) {
            longsToPut.add(ackLowestSequenceNumber());
          }
        } else {
          Uninterruptibles.sleepUninterruptibly(delayBeforeHalfClosingStream);
          // final event. ack everything remaining.
          while (!unackedSequenceNumbers.isEmpty()) {
            longsToPut.add(ackLowestSequenceNumber());
          }
          if (finished) {
            longsToPut.add(SENTINEL_VALUE);
          }
        }
      }
      for (Long seqNum : longsToPut) {
        Uninterruptibles.putUninterruptibly(ackQueue, seqNum);
      }
    }

    @GuardedBy("this")
    private Long ackLowestSequenceNumber() {
      Long firstUnacked = unackedSequenceNumbers.first();
      unackedSequenceNumbers.remove(firstUnacked);
      return firstUnacked;
    }

    @Override
    public synchronized void onError(Throwable t) {
      finished = true;
      responseObserver.onError(t);
    }

    @Override
    public void onCompleted() {
      boolean putSentinel;
      synchronized (this) {
        finished = true;
        putSentinel = unackedSequenceNumbers.isEmpty();
      }
      if (putSentinel) {
        Uninterruptibles.putUninterruptibly(ackQueue, SENTINEL_VALUE);
      }
    }

    static final Long SENTINEL_VALUE = -1L;

    private class AckingThread implements Runnable {

      @Override
      public void run() {
        while (true) {
          Long firstUnacked = Uninterruptibles.takeUninterruptibly(ackQueue);
          synchronized (DelayingPublishBuildStreamObserver.this) {
            if (firstUnacked.equals(SENTINEL_VALUE)) {
              Uninterruptibles.sleepUninterruptibly(delayBeforeClosingStream);
              responseObserver.onCompleted();
              return;
            }
            responseObserver.onNext(
                PublishBuildToolEventStreamResponse.newBuilder()
                    .setStreamId(streamId)
                    .setSequenceNumber(firstUnacked)
                    .build());
          }
        }
      }
    }
  }
}
