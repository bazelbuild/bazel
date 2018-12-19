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

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.v1.PublishBuildEventGrpc;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventBlockingStub;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventStub;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import javax.annotation.Nullable;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public abstract class BuildEventServiceGrpcClient implements BuildEventServiceClient {
  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.ofSeconds(15);

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;
  private volatile StreamObserver<PublishBuildToolEventStreamRequest> stream;
  private volatile SettableFuture<Status> streamStatus;

  public BuildEventServiceGrpcClient(Channel channel, @Nullable CallCredentials callCredentials) {
    this(
        withCallCredentials(PublishBuildEventGrpc.newStub(channel), callCredentials),
        withCallCredentials(PublishBuildEventGrpc.newBlockingStub(channel), callCredentials));
  }

  @VisibleForTesting
  protected BuildEventServiceGrpcClient(
      PublishBuildEventStub besAsync, PublishBuildEventBlockingStub besBlocking) {
    this.besAsync = besAsync;
    this.besBlocking = besBlocking;
  }

  private static <T extends AbstractStub<T>> T withCallCredentials(
      T stub, @Nullable CallCredentials callCredentials) {
    stub = callCredentials != null ? stub.withCallCredentials(callCredentials) : stub;
    return stub;
  }

  @Override
  public void publish(PublishLifecycleEventRequest lifecycleEvent)
      throws StatusException, InterruptedException {
    throwIfInterrupted();
    try {
      besBlocking
          .withDeadlineAfter(RPC_TIMEOUT.toMillis(), MILLISECONDS)
          .publishLifecycleEvent(lifecycleEvent);
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      throw e.getStatus().asException();
    }
  }

  @Override
  public ListenableFuture<Status> openStream(AckCallback ackCallback) throws InterruptedException {
    Preconditions.checkState(
        stream == null, "Starting a new stream without closing the previous one");
    streamStatus = SettableFuture.create();
    ListenableFuture<Status> streamStatus0 = streamStatus;
    try {
      stream =
          besAsync.publishBuildToolEventStream(
              new StreamObserver<PublishBuildToolEventStreamResponse>() {
                @Override
                public void onNext(PublishBuildToolEventStreamResponse response) {
                  ackCallback.apply(response);
                }

                @Override
                public void onError(Throwable t) {
                  Status error = Status.fromThrowable(t);
                  if (error.getCode() == Status.CANCELLED.getCode()
                      && error.getCause() != null
                      && Status.fromThrowable(error.getCause()).getCode()
                          != Status.UNKNOWN.getCode()) {
                    // gRPC likes to wrap Status(Runtime)Exceptions in StatusRuntimeExceptions. If
                    // the status is cancelled and has a Status(Runtime)Exception as a cause it
                    // means the error was generated client side.
                    error = Status.fromThrowable(error.getCause());
                  }
                  stream = null;
                  streamStatus.set(error);
                  streamStatus = null;
                }

                @Override
                public void onCompleted() {
                  stream = null;
                  streamStatus.set(Status.OK);
                  streamStatus = null;
                }
              });
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      setStreamStatus(Status.fromThrowable(e));
    }
    return streamStatus0;
  }

  @Override
  public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent)
      throws InterruptedException {
    throwIfInterrupted();
    StreamObserver<PublishBuildToolEventStreamRequest> stream0 = stream;
    try {
      if (stream0 != null) {
        stream0.onNext(buildEvent);
      }
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      setStreamStatus(Status.fromThrowable(e));
    }
  }

  @Override
  public void halfCloseStream() {
    StreamObserver<PublishBuildToolEventStreamRequest> stream0 = stream;
    if (stream0 != null) {
      stream0.onCompleted();
    }
  }

  @Override
  public void abortStream(Status status) {
    StreamObserver<PublishBuildToolEventStreamRequest> stream0 = stream;
    if (stream0 != null) {
      stream0.onError(status.asException());
    }
  }

  @Override
  public String userReadableError(Throwable t) {
    if (t instanceof StatusException) {
      Throwable rootCause = Throwables.getRootCause(t);
      String message = ((StatusException) t).getStatus().getCode().name();
      message += ": " + rootCause.getMessage();
      return message;
    } else {
      return t.getMessage();
    }
  }

  @Override
  public abstract void shutdown();

  private void setStreamStatus(Status status) {
    SettableFuture<Status> streamStatus0 = streamStatus;
    if (streamStatus0 != null) {
      streamStatus0.set(status);
    }
  }

  private static void throwIfInterrupted() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
  }
}
