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
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.v1.PublishBuildEventGrpc;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventBlockingStub;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventStub;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import io.grpc.CallCredentials;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import javax.annotation.Nullable;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public class BuildEventServiceGrpcClient implements BuildEventServiceClient {
  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.ofSeconds(15);

  private final ManagedChannel channel;

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;

  public BuildEventServiceGrpcClient(
      ManagedChannel channel, @Nullable CallCredentials callCredentials) {
    this(
        withCallCredentials(PublishBuildEventGrpc.newStub(channel), callCredentials),
        withCallCredentials(PublishBuildEventGrpc.newBlockingStub(channel), callCredentials),
        channel);
  }

  @VisibleForTesting
  protected BuildEventServiceGrpcClient(
      PublishBuildEventStub besAsync,
      PublishBuildEventBlockingStub besBlocking,
      ManagedChannel channel) {
    this.besAsync = besAsync;
    this.besBlocking = besBlocking;
    this.channel = channel;
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

  private static class BESGrpcStreamContext implements StreamContext {
    private final StreamObserver<PublishBuildToolEventStreamRequest> stream;
    private final SettableFuture<Status> streamStatus;

    public BESGrpcStreamContext(PublishBuildEventStub besAsync, AckCallback ackCallback) {
      this.streamStatus = SettableFuture.create();
      this.stream =
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
                  streamStatus.set(error);
                }

                @Override
                public void onCompleted() {
                  streamStatus.set(Status.OK);
                }
              });
    }

    @Override
    public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent)
        throws InterruptedException {
      throwIfInterrupted();
      try {
        stream.onNext(buildEvent);
      } catch (StatusRuntimeException e) {
        Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
        streamStatus.set(Status.fromThrowable(e));
      }
    }

    @Override
    public void halfCloseStream() {
      stream.onCompleted();
    }

    @Override
    public void abortStream(Status status) {
      stream.onError(status.asException());
    }

    @Override
    public ListenableFuture<Status> getStatus() {
      return streamStatus;
    }
  }

  @Override
  public StreamContext openStream(AckCallback ackCallback) throws InterruptedException {
    try {
      return new BESGrpcStreamContext(besAsync, ackCallback);
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      ListenableFuture<Status> status = Futures.immediateFuture(Status.fromThrowable(e));
      return new StreamContext() {
        @Override
        public ListenableFuture<Status> getStatus() {
          return status;
        }

        @Override
        public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent) {}

        @Override
        public void halfCloseStream() {}

        @Override
        public void abortStream(Status status) {}
      };
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
  public void shutdown() {
    channel.shutdown();
  }

  private static void throwIfInterrupted() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
  }
}
