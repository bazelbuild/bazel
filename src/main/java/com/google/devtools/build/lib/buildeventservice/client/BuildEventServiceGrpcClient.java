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

import static com.google.common.base.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

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
import java.util.function.Function;
import javax.annotation.Nullable;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public abstract class BuildEventServiceGrpcClient implements BuildEventServiceClient {
  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.ofSeconds(15);

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;
  private volatile StreamObserver<PublishBuildToolEventStreamRequest> stream;

  public BuildEventServiceGrpcClient(Channel channel, @Nullable CallCredentials callCredentials) {
    this.besAsync = withCallCredentials(PublishBuildEventGrpc.newStub(channel), callCredentials);
    this.besBlocking =
        withCallCredentials(PublishBuildEventGrpc.newBlockingStub(channel), callCredentials);
  }

  private static <T extends AbstractStub<T>> T withCallCredentials(
      T stub, @Nullable CallCredentials callCredentials) {
    stub = callCredentials != null ? stub.withCallCredentials(callCredentials) : stub;
    return stub;
  }

  @Override
  public void publish(PublishLifecycleEventRequest lifecycleEvent)
      throws StatusException, InterruptedException {
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
  public ListenableFuture<Status> openStream(
      Function<PublishBuildToolEventStreamResponse, Void> ackCallback)
      throws StatusException, InterruptedException {
    Preconditions.checkState(
        stream == null, "Starting a new stream without closing the previous one");
    SettableFuture<Status> streamFinished = SettableFuture.create();
    stream = createStream(ackCallback, streamFinished);
    return streamFinished;
  }

  private StreamObserver<PublishBuildToolEventStreamRequest> createStream(
      final Function<PublishBuildToolEventStreamResponse, Void> ackCallback,
      final SettableFuture<Status> streamFinished)
      throws StatusException, InterruptedException {
    try {
      return besAsync.publishBuildToolEventStream(
          new StreamObserver<PublishBuildToolEventStreamResponse>() {
            @Override
            public void onNext(PublishBuildToolEventStreamResponse response) {
              ackCallback.apply(response);
            }

            @Override
            public void onError(Throwable t) {
              stream = null;
              streamFinished.set(Status.fromThrowable(t));
            }

            @Override
            public void onCompleted() {
              stream = null;
              streamFinished.set(Status.OK);
            }
          });
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      throw e.getStatus().asException();
    }
  }

  @Override
  public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent)
      throws StatusException, InterruptedException {
    StreamObserver<PublishBuildToolEventStreamRequest> stream0 = stream;
    checkState(stream0 != null, "Attempting to send over a closed stream");
    try {
      stream0.onNext(buildEvent);
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      throw e.getStatus().asException();
    }
  }

  @Override
  public void closeStream() {
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
  public boolean isStreamActive() {
    return stream != null;
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
  public abstract void shutdown() throws InterruptedException;
}
