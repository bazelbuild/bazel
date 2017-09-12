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

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.util.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.base.Function;
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
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public class BuildEventServiceGrpcClient implements BuildEventServiceClient {

  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.ofSeconds(15);

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;
  private final ManagedChannel channel;
  private final AtomicReference<StreamObserver<PublishBuildToolEventStreamRequest>> streamReference;

  public BuildEventServiceGrpcClient(
      ManagedChannel channel,
      @Nullable CallCredentials callCredentials) {
    this.channel = channel;
    this.besAsync = withCallCredentials(
        PublishBuildEventGrpc.newStub(channel), callCredentials);
    this.besBlocking = withCallCredentials(
        PublishBuildEventGrpc.newBlockingStub(channel), callCredentials);
    this.streamReference = new AtomicReference<>(null);
  }

  private static <T extends AbstractStub<T>> T withCallCredentials(
      T stub, @Nullable CallCredentials callCredentials) {
    stub = callCredentials != null ? stub.withCallCredentials(callCredentials) : stub;
    return stub;
  }

  @Override
  public Status publish(PublishLifecycleEventRequest lifecycleEvent) throws Exception {
    try {
      besBlocking
          .withDeadlineAfter(RPC_TIMEOUT.toMillis(), MILLISECONDS)
          .publishLifecycleEvent(lifecycleEvent);
    } catch (StatusRuntimeException e) {
      Throwable rootCause = Throwables.getRootCause(e);
      Throwables.throwIfInstanceOf(rootCause, InterruptedException.class);
      throw e;
    }
    return Status.OK;
  }

  @Override
  public ListenableFuture<Status> openStream(
      Function<PublishBuildToolEventStreamResponse, Void> ack)
      throws Exception {
    SettableFuture<Status> streamFinished = SettableFuture.create();
    checkState(
        streamReference.compareAndSet(null, createStream(ack, streamFinished)),
        "Starting a new stream without closing the previous one");
    return streamFinished;
  }

  private StreamObserver<PublishBuildToolEventStreamRequest> createStream(
      final Function<PublishBuildToolEventStreamResponse, Void> ack,
      final SettableFuture<Status> streamFinished) throws InterruptedException {
    try {
      return besAsync.publishBuildToolEventStream(
          new StreamObserver<PublishBuildToolEventStreamResponse>() {
            @Override
            public void onNext(PublishBuildToolEventStreamResponse response) {
              ack.apply(response);
            }

            @Override
            public void onError(Throwable t) {
              streamReference.set(null);
              streamFinished.setException(t);
            }

            @Override
            public void onCompleted() {
              streamReference.set(null);
              streamFinished.set(Status.OK);
            }
          });
    } catch (StatusRuntimeException e) {
      Throwable rootCause = Throwables.getRootCause(e);
      Throwables.throwIfInstanceOf(rootCause, InterruptedException.class);
      throw e;
    }
  }

  @Override
  public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent) throws Exception {
    try {
      checkNotNull(streamReference.get(), "Attempting to send over a closed or unopened stream")
          .onNext(buildEvent);
    } catch (StatusRuntimeException e) {
      Throwable rootCause = Throwables.getRootCause(e);
      Throwables.throwIfInstanceOf(rootCause, InterruptedException.class);
      throw e;
    }
  }

  @Override
  public void closeStream() {
    StreamObserver<PublishBuildToolEventStreamRequest> stream;
    if ((stream = streamReference.getAndSet(null)) != null) {
      stream.onCompleted();
    }
  }

  @Override
  public void abortStream(Status status) {
    StreamObserver<PublishBuildToolEventStreamRequest> stream;
    if ((stream = streamReference.getAndSet(null)) != null) {
      stream.onError(status.asException());
    }
  }

  @Override
  public boolean isStreamActive() {
    return streamReference.get() != null;
  }

  @Override
  public void shutdown() throws InterruptedException {
    this.channel.shutdown();
  }

  @Override
  public String userReadableError(Throwable t) {
    if (t instanceof StatusRuntimeException) {
      Throwable rootCause = Throwables.getRootCause(t);
      String message = ((StatusRuntimeException) t).getStatus().getCode().name();
      message += ": " + rootCause.getMessage();
      return message;
    } else {
      return t.getMessage();
    }
  }
}
