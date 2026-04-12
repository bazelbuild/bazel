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

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.v1.PublishBuildEventGrpc;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventBlockingStub;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventStub;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import javax.annotation.Nullable;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public class BuildEventServiceGrpcClient implements BuildEventServiceClient {
  private static final ImmutableSet<Status.Code> NON_RETRYABLE_STATUS_CODES =
      ImmutableSet.of(Status.Code.INVALID_ARGUMENT, Status.Code.PERMISSION_DENIED);

  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.ofSeconds(15);

  private final ManagedChannel channel;

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;

  public BuildEventServiceGrpcClient(
      ManagedChannel channel,
      @Nullable CallCredentials callCredentials,
      ClientInterceptor interceptor) {
    this.besAsync =
        configureStub(PublishBuildEventGrpc.newStub(channel), callCredentials, interceptor);
    this.besBlocking =
        configureStub(PublishBuildEventGrpc.newBlockingStub(channel), callCredentials, interceptor);
    this.channel = channel;
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

  private static <T extends AbstractStub<T>> T configureStub(
      T stub, @Nullable CallCredentials callCredentials, @Nullable ClientInterceptor interceptor) {
    stub = callCredentials != null ? stub.withCallCredentials(callCredentials) : stub;
    stub = interceptor != null ? stub.withInterceptors(interceptor) : stub;
    return stub;
  }

  @Override
  public void publish(CommandContext commandContext, LifecycleEvent lifecycleEvent)
      throws StreamException, InterruptedException {
    PublishLifecycleEventRequest request =
        BuildEventServiceProtoUtil.publishLifecycleEventRequest(commandContext, lifecycleEvent);
    throwIfInterrupted();
    try {
      besBlocking
          .withDeadlineAfter(RPC_TIMEOUT.toMillis(), MILLISECONDS)
          .withInterceptors(
              TracingMetadataUtils.attachMetadataInterceptor(
                  TracingMetadataUtils.buildMetadata(
                      commandContext.buildId(),
                      commandContext.invocationId(),
                      "publish_lifecycle_event",
                      /* actionMetadata= */ null)))
          .publishLifecycleEvent(request);
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      Status status = Status.fromThrowable(e);
      throw new StreamException(new GrpcStreamStatus(status), e);
    }
  }

  private static class BESGrpcStreamContext implements StreamContext {
    private final StreamObserver<PublishBuildToolEventStreamRequest> stream;
    private final SettableFuture<StreamStatus> streamStatus;
    private final CommandContext commandContext;

    public BESGrpcStreamContext(
        PublishBuildEventStub besAsync, CommandContext commandContext, AckCallback ackCallback) {
      this.commandContext = commandContext;
      this.streamStatus = SettableFuture.create();
      this.stream =
          besAsync
              .withInterceptors(
                  TracingMetadataUtils.attachMetadataInterceptor(
                      TracingMetadataUtils.buildMetadata(
                          commandContext.buildId(),
                          commandContext.invocationId(),
                          "publish_build_tool_event_stream",
                          /* actionMetadata= */ null)))
              .publishBuildToolEventStream(
                  new StreamObserver<PublishBuildToolEventStreamResponse>() {
                    @Override
                    public void onNext(PublishBuildToolEventStreamResponse response) {
                      ackCallback.apply(response.getSequenceNumber());
                    }

                    @Override
                    public void onError(Throwable t) {
                      Status status = Status.fromThrowable(t);
                      if (status.getCode() == Status.CANCELLED.getCode()
                          && status.getCause() != null
                          && Status.fromThrowable(status.getCause()).getCode()
                              != Status.UNKNOWN.getCode()) {
                        // gRPC likes to wrap Status(Runtime)Exceptions in StatusRuntimeExceptions.
                        // If the status is cancelled and has a Status(Runtime)Exception as a cause,
                        // it means the error was generated client side.
                        status = Status.fromThrowable(status.getCause());
                      }
                      streamStatus.set(new GrpcStreamStatus(status));
                    }

                    @Override
                    public void onCompleted() {
                      streamStatus.set(GrpcStreamStatus.OK);
                    }
                  });
    }

    @Override
    public void sendOverStream(StreamEvent streamEvent) throws InterruptedException {
      PublishBuildToolEventStreamRequest request =
          BuildEventServiceProtoUtil.publishBuildToolEventStreamRequest(
              commandContext, streamEvent);
      throwIfInterrupted();
      try {
        stream.onNext(request);
      } catch (StatusRuntimeException e) {
        Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
        streamStatus.set(new GrpcStreamStatus(Status.fromThrowable(e)));
      }
    }

    @Override
    public void halfCloseStream() {
      stream.onCompleted();
    }

    @Override
    public void abortStream(AbortReason reason, @Nullable String description) {
      Status status =
          switch (reason) {
            case CANCELLED -> Status.CANCELLED;
            case FAILED_PRECONDITION -> Status.FAILED_PRECONDITION;
          };
      if (description != null) {
        status = status.withDescription(description);
      }
      stream.onError(status.asException());
    }

    @Override
    public ListenableFuture<StreamStatus> getStatus() {
      return streamStatus;
    }
  }

  @Override
  public StreamContext openStream(CommandContext commandContext, AckCallback ackCallback)
      throws InterruptedException {
    try {
      return new BESGrpcStreamContext(besAsync, commandContext, ackCallback);
    } catch (StatusRuntimeException e) {
      Throwables.throwIfInstanceOf(Throwables.getRootCause(e), InterruptedException.class);
      ListenableFuture<StreamStatus> status =
          immediateFuture(new GrpcStreamStatus(Status.fromThrowable(e)));
      return new StreamContext() {
        @Override
        public ListenableFuture<StreamStatus> getStatus() {
          return status;
        }

        @Override
        public void sendOverStream(StreamEvent streamEvent) {}

        @Override
        public void halfCloseStream() {}

        @Override
        public void abortStream(AbortReason reason, @Nullable String description) {}
      };
    }
  }

  private static final class GrpcStreamStatus implements StreamStatus {
    private static final GrpcStreamStatus OK = new GrpcStreamStatus(Status.OK);

    private final Status status;

    GrpcStreamStatus(Status status) {
      this.status = status;
    }

    @Override
    public boolean isOk() {
      return status.isOk();
    }

    @Override
    public boolean isRetriable() {
      return !status.isOk()
          && !NON_RETRYABLE_STATUS_CODES.contains(status.getCode())
          && status.getCode() != Status.Code.FAILED_PRECONDITION;
    }

    @Override
    public boolean isFailedPrecondition() {
      return status.getCode() == Status.Code.FAILED_PRECONDITION;
    }

    @Override
    public String getErrorMessage() {
      StringBuilder sb = new StringBuilder();
      sb.append(status.getCode().name());
      if (!Strings.isNullOrEmpty(status.getDescription())) {
        sb.append(": ").append(status.getDescription());
      }
      return sb.toString();
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
