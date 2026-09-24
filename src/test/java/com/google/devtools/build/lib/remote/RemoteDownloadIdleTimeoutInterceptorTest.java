// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteDownloadIdleTimeoutInterceptor}. */
@RunWith(JUnit4.class)
public final class RemoteDownloadIdleTimeoutInterceptorTest {

  @Test
  public void byteStreamWrite_doesNotScheduleTimeout() {
    // ByteStream.Write is an upload RPC, so it must not inherit the download idle timeout even
    // though it belongs to the same gRPC service as ByteStream.Read.
    FakeClientCall<WriteRequest, WriteResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(Duration.ofSeconds(3), scheduler);

    ClientCall<WriteRequest, WriteResponse> call =
        interceptor.interceptCall(ByteStreamGrpc.getWriteMethod(), CallOptions.DEFAULT, channel);
    call.start(new ClientCall.Listener<>() {}, new Metadata());
    call.sendMessage(WriteRequest.getDefaultInstance());

    verify(scheduler, never()).schedule(any(Runnable.class), anyLong(), any(TimeUnit.class));
    assertThat(delegate.cancelled).isFalse();
  }

  @Test
  public void byteStreamRead_cancelsWhenIdleTimeoutFires() {
    // A ByteStream.Read with no response activity for the full timeout should be canceled so the
    // cache client can retry it rather than waiting forever.
    FakeClientCall<ReadRequest, ReadResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledTasks scheduledTasks = new ScheduledTasks();
    AtomicLong nanoTime = new AtomicLong();
    AtomicReference<Status> closedStatus = new AtomicReference<>();
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(
            Duration.ofSeconds(3), scheduledTasks.scheduler, nanoTime::get);

    ClientCall<ReadRequest, ReadResponse> call =
        interceptor.interceptCall(ByteStreamGrpc.getReadMethod(), CallOptions.DEFAULT, channel);
    call.start(
        new ClientCall.Listener<>() {
          @Override
          public void onClose(Status status, Metadata trailers) {
            closedStatus.set(status);
          }
        },
        new Metadata());
    // Advance the controlled clock to the deadline and run the captured timeout synchronously.
    nanoTime.set(Duration.ofSeconds(3).toNanos());
    scheduledTasks.tasks.get(0).run();

    assertThat(scheduledTasks.delaysNanos).containsExactly(Duration.ofSeconds(3).toNanos());
    assertThat(delegate.cancelled).isTrue();
    assertThat(delegate.cancelMessage)
        .isEqualTo(
            "Remote download idle timeout exceeded after PT3S for"
                + " google.bytestream.ByteStream/Read");
    assertThat(Status.fromThrowable(delegate.cancelCause).getCode())
        .isEqualTo(Status.Code.DEADLINE_EXCEEDED);
    // ClientCall.cancel normally closes the listener with CANCELLED. The interceptor translates
    // its own timeout-driven close so retry classification sees DEADLINE_EXCEEDED instead.
    assertThat(closedStatus.get().getCode()).isEqualTo(Status.Code.DEADLINE_EXCEEDED);
  }

  @Test
  public void byteStreamRead_defersTimeoutWithoutQueueingTasksForEachMessage() {
    // Activity should only move the monotonic deadline. Replacing the scheduled task for every
    // message would retain canceled tasks in the shared scheduler and grow its queue.
    FakeClientCall<ReadRequest, ReadResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledTasks scheduledTasks = new ScheduledTasks();
    AtomicLong nanoTime = new AtomicLong();
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(
            Duration.ofSeconds(3), scheduledTasks.scheduler, nanoTime::get);
    ImmutableList.Builder<ReadResponse> responses = ImmutableList.builder();

    ClientCall<ReadRequest, ReadResponse> call =
        interceptor.interceptCall(ByteStreamGrpc.getReadMethod(), CallOptions.DEFAULT, channel);
    call.start(
        new ClientCall.Listener<>() {
          @Override
          public void onMessage(ReadResponse message) {
            responses.add(message);
          }
        },
        new Metadata());
    ScheduledFuture<?> firstTimeout = scheduledTasks.futures.get(0);

    // The request at t=1 and response at t=2 move the deadline to t=5, but the original task for
    // t=3 remains the only task in the scheduler.
    nanoTime.set(Duration.ofSeconds(1).toNanos());
    call.sendMessage(ReadRequest.getDefaultInstance());

    nanoTime.set(Duration.ofSeconds(2).toNanos());
    ReadResponse response = ReadResponse.getDefaultInstance();
    delegate.listener.onMessage(response);

    assertThat(scheduledTasks.tasks).hasSize(1);
    // When the original task runs at t=3, it observes two seconds of remaining idle time and
    // schedules exactly one successor. Closing the call cancels that successor.
    nanoTime.set(Duration.ofSeconds(3).toNanos());
    scheduledTasks.tasks.get(0).run();
    ScheduledFuture<?> deferredTimeout = scheduledTasks.futures.get(1);
    delegate.listener.onClose(Status.OK, new Metadata());

    verify(firstTimeout, never()).cancel(false);
    verify(deferredTimeout).cancel(false);
    assertThat(scheduledTasks.tasks).hasSize(2);
    assertThat(scheduledTasks.delaysNanos)
        .containsExactly(Duration.ofSeconds(3).toNanos(), Duration.ofSeconds(2).toNanos())
        .inOrder();
    assertThat(responses.build()).containsExactly(response);
    assertThat(delegate.cancelled).isFalse();
  }

  @Test
  public void byteStreamRead_forwardsCallbacksWhenResetTimeoutFails() {
    // Timeout bookkeeping is secondary to the gRPC call. Make every attempt to schedule the
    // timeout fail, then verify the request and response still cross the interceptor boundary.
    FakeClientCall<ReadRequest, ReadResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
    RuntimeException resetFailure = new RuntimeException("reset failed");
    when(scheduler.schedule(any(Runnable.class), anyLong(), eq(NANOSECONDS)))
        .thenThrow(resetFailure);
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(Duration.ofSeconds(3), scheduler);
    ImmutableList.Builder<ReadResponse> responses = ImmutableList.builder();

    ClientCall<ReadRequest, ReadResponse> call =
        interceptor.interceptCall(ByteStreamGrpc.getReadMethod(), CallOptions.DEFAULT, channel);
    // start() installs the wrapped listener before the initial timeout reset fails.
    assertThrows(
        RuntimeException.class,
        () ->
            call.start(
                new ClientCall.Listener<>() {
                  @Override
                  public void onMessage(ReadResponse message) {
                    responses.add(message);
                  }
                },
                new Metadata()));

    ReadRequest request = ReadRequest.getDefaultInstance();
    // sendMessage() resets before forwarding, so the finally block is what preserves the request.
    assertThrows(RuntimeException.class, () -> call.sendMessage(request));

    ReadResponse response = ReadResponse.getDefaultInstance();
    // onMessage() follows the same rule: surface the reset failure without swallowing the data.
    assertThrows(RuntimeException.class, () -> delegate.listener.onMessage(response));

    assertThat(delegate.sentMessages).containsExactly(request);
    assertThat(responses.build()).containsExactly(response);
  }

  private static final class FakeChannel extends Channel {
    private final ClientCall<?, ?> call;

    private FakeChannel(ClientCall<?, ?> call) {
      this.call = call;
    }

    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> newCall(
        MethodDescriptor<ReqT, RespT> methodDescriptor, CallOptions callOptions) {
      @SuppressWarnings("unchecked")
      ClientCall<ReqT, RespT> typedCall = (ClientCall<ReqT, RespT>) call;
      return typedCall;
    }

    @Override
    public String authority() {
      return "test";
    }
  }

  private static final class FakeClientCall<ReqT, RespT> extends ClientCall<ReqT, RespT> {
    private Listener<RespT> listener;
    private final List<ReqT> sentMessages = new ArrayList<>();
    private boolean cancelled;
    private String cancelMessage;
    private Throwable cancelCause;

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      this.listener = responseListener;
    }

    @Override
    public void request(int numMessages) {}

    @Override
    public void cancel(String message, Throwable cause) {
      this.cancelled = true;
      this.cancelMessage = message;
      this.cancelCause = cause;
      listener.onClose(Status.CANCELLED.withDescription(message).withCause(cause), new Metadata());
    }

    @Override
    public void halfClose() {}

    @Override
    public void sendMessage(ReqT message) {
      sentMessages.add(message);
    }
  }

  private static final class ScheduledTasks {
    private final ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
    private final List<Runnable> tasks = new ArrayList<>();
    private final List<ScheduledFuture<?>> futures = new ArrayList<>();
    private final List<Long> delaysNanos = new ArrayList<>();

    ScheduledTasks() {
      when(scheduler.schedule(any(Runnable.class), anyLong(), eq(NANOSECONDS)))
          .thenAnswer(
              invocation -> {
                tasks.add(invocation.getArgument(0));
                delaysNanos.add(invocation.getArgument(1));
                ScheduledFuture<?> future = mock(ScheduledFuture.class);
                futures.add(future);
                return future;
              });
    }
  }
}
