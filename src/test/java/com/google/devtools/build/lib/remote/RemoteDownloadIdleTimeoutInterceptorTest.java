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
import static java.nio.charset.StandardCharsets.UTF_8;
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
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteDownloadIdleTimeoutInterceptor}. */
@RunWith(JUnit4.class)
public final class RemoteDownloadIdleTimeoutInterceptorTest {

  @Test
  public void unaryCall_doesNotScheduleTimeout() {
    FakeClientCall<String, String> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledExecutorService scheduler = mock(ScheduledExecutorService.class);
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(Duration.ofSeconds(3), scheduler);

    ClientCall<String, String> call =
        interceptor.interceptCall(
            method(MethodDescriptor.MethodType.UNARY), CallOptions.DEFAULT, channel);
    call.start(new ClientCall.Listener<>() {}, new Metadata());
    call.sendMessage("request");

    verify(scheduler, never()).schedule(any(Runnable.class), anyLong(), any(TimeUnit.class));
    assertThat(delegate.cancelled).isFalse();
  }

  @Test
  public void byteStreamWrite_doesNotScheduleTimeout() {
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
    FakeClientCall<ReadRequest, ReadResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledTasks scheduledTasks = new ScheduledTasks();
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(Duration.ofSeconds(3), scheduledTasks.scheduler);

    ClientCall<ReadRequest, ReadResponse> call =
        interceptor.interceptCall(ByteStreamGrpc.getReadMethod(), CallOptions.DEFAULT, channel);
    call.start(new ClientCall.Listener<>() {}, new Metadata());
    scheduledTasks.tasks.get(0).run();

    assertThat(delegate.cancelled).isTrue();
    assertThat(delegate.cancelMessage)
        .isEqualTo(
            "Remote download idle timeout exceeded after PT3S for"
                + " google.bytestream.ByteStream/Read");
    assertThat(Status.fromThrowable(delegate.cancelCause).getCode())
        .isEqualTo(Status.Code.DEADLINE_EXCEEDED);
  }

  @Test
  public void byteStreamRead_resetsTimeoutOnRequestAndResponseMessages() {
    FakeClientCall<ReadRequest, ReadResponse> delegate = new FakeClientCall<>();
    FakeChannel channel = new FakeChannel(delegate);
    ScheduledTasks scheduledTasks = new ScheduledTasks();
    RemoteDownloadIdleTimeoutInterceptor interceptor =
        new RemoteDownloadIdleTimeoutInterceptor(Duration.ofSeconds(3), scheduledTasks.scheduler);
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

    call.sendMessage(ReadRequest.getDefaultInstance());
    ScheduledFuture<?> secondTimeout = scheduledTasks.futures.get(1);

    ReadResponse response = ReadResponse.getDefaultInstance();
    delegate.listener.onMessage(response);
    ScheduledFuture<?> thirdTimeout = scheduledTasks.futures.get(2);
    delegate.listener.onClose(Status.OK, new Metadata());

    verify(firstTimeout).cancel(false);
    verify(secondTimeout).cancel(false);
    verify(thirdTimeout).cancel(false);
    assertThat(responses.build()).containsExactly(response);
    assertThat(delegate.cancelled).isFalse();
  }

  @Test
  public void byteStreamRead_forwardsCallbacksWhenResetTimeoutFails() {
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
    assertThrows(RuntimeException.class, () -> call.sendMessage(request));

    ReadResponse response = ReadResponse.getDefaultInstance();
    assertThrows(RuntimeException.class, () -> delegate.listener.onMessage(response));

    assertThat(delegate.sentMessages).containsExactly(request);
    assertThat(responses.build()).containsExactly(response);
  }

  private static MethodDescriptor<String, String> method(MethodDescriptor.MethodType type) {
    return MethodDescriptor.<String, String>newBuilder()
        .setType(type)
        .setFullMethodName("service/method")
        .setRequestMarshaller(new StringMarshaller())
        .setResponseMarshaller(new StringMarshaller())
        .build();
  }

  private static final class StringMarshaller implements MethodDescriptor.Marshaller<String> {
    @Override
    public InputStream stream(String value) {
      return new ByteArrayInputStream(value.getBytes(UTF_8));
    }

    @Override
    public String parse(InputStream stream) {
      return "";
    }
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

    ScheduledTasks() {
      when(scheduler.schedule(any(Runnable.class), anyLong(), eq(NANOSECONDS)))
          .thenAnswer(
              invocation -> {
                assertThat(invocation.getArgument(1, Long.class))
                    .isEqualTo(Duration.ofSeconds(3).toNanos());
                tasks.add(invocation.getArgument(0));
                ScheduledFuture<?> future = mock(ScheduledFuture.class);
                futures.add(future);
                return future;
              });
    }
  }
}
