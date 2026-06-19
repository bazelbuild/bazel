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

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import com.google.bytestream.ByteStreamGrpc;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.time.Duration;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import javax.annotation.Nullable;

/** Cancels remote downloads after a configured period without response data. */
final class RemoteDownloadIdleTimeoutInterceptor implements ClientInterceptor {

  private final long timeoutNanos;
  private final ScheduledExecutorService scheduler;

  RemoteDownloadIdleTimeoutInterceptor(Duration timeout, ScheduledExecutorService scheduler) {
    this.timeoutNanos = timeout.toNanos();
    this.scheduler = scheduler;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    ClientCall<ReqT, RespT> call = next.newCall(method, callOptions);
    if (method != ByteStreamGrpc.getReadMethod()) {
      return call;
    }
    return new IdleTimeoutCall<>(call, method.getFullMethodName(), timeoutNanos, scheduler);
  }

  private static final class IdleTimeoutCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {

    private final String methodName;
    private final long timeoutNanos;
    private final ScheduledExecutorService scheduler;
    private final Object lock = new Object();

    @Nullable private ScheduledFuture<?> timeoutFuture;
    private boolean closed;
    private boolean timedOut;

    IdleTimeoutCall(
        ClientCall<ReqT, RespT> delegate,
        String methodName,
        long timeoutNanos,
        ScheduledExecutorService scheduler) {
      super(delegate);
      this.methodName = methodName;
      this.timeoutNanos = timeoutNanos;
      this.scheduler = scheduler;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {
            @Override
            public void onHeaders(Metadata headers) {
              resetTimeout();
              super.onHeaders(headers);
            }

            @Override
            public void onMessage(RespT message) {
              resetTimeout();
              super.onMessage(message);
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              closeTimeout();
              super.onClose(status, trailers);
            }
          },
          headers);
      resetTimeout();
    }

    @Override
    public void sendMessage(ReqT message) {
      resetTimeout();
      super.sendMessage(message);
    }

    @Override
    public void halfClose() {
      resetTimeout();
      super.halfClose();
    }

    @Override
    public void cancel(@Nullable String message, @Nullable Throwable cause) {
      closeTimeout();
      super.cancel(message, cause);
    }

    private void resetTimeout() {
      synchronized (lock) {
        if (closed || timedOut) {
          return;
        }
        if (timeoutFuture != null) {
          timeoutFuture.cancel(false);
        }
        timeoutFuture = scheduler.schedule(this::onTimeout, timeoutNanos, NANOSECONDS);
      }
    }

    private void closeTimeout() {
      synchronized (lock) {
        closed = true;
        if (timeoutFuture != null) {
          timeoutFuture.cancel(false);
          timeoutFuture = null;
        }
      }
    }

    private void onTimeout() {
      String message =
          "Remote download idle timeout exceeded after "
              + Duration.ofNanos(timeoutNanos)
              + " for "
              + methodName;
      synchronized (lock) {
        if (closed || timedOut) {
          return;
        }
        timedOut = true;
        timeoutFuture = null;
      }
      super.cancel(message, Status.DEADLINE_EXCEEDED.withDescription(message).asRuntimeException());
    }
  }
}
