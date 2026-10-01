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
import com.google.common.annotations.VisibleForTesting;
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
import java.util.function.LongSupplier;
import javax.annotation.Nullable;

/** Cancels remote downloads after a configured period without response data. */
final class RemoteDownloadIdleTimeoutInterceptor implements ClientInterceptor {

  private final long timeoutNanos;
  private final ScheduledExecutorService scheduler;
  private final LongSupplier nanoTime;

  RemoteDownloadIdleTimeoutInterceptor(Duration timeout, ScheduledExecutorService scheduler) {
    this(timeout, scheduler, System::nanoTime);
  }

  @VisibleForTesting
  RemoteDownloadIdleTimeoutInterceptor(
      Duration timeout, ScheduledExecutorService scheduler, LongSupplier nanoTime) {
    this.timeoutNanos = timeout.toNanos();
    this.scheduler = scheduler;
    this.nanoTime = nanoTime;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    ClientCall<ReqT, RespT> call = next.newCall(method, callOptions);
    if (method != ByteStreamGrpc.getReadMethod()) {
      return call;
    }
    return new IdleTimeoutCall<>(
        call, method.getFullMethodName(), timeoutNanos, scheduler, nanoTime);
  }

  private static final class IdleTimeoutCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {

    private final String methodName;
    private final long timeoutNanos;
    private final ScheduledExecutorService scheduler;
    private final LongSupplier nanoTime;
    private final Object lock = new Object();

    @Nullable private ScheduledFuture<?> timeoutFuture;
    private long deadlineNanos;
    private boolean closed;
    private boolean timedOut;

    IdleTimeoutCall(
        ClientCall<ReqT, RespT> delegate,
        String methodName,
        long timeoutNanos,
        ScheduledExecutorService scheduler,
        LongSupplier nanoTime) {
      super(delegate);
      this.methodName = methodName;
      this.timeoutNanos = timeoutNanos;
      this.scheduler = scheduler;
      this.nanoTime = nanoTime;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {
            @Override
            public void onHeaders(Metadata headers) {
              try {
                resetTimeout();
              } finally {
                super.onHeaders(headers);
              }
            }

            @Override
            public void onMessage(RespT message) {
              try {
                resetTimeout();
              } finally {
                super.onMessage(message);
              }
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              Status statusToForward = status;
              synchronized (lock) {
                if (timedOut && status.getCode() == Status.Code.CANCELLED) {
                  statusToForward =
                      Status.DEADLINE_EXCEEDED
                          .withDescription(timeoutMessage())
                          .withCause(status.getCause());
                }
              }
              try {
                closeTimeout();
              } finally {
                super.onClose(statusToForward, trailers);
              }
            }
          },
          headers);
      resetTimeout();
    }

    @Override
    public void sendMessage(ReqT message) {
      try {
        resetTimeout();
      } finally {
        super.sendMessage(message);
      }
    }

    @Override
    public void halfClose() {
      try {
        resetTimeout();
      } finally {
        super.halfClose();
      }
    }

    @Override
    public void cancel(@Nullable String message, @Nullable Throwable cause) {
      try {
        closeTimeout();
      } finally {
        super.cancel(message, cause);
      }
    }

    private void resetTimeout() {
      synchronized (lock) {
        if (closed || timedOut) {
          return;
        }
        deadlineNanos = nanoTime.getAsLong() + timeoutNanos;
        if (timeoutFuture == null) {
          timeoutFuture = scheduler.schedule(this::onTimeout, timeoutNanos, NANOSECONDS);
        }
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
      synchronized (lock) {
        if (closed || timedOut) {
          return;
        }
        long remainingNanos = deadlineNanos - nanoTime.getAsLong();
        if (remainingNanos > 0) {
          timeoutFuture = null;
          timeoutFuture = scheduler.schedule(this::onTimeout, remainingNanos, NANOSECONDS);
          return;
        }
        timedOut = true;
        timeoutFuture = null;
      }
      String message = timeoutMessage();
      super.cancel(message, Status.DEADLINE_EXCEEDED.withDescription(message).asRuntimeException());
    }

    private String timeoutMessage() {
      return "Remote download idle timeout exceeded after "
          + Duration.ofNanos(timeoutNanos)
          + " for "
          + methodName;
    }
  }
}
