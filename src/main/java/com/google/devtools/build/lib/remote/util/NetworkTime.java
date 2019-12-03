// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.common.base.Stopwatch;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.Context;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.time.Duration;

/** Reentrant wall clock stopwatch and grpc interceptor for network waits. */
@ThreadSafety.ThreadSafe
public class NetworkTime {
  public static final Context.Key<NetworkTime> CONTEXT_KEY =
      Context.key("remote-network-time");

  private final Stopwatch wallTime = Stopwatch.createUnstarted();
  private int outstanding = 0;
  private boolean enabled = true;

  synchronized private void start() {
    if (enabled) {
      if (!wallTime.isRunning()) {
        wallTime.start();
      }
      outstanding++;
    }
  }

  synchronized private void stop() {
    if (enabled) {
      if (--outstanding == 0) {
        wallTime.stop();
      }
    }
  }

  synchronized public void enable() {
    enabled = true;
  }

  synchronized public void disable() {
    enabled = false;
  }

  public Duration getDuration() {
    return wallTime.elapsed();
  }

  public static class Interceptor implements ClientInterceptor {
    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
      ClientCall<ReqT, RespT> call = next.newCall(method, callOptions);
      NetworkTime networkTime = CONTEXT_KEY.get();
      if (networkTime != null) {
        call = new NetworkTimeCall<>(call, networkTime);
      }
      return call;
    }
  }

  private static class NetworkTimeCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    private final NetworkTime networkTime;
    private boolean firstMessage = true;

    protected NetworkTimeCall(
        ClientCall<ReqT, RespT> delegate,
        NetworkTime networkTime) {
      super(delegate);
      this.networkTime = networkTime;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {
            @Override
            public void onMessage(RespT message) {
              super.onMessage(message);
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              networkTime.stop();
              super.onClose(status, trailers);
            }
          },
          headers);
    }

    @Override
    public void sendMessage(ReqT message) {
      if (firstMessage) {
        networkTime.start();
        firstMessage = false;
      }
      super.sendMessage(message);
    }
  }
}
