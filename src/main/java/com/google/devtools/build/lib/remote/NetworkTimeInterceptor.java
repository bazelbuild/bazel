// Copyright 2020 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.ExecutionGrpc;
import com.google.devtools.build.lib.remote.common.NetworkTime;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.util.function.Supplier;

/** The ClientInterceptor used to track network time. */
public class NetworkTimeInterceptor implements ClientInterceptor {

  private final Supplier<NetworkTime> networkTimeSupplier;

  public NetworkTimeInterceptor(Supplier<NetworkTime> networkTimeSupplier) {
    this.networkTimeSupplier = networkTimeSupplier;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    ClientCall<ReqT, RespT> call = next.newCall(method, callOptions);
    // prevent accounting for execution wait time
    if (method != ExecutionGrpc.getExecuteMethod()
        && method != ExecutionGrpc.getWaitExecutionMethod()) {
      NetworkTime networkTime = networkTimeSupplier.get();
      if (networkTime != null) {
        call = new NetworkTimeCall<>(call, networkTime);
      }
    }
    return call;
  }

  private static class NetworkTimeCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    private final NetworkTime networkTime;
    private boolean firstMessage = true;

    protected NetworkTimeCall(ClientCall<ReqT, RespT> delegate, NetworkTime networkTime) {
      super(delegate);
      this.networkTime = networkTime;
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {

            @Override
            public void onClose(Status status, Metadata trailers) {
              try {
                networkTime.stop();
              } catch (RuntimeException e) {
                // An unchecked exception means we have bugs in the above try block, force crash
                // Bazel so we can have a chance to look into.
                throw new AssertionError(
                    "networkTime.stop() must not throw unchecked exception: " + networkTime, e);
              } finally {
                // Make sure to call super.onClose, otherwise gRPC will silently hang indefinitely.
                // See https://github.com/grpc/grpc-java/pull/6107.
                super.onClose(status, trailers);
              }
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
