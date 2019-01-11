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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.metrics.RemoteMetrics;
import com.google.protobuf.Message;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall.SimpleForwardingClientCall;
import io.grpc.ForwardingClientCallListener.SimpleForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;

/**
 * Collects read/write metrics for gRPC-based remote caching and execution.
 */
public class GrpcMetricsInterceptor implements ClientInterceptor {

  private final RemoteMetrics metrics;

  public GrpcMetricsInterceptor(RemoteMetrics metrics) {
    this.metrics = Preconditions.checkNotNull(metrics);
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> methodDescriptor, CallOptions callOptions, Channel channel) {
    ClientCall<ReqT, RespT> call = channel.newCall(methodDescriptor, callOptions);
    return new MetricsCollectingClientCall<>(call);
  }

  private class MetricsCollectingClientCall<ReqT, RespT> extends SimpleForwardingClientCall<ReqT, RespT> {

    protected MetricsCollectingClientCall(ClientCall<ReqT, RespT> delegate) {
      super(delegate);
    }

    @Override
    public void sendMessage(ReqT message) {
      metrics.bytesSent(((Message) message).getSerializedSize());
      super.sendMessage(message);
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      super.start(new SimpleForwardingClientCallListener<RespT>(responseListener) {
        @Override
        public void onMessage(RespT message) {
          metrics.bytesReceived(((Message) message).getSerializedSize());
          super.onMessage(message);
        }
      }, headers);
    }
  }
}
