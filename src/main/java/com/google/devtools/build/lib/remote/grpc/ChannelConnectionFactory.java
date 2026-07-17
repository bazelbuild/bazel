// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.grpc;

import static java.util.concurrent.TimeUnit.SECONDS;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;

/** A {@link ConnectionFactory} which creates {@link ChannelConnection}. */
public interface ChannelConnectionFactory extends ConnectionFactory {
  @Override
  Single<? extends ChannelConnection> create();

  /** Returns the max concurrency supported by the underlying {@link ManagedChannel}. */
  int maxConcurrency();

  /** A {@link Connection} which wraps around {@link ManagedChannel}. */
  class ChannelConnection implements Connection {
    private final ManagedChannel channel;

    public ChannelConnection(ManagedChannel channel) {
      this.channel = channel;
    }

    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> call(
        MethodDescriptor<ReqT, RespT> method, CallOptions options) {
      return channel.newCall(method, options);
    }

    @Override
    public void close() throws IOException {
      // Clear interrupted status to prevent failure to await, indicated with #13512
      boolean wasInterrupted = Thread.interrupted();
      // There is a bug (b/183340374) in gRPC that client doesn't try to close connections with
      // shutdown() if the channel received GO_AWAY frames. Using shutdownNow() here as a
      // workaround.
      try {
        channel.shutdownNow();
        channel.awaitTermination(Integer.MAX_VALUE, SECONDS);
      } catch (InterruptedException e) {
        throw new IOException(e.getMessage(), e);
      } finally {
        if (wasInterrupted) {
          Thread.currentThread().interrupt();
        }
      }
    }

    public ManagedChannel getChannel() {
      return channel;
    }
  }
}
