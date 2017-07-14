// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.ManagedChannel;
import io.grpc.netty.NegotiationType;
import io.grpc.netty.NettyChannelBuilder;
import java.io.IOException;

/** Helper methods for gRPC calls */
@ThreadSafe
final class GrpcUtils {
  public static ManagedChannel createChannel(String target, ChannelOptions channelOptions)
      throws IOException {
    try {
      NettyChannelBuilder builder =
          NettyChannelBuilder.forTarget(target)
              .negotiationType(
                  channelOptions.tlsEnabled() ? NegotiationType.TLS : NegotiationType.PLAINTEXT);
      if (channelOptions.getSslContext() != null) {
        builder.sslContext(channelOptions.getSslContext());
        if (channelOptions.getTlsAuthorityOverride() != null) {
          builder.overrideAuthority(channelOptions.getTlsAuthorityOverride());
        }
      }
      return builder.build();
    } catch (RuntimeException e) {
      // gRPC might throw all kinds of RuntimeExceptions: StatusRuntimeException,
      // IllegalStateException, NullPointerException, ...
      String message = "Failed to connect to the remote cache/executor '%s': %s";
      throw new IOException(String.format(message, target, e.getMessage()));
    }
  }
}
