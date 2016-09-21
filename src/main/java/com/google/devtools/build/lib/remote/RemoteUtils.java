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

import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import java.net.URI;
import java.net.URISyntaxException;

/** Helper methods for gRPC calls */
@ThreadSafe
public final class RemoteUtils {
  public static ManagedChannel createChannel(String hostAndPort)
      throws InvalidConfigurationException {
    try {
      URI uri = new URI("dummy://" + hostAndPort);
      if (uri.getHost() == null || uri.getPort() == -1) {
        throw new URISyntaxException("Invalid host or port.", "");
      }
      return NettyChannelBuilder.forAddress(uri.getHost(), uri.getPort())
          .usePlaintext(true)
          .build();
    } catch (URISyntaxException e) {
      throw new InvalidConfigurationException(
          "Invalid argument for the address of remote cache server: " + hostAndPort);
    }
  }
}
