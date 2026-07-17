// Copyright 2023 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import io.grpc.ManagedChannel;
import io.reactivex.rxjava3.core.Single;

/**
 * A {@link ChannelConnectionFactory} that create {@link ChannelConnectionWithServerCapabilities}.
 */
public interface ChannelConnectionWithServerCapabilitiesFactory extends ChannelConnectionFactory {

  @Override
  Single<? extends ChannelConnectionWithServerCapabilities> create();

  /** A {@link ChannelConnection} that provides {@link ServerCapabilities}. */
  class ChannelConnectionWithServerCapabilities extends ChannelConnection {
    private final Single<ServerCapabilities> serverCapabilities;

    public ChannelConnectionWithServerCapabilities(
        ManagedChannel channel, Single<ServerCapabilities> serverCapabilities) {
      super(channel);
      this.serverCapabilities = serverCapabilities;
    }

    public Single<ServerCapabilities> getServerCapabilities() {
      return serverCapabilities;
    }
  }
}
