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

import build.bazel.remote.asset.v1.FetchGrpc;
import build.bazel.remote.execution.v2.ActionCacheGrpc;
import build.bazel.remote.execution.v2.CapabilitiesGrpc;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import com.google.bytestream.ByteStreamGrpc;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import java.time.Duration;

/** Builds Bazel's generated gRPC service config for remote services. */
public final class RemoteGrpcServiceConfig {
  private RemoteGrpcServiceConfig() {}

  public static ImmutableMap<String, ?> create(RemoteOptions options) {
    return create(options.getRemoteTimeout());
  }

  static ImmutableMap<String, ?> create(Duration remoteTimeout) {
    return ImmutableMap.of(
        "methodConfig",
        ImmutableList.of(
            ImmutableMap.of(
                "name",
                ImmutableList.of(
                    ImmutableMap.of("service", ActionCacheGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", CapabilitiesGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", ContentAddressableStorageGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", ByteStreamGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", FetchGrpc.SERVICE_NAME)),
                "timeout",
                remoteTimeout.toSeconds() + "s")));
  }
}
