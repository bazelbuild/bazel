// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventservice.client;

import io.grpc.CallCredentials;
import io.grpc.ManagedChannel;
import javax.annotation.Nullable;

/**
 * Implementation of BuildEventServiceClient that uploads data using gRPC and is responsible for
 * closing the channel used to communicate with BES.
 */
public class ManagedBuildEventServiceGrpcClient extends BuildEventServiceGrpcClient {
  private final ManagedChannel channel;

  public ManagedBuildEventServiceGrpcClient(
      ManagedChannel channel, @Nullable CallCredentials callCredentials) {
    super(channel, callCredentials);
    this.channel = channel;
  }

  @Override
  public void shutdown() {
    channel.shutdown();
  }
}
