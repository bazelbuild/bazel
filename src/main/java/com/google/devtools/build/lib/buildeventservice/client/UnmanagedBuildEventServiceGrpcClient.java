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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventBlockingStub;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventStub;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import javax.annotation.Nullable;

/**
 * Implementation of BuildEventServiceClient that uploads data using gRPC and is <b>not</b>
 * responsible for closing the channel used to communicate with BES.
 */
public class UnmanagedBuildEventServiceGrpcClient extends BuildEventServiceGrpcClient {
  public UnmanagedBuildEventServiceGrpcClient(
      Channel channel, @Nullable CallCredentials callCredentials) {
    super(channel, callCredentials);
  }

  @VisibleForTesting
  public UnmanagedBuildEventServiceGrpcClient(
      PublishBuildEventStub besAsync, PublishBuildEventBlockingStub besBlocking) {
    super(besAsync, besBlocking);
  }

  @Override
  public void shutdown() {
    // Nothing to do. We handle an unmanaged channel so it's not our responsibility to shut it down.
  }
}
