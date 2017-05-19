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
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import io.grpc.ManagedChannel;
import java.util.Iterator;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
public class GrpcRemoteExecutor extends GrpcActionCache {
  public static boolean isRemoteExecutionOptions(RemoteOptions options) {
    return options.remoteExecutor != null;
  }

  private final GrpcExecutionInterface executionIface;

  public GrpcRemoteExecutor(
      RemoteOptions options,
      GrpcCasInterface casIface,
      GrpcExecutionCacheInterface cacheIface,
      GrpcExecutionInterface executionIface) {
    super(options, casIface, cacheIface);
    this.executionIface = executionIface;
  }

  public GrpcRemoteExecutor(
      ManagedChannel channel, ChannelOptions channelOptions, RemoteOptions options) {
    super(
        options,
        GrpcInterfaces.casInterface(options.remoteTimeout, channel, channelOptions),
        GrpcInterfaces.executionCacheInterface(
            options.remoteTimeout, channel, channelOptions));
    this.executionIface =
        GrpcInterfaces.executionInterface(options.remoteTimeout, channel, channelOptions);
  }

  public ExecuteReply executeRemotely(ExecuteRequest request) {
    Iterator<ExecuteReply> replies = executionIface.execute(request);
    ExecuteReply reply = null;
    while (replies.hasNext()) {
      reply = replies.next();
      // We can handle the action execution progress here.
    }
    if (reply == null) {
      return ExecuteReply.newBuilder()
          .setStatus(
              ExecutionStatus.newBuilder()
                  .setExecuted(false)
                  .setSucceeded(false)
                  .setError(ExecutionStatus.ErrorCode.UNKNOWN_ERROR)
                  .setErrorDetail("Remote server terminated the connection"))
          .build();
    }
    return reply;
  }
}
