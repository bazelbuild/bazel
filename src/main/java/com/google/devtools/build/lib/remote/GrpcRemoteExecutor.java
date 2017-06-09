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
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionBlockingStub;
import com.google.longrunning.Operation;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.Durations;
import io.grpc.Channel;
import io.grpc.protobuf.StatusProto;
import java.util.concurrent.TimeUnit;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
public class GrpcRemoteExecutor {
  private final RemoteOptions options;
  private final ChannelOptions channelOptions;
  private final Channel channel;

  public static boolean isRemoteExecutionOptions(RemoteOptions options) {
    return options.remoteExecutor != null;
  }

  public GrpcRemoteExecutor(Channel channel, ChannelOptions channelOptions, RemoteOptions options) {
    this.options = options;
    this.channelOptions = channelOptions;
    this.channel = channel;
  }

  public ExecuteResponse executeRemotely(ExecuteRequest request) {
    // TODO(olaola): handle longrunning Operations by using the Watcher API to wait for results.
    // For now, only support actions with wait_for_completion = true.
    Preconditions.checkArgument(request.getWaitForCompletion());
    int actionSeconds = (int) Durations.toSeconds(request.getAction().getTimeout());
    ExecutionBlockingStub stub =
        ExecutionGrpc.newBlockingStub(channel)
            .withCallCredentials(channelOptions.getCallCredentials())
            .withDeadlineAfter(options.remoteTimeout + actionSeconds, TimeUnit.SECONDS);
    Operation op = stub.execute(request);
    Preconditions.checkState(op.getDone());
    Preconditions.checkState(op.getResultCase() != Operation.ResultCase.RESULT_NOT_SET);
    if (op.getResultCase() == Operation.ResultCase.ERROR) {
      throw StatusProto.toStatusRuntimeException(op.getError());
    }
    try {
      return op.getResponse().unpack(ExecuteResponse.class);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }
}
