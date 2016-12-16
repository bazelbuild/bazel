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
import com.google.devtools.build.lib.remote.ExecuteServiceGrpc.ExecuteServiceBlockingStub;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import io.grpc.ManagedChannel;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
public class RemoteWorkExecutor {
  /** Channel over which to send work to run remotely. */
  private final ManagedChannel channel;
  private final RemoteOptions options;

  public RemoteWorkExecutor(RemoteOptions options) throws InvalidConfigurationException {
    this.options = options;
    channel = RemoteUtils.createChannel(options.remoteWorker);
  }

  public static boolean isRemoteExecutionOptions(RemoteOptions options) {
    return options.remoteWorker != null;
  }

  public ExecuteReply executeRemotely(ExecuteRequest request) {
    ExecuteServiceBlockingStub stub =
        ExecuteServiceGrpc.newBlockingStub(channel)
            .withDeadlineAfter(
                options.grpcTimeoutSeconds + request.getTimeoutMillis() / 1000, TimeUnit.SECONDS);
    Iterator<ExecuteReply> replies = stub.execute(request);
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
