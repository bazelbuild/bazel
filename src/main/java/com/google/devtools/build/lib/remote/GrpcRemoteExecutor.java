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

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionBlockingStub;
import com.google.longrunning.Operation;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.Durations;
import com.google.rpc.Status;
import com.google.watcher.v1.Change;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
import com.google.watcher.v1.WatcherGrpc;
import com.google.watcher.v1.WatcherGrpc.WatcherBlockingStub;
import io.grpc.Channel;
import io.grpc.protobuf.StatusProto;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
public class GrpcRemoteExecutor {
  private final RemoteOptions options;
  private final ChannelOptions channelOptions;
  private final Channel channel;

  // Reuse the gRPC stub.
  private final Supplier<ExecutionBlockingStub> execBlockingStub =
    Suppliers.memoize(
        new Supplier<ExecutionBlockingStub>() {
          @Override
          public ExecutionBlockingStub get() {
            return ExecutionGrpc.newBlockingStub(channel)
                .withCallCredentials(channelOptions.getCallCredentials())
                .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
          }
        });

  public static boolean isRemoteExecutionOptions(RemoteOptions options) {
    return options.remoteExecutor != null;
  }

  public GrpcRemoteExecutor(Channel channel, ChannelOptions channelOptions, RemoteOptions options) {
    this.options = options;
    this.channelOptions = channelOptions;
    this.channel = channel;
  }

  private @Nullable ExecuteResponse getOperationResponse(Operation op) {
    if (op.getResultCase() == Operation.ResultCase.ERROR) {
      throw StatusProto.toStatusRuntimeException(op.getError());
    }
    if (op.getDone()) {
      Preconditions.checkState(op.getResultCase() != Operation.ResultCase.RESULT_NOT_SET);
      try {
        return op.getResponse().unpack(ExecuteResponse.class);
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
    }
    return null;
  }

  public ExecuteResponse executeRemotely(ExecuteRequest request) {
    Operation op = execBlockingStub.get().execute(request);
    ExecuteResponse resp = getOperationResponse(op);
    if (resp != null) {
      return resp;
    }
    int actionSeconds = (int) Durations.toSeconds(request.getAction().getTimeout());
    WatcherBlockingStub stub =
        WatcherGrpc.newBlockingStub(channel)
            .withCallCredentials(channelOptions.getCallCredentials())
            .withDeadlineAfter(options.remoteTimeout + actionSeconds, TimeUnit.SECONDS);
    Request wr = Request.newBuilder().setTarget(op.getName()).build();
    Iterator<ChangeBatch> replies = stub.watch(wr);
    while (replies.hasNext()) {
      ChangeBatch cb = replies.next();
      for (Change ch : cb.getChangesList()) {
        switch (ch.getState()) {
          case INITIAL_STATE_SKIPPED:
            continue;
          case ERROR:
            try {
              throw StatusProto.toStatusRuntimeException(ch.getData().unpack(Status.class));
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
          case DOES_NOT_EXIST:
            throw new RuntimeException(
                String.format("Operation %s lost on the remote server.", op.getName()));
          case EXISTS:
            try {
              op = ch.getData().unpack(Operation.class);
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
            resp = getOperationResponse(op);
            if (resp != null) {
              return resp;
            }
            continue;
          default:
            throw new RuntimeException(String.format("Illegal change state: %s", ch.getState()));
        }
      }
    }
    throw new RuntimeException(
        String.format("Watch request for %s terminated with no result.", op.getName()));
  }
}
