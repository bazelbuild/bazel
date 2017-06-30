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
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionBlockingStub;
import com.google.longrunning.Operation;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.Status;
import com.google.watcher.v1.Change;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
import com.google.watcher.v1.WatcherGrpc;
import com.google.watcher.v1.WatcherGrpc.WatcherBlockingStub;
import io.grpc.Channel;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.StatusProto;
import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
public class GrpcRemoteExecutor {
  private final RemoteOptions options;
  private final ChannelOptions channelOptions;
  private final Channel channel;
  private final Retrier retrier;

  // Reuse the gRPC stubs.
  private final Supplier<ExecutionBlockingStub> execBlockingStub;
  private final Supplier<WatcherBlockingStub> watcherBlockingStub;

  public static boolean isRemoteExecutionOptions(RemoteOptions options) {
    return options.remoteExecutor != null;
  }

  public GrpcRemoteExecutor(Channel channel, ChannelOptions channelOptions, RemoteOptions options) {
    this.options = options;
    this.channelOptions = channelOptions;
    this.channel = channel;
    this.retrier = new Retrier(options);
    execBlockingStub =
        Suppliers.memoize(
            () ->
                ExecutionGrpc.newBlockingStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials())
                    .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS));
    // Do not set a deadline on this call, because it is hard to estimate in
    // advance how much time we should give the server to execute an action
    // remotely (scheduling, queing, optional retries, etc.)
    // It is the server's responsibility to respect the Action timeout field.
    watcherBlockingStub =
        Suppliers.memoize(
            () ->
                WatcherGrpc.newBlockingStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials()));
  }

  private @Nullable ExecuteResponse getOperationResponse(Operation op)
      throws IOException, UserExecException {
    if (op.getResultCase() == Operation.ResultCase.ERROR) {
      StatusRuntimeException e = StatusProto.toStatusRuntimeException(op.getError());
      if (e.getStatus().getCode() == Code.DEADLINE_EXCEEDED) {
        // This was caused by the command itself exceeding the timeout,
        // therefore it is not retriable.
        // TODO(olaola): this should propagate a timeout SpawnResult instead of raising.
        throw new UserExecException("Remote execution time out", true);
      }
      throw e;
    }
    if (op.getDone()) {
      Preconditions.checkState(op.getResultCase() != Operation.ResultCase.RESULT_NOT_SET);
      try {
        return op.getResponse().unpack(ExecuteResponse.class);
      } catch (InvalidProtocolBufferException e) {
        throw new IOException(e);
      }
    }
    return null;
  }

  public ExecuteResponse executeRemotely(ExecuteRequest request)
      throws InterruptedException, IOException, UserExecException {
    Operation op = retrier.execute(() -> execBlockingStub.get().execute(request));
    ExecuteResponse resp = getOperationResponse(op);
    if (resp != null) {
      return resp;
    }
    Request wr = Request.newBuilder().setTarget(op.getName()).build();
    return retrier.execute(
        () -> {
          Iterator<ChangeBatch> replies = watcherBlockingStub.get().watch(wr);
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
                  // TODO(olaola): either make this retriable, or use a different exception.
                  throw new IOException(
                      String.format("Operation %s lost on the remote server.", op.getName()));
                case EXISTS:
                  Operation o;
                  try {
                    o = ch.getData().unpack(Operation.class);
                  } catch (InvalidProtocolBufferException e) {
                    throw new RuntimeException(e);
                  }
                  ExecuteResponse r = getOperationResponse(o);
                  if (r != null) {
                    return r;
                  }
                  continue;
                default:
                  // This can only happen if the enum gets unexpectedly extended.
                  throw new IOException(String.format("Illegal change state: %s", ch.getState()));
              }
            }
          }
          throw new IOException(
              String.format("Watch request for %s terminated with no result.", op.getName()));
        });
  }
}
