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

import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionGrpc;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionBlockingStub;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.longrunning.Operation;
import com.google.rpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** A remote work executor that uses gRPC for communicating the work, inputs and outputs. */
@ThreadSafe
class GrpcRemoteExecutor implements RemoteExecutionClient {

  private final ReferenceCountedChannel channel;
  private final CallCredentialsProvider callCredentialsProvider;
  private final RemoteRetrier retrier;

  private final AtomicBoolean closed = new AtomicBoolean();

  public GrpcRemoteExecutor(
      ReferenceCountedChannel channel,
      CallCredentialsProvider callCredentialsProvider,
      RemoteRetrier retrier) {
    this.channel = channel;
    this.callCredentialsProvider = callCredentialsProvider;
    this.retrier = retrier;
  }

  private ExecutionBlockingStub execBlockingStub(RequestMetadata metadata) {
    return ExecutionGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataInterceptor(metadata))
        .withCallCredentials(callCredentialsProvider.getCallCredentials());
  }

  private void handleStatus(Status statusProto, @Nullable ExecuteResponse resp) {
    if (statusProto.getCode() == Code.OK.value()) {
      return;
    }
    throw new ExecutionStatusException(statusProto, resp);
  }

  @Nullable
  private ExecuteResponse getOperationResponse(Operation op) throws IOException {
    if (op.getResultCase() == Operation.ResultCase.ERROR) {
      handleStatus(op.getError(), null);
    }
    if (op.getDone()) {
      Preconditions.checkState(op.getResultCase() != Operation.ResultCase.RESULT_NOT_SET);
      ExecuteResponse resp = op.getResponse().unpack(ExecuteResponse.class);
      if (resp.hasStatus()) {
        handleStatus(resp.getStatus(), resp);
      }
      Preconditions.checkState(
          resp.hasResult(), "Unexpected result of remote execution: no result");
      return resp;
    }
    return null;
  }

  /* Execute has two components: the Execute call and (optionally) the WaitExecution call.
   * This is the simple flow without any errors:
   *
   * - A call to Execute returns streamed updates on an Operation object.
   * - We wait until the Operation is finished.
   *
   * Error possibilities:
   * - An Execute call may fail with a retriable error (raise a StatusRuntimeException).
   *   - If the failure occurred before the first Operation is returned, we retry the call.
   *   - Otherwise, we call WaitExecution on the Operation.
   * - A WaitExecution call may fail with a retriable error (raise a StatusRuntimeException).
   *   In that case, we retry the WaitExecution call on the same operation object.
   * - A WaitExecution call may fail with a NOT_FOUND error (raise a StatusRuntimeException).
   *   That means the Operation was lost on the server, and we will retry to Execute.
   * - Any call can return an Operation object with an error status in the result. Such Operations
   *   are completed and failed; however, some of these errors may be retriable. These errors should
   *   trigger a retry of the Execute call, resulting in a new Operation.
   * */
  @Override
  public ExecuteResponse executeRemotely(
      RemoteActionExecutionContext context, ExecuteRequest request, OperationObserver observer)
      throws IOException, InterruptedException {
    // Execute has two components: the Execute call and (optionally) the WaitExecution call.
    // This is the simple flow without any errors:
    //
    // - A call to Execute returns streamed updates on an Operation object.
    // - We wait until the Operation is finished.
    //
    // Error possibilities:
    // - An Execute call may fail with a retriable error (raise a StatusRuntimeException).
    //   - If the failure occurred before the first Operation is returned, we retry the call.
    //   - Otherwise, we call WaitExecution on the Operation.
    // - A WaitExecution call may fail with a retriable error (raise a StatusRuntimeException).
    //   In that case, we retry the WaitExecution call on the same operation object.
    // - A WaitExecution call may fail with a NOT_FOUND error (raise a StatusRuntimeException).
    //   That means the Operation was lost on the server, and we will retry to Execute.
    // - Any call can return an Operation object with an error status in the result. Such Operations
    //   are completed and failed; however, some of these errors may be retriable. These errors
    //   should trigger a retry of the Execute call, resulting in a new Operation.

    // Will be modified by the retried handler.
    final AtomicReference<Operation> operation =
        new AtomicReference<>(Operation.getDefaultInstance());
    final AtomicBoolean waitExecution =
        new AtomicBoolean(false); // Whether we should call WaitExecution.
    try {
      return Utils.refreshIfUnauthenticated(
          () ->
              retrier.execute(
                  () -> {
                    // Retry calls to Execute()/WaitExecute() "infinitely" if the server terminates
                    // one of
                    // them status OK and an Operation that does not have done=True set. This is
                    // legal
                    // according to the remote execution protocol i.e. if the execution takes longer
                    // than a connection timeout. This is not an error condition and is thus handled
                    // outside of the retrier.
                    while (true) {
                      final Iterator<Operation> replies;
                      if (waitExecution.get()) {
                        WaitExecutionRequest wr =
                            WaitExecutionRequest.newBuilder()
                                .setName(operation.get().getName())
                                .build();
                        replies = execBlockingStub(context.getRequestMetadata()).waitExecution(wr);
                      } else {
                        replies = execBlockingStub(context.getRequestMetadata()).execute(request);
                      }
                      try {
                        while (replies.hasNext()) {
                          Operation o = replies.next();
                          operation.set(o);
                          waitExecution.set(!operation.get().getDone());

                          // Update execution progress to the caller.
                          //
                          // After called `execute` above, the action is actually waiting for an
                          // available
                          // gRPC connection to be sent. Once we get a reply from server, we know
                          // the
                          // connection is up and indicate to the caller the fact by forwarding the
                          // `operation`.
                          //
                          // The accurate execution status of the action relies on the server
                          // implementation:
                          //   1. Server can reply the accurate status in
                          // `operation.metadata.stage`;
                          //   2. Server may send a reply without metadata. In this case, we assume
                          // the
                          //      action is accepted by the server and will be executed ASAP;
                          //   3. Server may execute the action silently and send a reply once it is
                          // done.
                          observer.onNext(o);

                          ExecuteResponse r = getOperationResponse(o);
                          if (r != null) {
                            return r;
                          }
                        }
                        // The operation completed successfully but without a result.
                        if (!waitExecution.get()) {
                          throw new IOException(
                              String.format(
                                  "Remote server error: execution request for %s terminated with no"
                                      + " result.",
                                  operation.get().getName()));
                        }
                      } catch (StatusRuntimeException e) {
                        if (e.getStatus().getCode() == Code.NOT_FOUND) {
                          // Operation was lost on the server. Retry Execute.
                          waitExecution.set(false);
                        }
                        throw e;
                      } finally {
                        // The blocking streaming call closes correctly only when trailers and a
                        // Status
                        // are received from the server so that onClose() is called on this call's
                        // CallListener. Under normal circumstances (no cancel/errors), these are
                        // guaranteed to be sent by the server only if replies.hasNext() has been
                        // called
                        // after all replies from the stream have been consumed.
                        try {
                          while (replies.hasNext()) {
                            replies.next();
                          }
                        } catch (StatusRuntimeException e) {
                          // Cleanup: ignore exceptions, because the meaningful errors have already
                          // been
                          // propagated.
                        }
                      }
                    }
                  }),
          callCredentialsProvider);
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    channel.release();
  }
}
