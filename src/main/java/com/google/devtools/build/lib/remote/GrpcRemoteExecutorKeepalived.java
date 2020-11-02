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
import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteRetrier.ProgressiveBackoff;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.longrunning.Operation;
import com.google.rpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A remote work executor that uses gRPC for communicating the work, inputs and outputs.
 *
 * <p>It differs from {@link GrpcRemoteExecutor} by setting timeout on each execution calls to
 * ensure we never be stuck due to network issues.
 *
 * @see <a href="https://docs.google.com/document/d/1NgDPsCIwprDdqC1zj0qQrh5KGK2hQTSTux1DAvi4rSc">
 * Keepalived Remote Execution</a>
 */
@ThreadSafe
public class GrpcRemoteExecutorKeepalived implements RemoteExecutionClient {

  private final RemoteOptions remoteOptions;
  private final ReferenceCountedChannel channel;
  private final CallCredentialsProvider callCredentialsProvider;
  private final RemoteRetrier retrier;

  private final AtomicBoolean closed = new AtomicBoolean();

  public GrpcRemoteExecutorKeepalived(
      RemoteOptions remoteOptions,
      ReferenceCountedChannel channel,
      CallCredentialsProvider callCredentialsProvider,
      RemoteRetrier retrier) {
    this.remoteOptions = remoteOptions;
    this.channel = channel;
    this.callCredentialsProvider = callCredentialsProvider;
    this.retrier = retrier;
  }

  private ExecutionBlockingStub executionBlockingStub() {
    return ExecutionGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(callCredentialsProvider.getCallCredentials())
        .withDeadlineAfter(remoteOptions.remoteTimeout.getSeconds(), TimeUnit.SECONDS);
  }

  private static class Execution {
    private final ExecuteRequest request;
    private final OperationObserver observer;
    private final RemoteRetrier retrier;
    private final CallCredentialsProvider callCredentialsProvider;
    private final ProgressiveBackoff backoff;
    private final Supplier<ExecutionBlockingStub> executionBlockingStubSupplier;

    private Operation lastOperation;

    Execution(ExecuteRequest request,
        OperationObserver observer,
        RemoteRetrier retrier,
        CallCredentialsProvider callCredentialsProvider,
        Supplier<ExecutionBlockingStub> executionBlockingStubSupplier) {
      this.request = request;
      this.observer = observer;
      this.retrier = retrier;
      this.callCredentialsProvider = callCredentialsProvider;
      this.backoff = new ProgressiveBackoff(this.retrier::newBackoff);
      this.executionBlockingStubSupplier = executionBlockingStubSupplier;
    }

    ExecuteResponse start() throws IOException, InterruptedException {
      // Execute has two components: the Execute call and (optionally) the WaitExecution call.
      // This is the simple flow without any errors:
      //
      // - A call to Execute returns streamed updates on an Operation object.
      // - We wait until the Operation is finished.
      //
      // Error possibilities:
      // - An Execute call may fail with a retriable error (raise a StatusRuntimeException).
      //   - If the failure occurred before the first Operation is returned and tells us the
      //     execution is accepted, we retry the call.
      //   - Otherwise, we call WaitExecution on the Operation.
      // - A WaitExecution call may fail with a retriable error (raise a StatusRuntimeException).
      //   In that case, we retry the WaitExecution call on the same operation object.
      // - A WaitExecution call may fail with a NOT_FOUND error (raise a StatusRuntimeException).
      //   That means the Operation was lost on the server, and we will retry to Execute.
      // - Any call can return an Operation object with an error status in the result. Such
      //   Operations are completed and failed; however, some of these errors may be retriable.
      //   These errors should trigger a retry of the Execute call, resulting in a new Operation.
      Preconditions.checkState(lastOperation == null);

      ExecuteResponse response = null;
      while (response == null) {
        response = retrier
            .execute(() -> Utils.refreshIfUnauthenticated(this::execute, callCredentialsProvider),
                backoff);

        if (response == null) {
          response = retrier.execute(
              () -> Utils.refreshIfUnauthenticated(this::waitExecution, callCredentialsProvider),
              backoff);
        }
      }

      return response;
    }

    @Nullable
    ExecuteResponse execute() throws IOException {
      Preconditions.checkState(lastOperation == null);

      try {
        Iterator<Operation> operations = executionBlockingStubSupplier.get().execute(request);
        return handleStreamOperations(operations, /* resetBackoff */ false);
      } catch (StatusRuntimeException e) {
        if (lastOperation != null) {
          // By returning null, we are going to call WaitExecution in a loop.
          return null;
        }
        throw new IOException(e);
      }
    }

    @Nullable
    ExecuteResponse waitExecution() throws IOException {
      Preconditions.checkState(lastOperation != null);

      WaitExecutionRequest request = WaitExecutionRequest.newBuilder()
          .setName(lastOperation.getName())
          .build();
      try {
        Iterator<Operation> operations = executionBlockingStubSupplier.get()
            .waitExecution(request);
        return handleStreamOperations(operations, /* resetBackoff */ true);
      } catch (StatusRuntimeException e) {
        if (e.getStatus().getCode() == Code.NOT_FOUND) {
          // Operation was lost on the server. Retry Execute.
          lastOperation = null;
          return null;
        }
        throw new IOException(e);
      }
    }

    ExecuteResponse handleStreamOperations(Iterator<Operation> operations, Boolean resetBackoff)
        throws IOException {
      try {
        while (operations.hasNext()) {
          Operation operation = operations.next();

          if (resetBackoff) {
            // Assuming the server has made progress since we received the response. Reset the backoff
            // so that this request has a full deck of retries
            backoff.reset();
          }

          ExecuteResponse response = handleOperation(operation, observer);
          if (response != null) {
            return response;
          }

          lastOperation = operation;
        }

        // The operation completed successfully but without a result.
        throw new IOException("Remote server error: execution terminated with no result.");
      } finally {
        close(operations);
      }
    }

    void close(Iterator<Operation> operations) {
      // The blocking streaming call closes correctly only when trailers and a Status are received
      // from the server so that onClose() is called on this call's CallListener. Under normal
      // circumstances (no cancel/errors), these are guaranteed to be sent by the server only if
      // operations.hasNext() has been called after all replies from the stream have been
      // consumed.
      try {
        while (operations.hasNext()) {
          operations.next();
        }
      } catch (StatusRuntimeException e) {
        // Cleanup: ignore exceptions, because the meaningful errors have already been propagated.
      }
    }

    static void throwIfError(Status status, @Nullable ExecuteResponse resp) {
      if (status.getCode() == Code.OK.value()) {
        return;
      }
      throw new ExecutionStatusException(status, resp);
    }

    @Nullable
    static ExecuteResponse handleOperation(Operation operation, OperationObserver observer)
        throws IOException {
      // Update execution progress to the caller.
      //
      // After called `execute` above, the action is actually waiting for an available gRPC
      // connection to be sent. Once we get a reply from server, we know the connection is up and
      // indicate to the caller the fact by forwarding the `operation`.
      //
      // The accurate execution status of the action relies on the server
      // implementation:
      //   1. Server can reply the accurate status in `operation.metadata.stage`;
      //   2. Server may send a reply without metadata. In this case, we assume the action is
      //      accepted by the server and will be executed ASAP;
      //   3. Server may execute the action silently and send a reply once it is done.
      observer.onNext(operation);

      if (operation.getResultCase() == Operation.ResultCase.ERROR) {
        throwIfError(operation.getError(), null);
      }

      if (operation.getDone()) {
        Preconditions.checkState(operation.getResultCase() != Operation.ResultCase.RESULT_NOT_SET);
        ExecuteResponse response = operation.getResponse().unpack(ExecuteResponse.class);
        if (response.hasStatus()) {
          throwIfError(response.getStatus(), response);
        }
        Preconditions.checkState(
            response.hasResult(), "Unexpected result of remote execution: no result");
        return response;
      }

      return null;
    }
  }

  @Override
  public ExecuteResponse executeRemotely(ExecuteRequest request, OperationObserver observer)
      throws IOException, InterruptedException {
    Execution execution = new Execution(request, observer, retrier, callCredentialsProvider,
        this::executionBlockingStub);
    return execution.start();
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    channel.release();
  }
}
