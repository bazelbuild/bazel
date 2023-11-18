// Copyright 2020 The Bazel Authors. All rights reserved.
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
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.rpc.Code;
import com.google.rpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/** A fake implementation of the {@link ExecutionImplBase}. */
public class FakeExecutionService extends ExecutionImplBase {

  private final OperationProvider executeOperationProvider = new OperationProvider();
  private final OperationProvider waitExecutionOperationProvider = new OperationProvider();
  private int execTimes;
  private int waitTimes;

  static class OperationProvider {

    // Map from the request to the list of operations to be returned for each instance of
    // that request, with Supplier used for either throwing an exception or returning an Operation.
    private final Map<String, Deque<ImmutableList<Supplier<Operation>>>> operationProvider =
        new HashMap<>();

    public void append(String name, ImmutableList<Supplier<Operation>> suppliers) {
      operationProvider
          .computeIfAbsent(name, key -> new ArrayDeque<>())
          .add(ImmutableList.copyOf(suppliers));
    }

    public boolean hasNext(String name) {
      Deque<ImmutableList<Supplier<Operation>>> q = operationProvider.get(name);
      return q != null && !q.isEmpty();
    }

    public ImmutableList<Supplier<Operation>> next(String name) {
      return operationProvider.get(name).removeFirst();
    }
  }

  public OnetimeOperationSupplierBuilder whenExecute(ExecuteRequest request) {
    return new OnetimeOperationSupplierBuilder(executeOperationProvider, request);
  }

  public OnetimeOperationSupplierBuilder whenWaitExecution(ExecuteRequest request) {
    return new OnetimeOperationSupplierBuilder(waitExecutionOperationProvider, request);
  }

  public static Operation ackOperation(ExecuteRequest request) {
    return Operation.newBuilder().setName(getResourceName(request)).setDone(false).build();
  }

  public static Operation doneOperation(ExecuteRequest request, ExecuteResponse response) {
    return Operation.newBuilder()
        .setName(getResourceName(request))
        .setDone(true)
        .setResponse(Any.pack(response))
        .build();
  }

  static class OnetimeOperationSupplierBuilder {

    private final OperationProvider provider;
    private final ExecuteRequest request;
    private final List<Supplier<Operation>> operations = new ArrayList<>();

    public OnetimeOperationSupplierBuilder(OperationProvider provider, ExecuteRequest request) {
      this.provider = provider;
      this.request = request;
    }

    @CanIgnoreReturnValue
    public OnetimeOperationSupplierBuilder thenAck() {
      Operation operation = ackOperation(request);
      operations.add(() -> operation);
      return this;
    }

    public void thenDone() {
      Operation operation =
          Operation.newBuilder().setName(getResourceName(request)).setDone(true).build();
      operations.add(() -> operation);
      finish();
    }

    public void thenDone(ExecuteResponse response) {
      Operation operation = doneOperation(request, response);
      operations.add(() -> operation);
      finish();
    }

    public void thenError(Code code) {
      // From REAPI Spec:
      // > Errors discovered during creation of the `Operation` will be reported
      // > as gRPC Status errors, while errors that occurred while running the
      // > action will be reported in the `status` field of the `ExecuteResponse`. The
      // > server MUST NOT set the `error` field of the `Operation` proto.
      Operation operation =
          doneOperation(
              request,
              ExecuteResponse.newBuilder()
                  .setStatus(Status.newBuilder().setCode(code.getNumber()))
                  .build());
      operations.add(() -> operation);
      finish();
    }

    public void thenError(RuntimeException e) {
      operations.add(
          () -> {
            throw e;
          });
      finish();
    }

    public void finish() {
      String name = getResourceName(request);
      provider.append(name, ImmutableList.copyOf(operations));
    }
  }

  public static String getResourceName(ExecuteRequest request) {
    return String.format("operations/%s", request.getActionDigest().getHash());
  }

  @Override
  public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
    execTimes += 1;
    serve(responseObserver, getResourceName(request), executeOperationProvider);
  }

  @Override
  public void waitExecution(
      WaitExecutionRequest request, StreamObserver<Operation> responseObserver) {
    waitTimes += 1;
    serve(responseObserver, request.getName(), waitExecutionOperationProvider);
  }

  private static void serve(
      StreamObserver<Operation> responseObserver, String name, OperationProvider provider) {
    if (provider.hasNext(name)) {
      ImmutableList<Supplier<Operation>> suppliers = provider.next(name);
      for (Supplier<Operation> supplier : suppliers) {
        responseObserver.onNext(supplier.get());
      }
      responseObserver.onCompleted();
    } else {
      responseObserver.onError(io.grpc.Status.UNIMPLEMENTED.asRuntimeException());
    }
  }

  public int getExecTimes() {
    return execTimes;
  }

  public int getWaitTimes() {
    return waitTimes;
  }
}
