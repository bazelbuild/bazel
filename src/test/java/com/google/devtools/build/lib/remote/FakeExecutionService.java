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

public class FakeExecutionService extends ExecutionImplBase {

  private final OperationProvider executeOperationProvider = new OperationProvider();
  private final OperationProvider waitExecutionOperationProvider = new OperationProvider();
  private int execTimes;
  private int waitTimes;

  public static class OperationProvider {

    private final Map<String, Deque<ImmutableList<Supplier<Operation>>>> operationProvider =
        new HashMap<>();

    public void append(String name, ImmutableList<Supplier<Operation>> suppliers) {
      operationProvider.computeIfAbsent(name, key -> new ArrayDeque<>())
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

  public static class OnetimeOperationSupplierBuilder {

    private final OperationProvider provider;
    private final ExecuteRequest request;
    private final List<Supplier<Operation>> operations = new ArrayList<>();

    public OnetimeOperationSupplierBuilder(OperationProvider provider, ExecuteRequest request) {
      this.provider = provider;
      this.request = request;
    }

    public OnetimeOperationSupplierBuilder thenAck() {
      String name = getResourceName(request);
      operations.add(() -> Operation.newBuilder()
          .setName(name)
          .setDone(false)
          .build()
      );
      return this;
    }

    public void thenDone(ExecuteResponse response) {
      String name = getResourceName(request);
      operations.add(() -> Operation.newBuilder()
          .setName(name)
          .setDone(true)
          .setResponse(Any.pack(response))
          .build());
      finish();
    }

    public void thenError(Code code) {
      String name = getResourceName(request);
      operations.add(() -> Operation.newBuilder()
          .setName(name)
          .setDone(true)
          .setError(Status.newBuilder().setCode(code.getNumber()))
          .build());
      finish();
    }

    public void thenError(RuntimeException e) {
      operations.add(() -> {
        throw e;
      });
      finish();
    }

    private void finish() {
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
  public void waitExecution(WaitExecutionRequest request,
      StreamObserver<Operation> responseObserver) {
    waitTimes += 1;
    serve(responseObserver, request.getName(), waitExecutionOperationProvider);
  }

  private static void serve(StreamObserver<Operation> responseObserver, String name,
      OperationProvider provider) {
    if (provider.hasNext(name)) {
      ImmutableList<Supplier<Operation>> suppliers = provider.next(name);
      for (Supplier<Operation> supplier : suppliers) {
        responseObserver.onNext(supplier.get());
      }
      responseObserver.onCompleted();
    } else {
      responseObserver.onError(io.grpc.Status.UNAVAILABLE.asRuntimeException());
    }
  }

  public int getExecTimes() {
    return execTimes;
  }

  public int getWaitTimes() {
    return waitTimes;
  }
}
