// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote;

import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionImplBase;
import com.google.longrunning.Operation;
import io.grpc.stub.StreamObserver;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** A basic implementation of an {@link ExecutionImplBase} service. */
final class ExecutionServer extends ExecutionImplBase {
  private final ConcurrentHashMap<String, ExecuteRequest> operationsCache;

  public ExecutionServer(ConcurrentHashMap<String, ExecuteRequest> operationsCache) {
    this.operationsCache = operationsCache;
  }

  @Override
  public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
    // Defer the actual action execution to the Watcher.watch request.
    // There are a lot of errors for which we could fail early here, but deferring them all is
    // simpler.
    final String opName = UUID.randomUUID().toString();
    operationsCache.put(opName, request);
    responseObserver.onNext(Operation.newBuilder().setName(opName).build());
    responseObserver.onCompleted();
  }
}
