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

import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.rpc.Code;
import com.google.rpc.Status;
import com.google.watcher.v1.Change;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
import com.google.watcher.v1.WatcherGrpc.WatcherImplBase;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;

/** A basic implementation of an {@link WatcherImplBase} service. */
final class WatcherServer extends WatcherImplBase {
  private static final Logger logger = Logger.getLogger(WatcherServer.class.getName());

  private final ConcurrentHashMap<String, ListenableFuture<ActionResult>> operationsCache;

  public WatcherServer(ConcurrentHashMap<String, ListenableFuture<ActionResult>> operationsCache) {
    this.operationsCache = operationsCache;
  }

  @Override
  public void watch(Request wr, StreamObserver<ChangeBatch> responseObserver) {
    final String opName = wr.getTarget();
    ListenableFuture<ActionResult> future = operationsCache.get(opName);
    if (future == null) {
      responseObserver.onError(
          StatusProto.toStatusRuntimeException(
              Status.newBuilder()
                  .setCode(Code.NOT_FOUND.getNumber())
                  .setMessage("Operation not found: " + opName)
                  .build()));
      return;
    }

    future.addListener(() -> {
      try {
        try {
          ActionResult result = future.get();
          responseObserver.onNext(
              packExists(
                  Operation.newBuilder()
                      .setName(opName)
                      .setDone(true)
                      .setResponse(
                          Any.pack(ExecuteResponse.newBuilder().setResult(result).build()))));
          responseObserver.onCompleted();
        } catch (ExecutionException e) {
          Throwables.throwIfUnchecked(e.getCause());
          throw (Exception) e.getCause();
        }
      } catch (Exception e) {
        logger.log(Level.SEVERE, "Work failed: " + opName, e);
        responseObserver.onNext(
            ChangeBatch.newBuilder()
                .addChanges(
                    Change.newBuilder()
                        .setState(Change.State.EXISTS)
                        .setData(
                            Any.pack(
                                Operation.newBuilder()
                                    .setName(opName)
                                    .setError(StatusUtils.internalErrorStatus(e))
                                    .build()))
                        .build())
                .build());
        responseObserver.onCompleted();
        if (e instanceof InterruptedException) {
          Thread.currentThread().interrupt();
        }
      } finally {
        operationsCache.remove(opName);
      }
    }, MoreExecutors.directExecutor());
  }

  /** Constructs a ChangeBatch with an exists state change that contains the given operation. */
  private ChangeBatch packExists(Operation.Builder message) {
    return ChangeBatch.newBuilder()
        .addChanges(
            Change.newBuilder()
                .setState(Change.State.EXISTS)
                .setData(
                    Any.pack(message.build())))
        .build();
  }
}
