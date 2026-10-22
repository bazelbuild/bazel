// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.rpc.Code;
import com.google.rpc.PreconditionFailure;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An in-process sandbox backend: stores what is written to it by digest, serves what the test
 * makes {@link #serve readable}, and lets the test script {@code Execute}.
 */
final class FakeBackend implements AutoCloseable {
  /** What the backend does when it is asked to execute an action. */
  @FunctionalInterface
  interface ExecuteHandler {
    void execute(ExecuteRequest request, StreamObserver<Operation> responses) throws Exception;
  }

  final Map<String, ByteString> stored = new ConcurrentHashMap<>();
  final List<String> writes = Collections.synchronizedList(new ArrayList<>());
  final List<ExecuteRequest> executes = Collections.synchronizedList(new ArrayList<>());
  final AtomicInteger findMissingCalls = new AtomicInteger();
  private final Map<String, ByteString> readable = new ConcurrentHashMap<>();
  private volatile ExecuteHandler handler = (request, responses) -> {};
  private final String name = "fake-backend-" + UUID.randomUUID();
  private final Server server;

  FakeBackend() throws IOException {
    server =
        InProcessServerBuilder.forName(name)
            .addService(new Bytestream())
            .addService(new Cas())
            .addService(new Exec())
            .build()
            .start();
  }

  ManagedChannel newChannel() {
    return InProcessChannelBuilder.forName(name).build();
  }

  void setExecuteHandler(ExecuteHandler handler) {
    this.handler = handler;
  }

  /** Makes {@code data} what is returned when the blob with {@code digest} is read. */
  void serve(Digest digest, ByteString data) {
    readable.put(digest.getHash(), data);
  }

  /** Forgets a blob, as if it was evicted. */
  void evict(String hash) {
    stored.remove(hash);
  }

  void killServer() {
    server.shutdownNow();
  }

  static void respond(StreamObserver<Operation> responses, ActionResult result) {
    responses.onNext(
        Operation.newBuilder()
            .setDone(true)
            .setResponse(Any.pack(ExecuteResponse.newBuilder().setResult(result).build()))
            .build());
    responses.onCompleted();
  }

  static void respond(StreamObserver<Operation> responses, int code, String message) {
    responses.onNext(
        Operation.newBuilder()
            .setDone(true)
            .setResponse(
                Any.pack(
                    ExecuteResponse.newBuilder()
                        .setStatus(
                            com.google.rpc.Status.newBuilder()
                                .setCode(code)
                                .setMessage(message))
                        .build()))
            .build());
    responses.onCompleted();
  }

  static void respondMissing(StreamObserver<Operation> responses) {
    responses.onNext(
        Operation.newBuilder()
            .setDone(true)
            .setResponse(
                Any.pack(
                    ExecuteResponse.newBuilder()
                        .setStatus(
                            com.google.rpc.Status.newBuilder()
                                .setCode(Code.FAILED_PRECONDITION_VALUE)
                                .addDetails(
                                    Any.pack(
                                        PreconditionFailure.newBuilder()
                                            .addViolations(
                                                PreconditionFailure.Violation.newBuilder()
                                                    .setType("MISSING"))
                                            .build())))
                        .build()))
            .build());
    responses.onCompleted();
  }

  @Override
  public void close() {
    server.shutdownNow();
  }

  private static String hashOf(String resourceName, int fromEnd) {
    String[] parts = resourceName.split("/");
    return parts[parts.length - fromEnd];
  }

  private final class Bytestream extends ByteStreamImplBase {
    @Override
    public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> responseObserver) {
      return new StreamObserver<>() {
        final ByteArrayOutputStream data = new ByteArrayOutputStream();
        String hash;

        @Override
        public void onNext(WriteRequest request) {
          if (!request.getResourceName().isEmpty()) {
            hash = hashOf(request.getResourceName(), 2);
          }
          data.writeBytes(request.getData().toByteArray());
        }

        @Override
        public void onError(Throwable t) {}

        @Override
        public void onCompleted() {
          stored.put(hash, ByteString.copyFrom(data.toByteArray()));
          writes.add(hash);
          // The number of bytes received, which differs from the size of the blob for a file URI.
          responseObserver.onNext(WriteResponse.newBuilder().setCommittedSize(data.size()).build());
          responseObserver.onCompleted();
        }
      };
    }

    @Override
    public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
      ByteString data = readable.get(hashOf(request.getResourceName(), 2));
      if (data == null) {
        responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
        return;
      }
      responseObserver.onNext(ReadResponse.newBuilder().setData(data).build());
      responseObserver.onCompleted();
    }
  }

  private final class Cas extends ContentAddressableStorageImplBase {
    @Override
    public void findMissingBlobs(
        FindMissingBlobsRequest request, StreamObserver<FindMissingBlobsResponse> observer) {
      findMissingCalls.incrementAndGet();
      FindMissingBlobsResponse.Builder response = FindMissingBlobsResponse.newBuilder();
      for (Digest digest : request.getBlobDigestsList()) {
        if (!stored.containsKey(digest.getHash())) {
          response.addMissingBlobDigests(digest);
        }
      }
      observer.onNext(response.build());
      observer.onCompleted();
    }
  }

  private final class Exec extends ExecutionImplBase {
    @Override
    public void execute(ExecuteRequest request, StreamObserver<Operation> responses) {
      executes.add(request);
      try {
        handler.execute(request, responses);
      } catch (Exception e) {
        responses.onError(Status.INTERNAL.withCause(e).asRuntimeException());
      }
    }
  }
}
