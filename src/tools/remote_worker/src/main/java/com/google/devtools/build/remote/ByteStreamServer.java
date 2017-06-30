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

import static java.util.logging.Level.WARNING;

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.devtools.build.lib.remote.CacheNotFoundException;
import com.google.devtools.build.lib.remote.Chunker;
import com.google.devtools.build.lib.remote.Digests;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import com.google.rpc.Code;
import com.google.rpc.Status;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/** A basic implementation of a {@link ByteStreamImplBase} service. */
final class ByteStreamServer extends ByteStreamImplBase {
  private static final Logger logger = Logger.getLogger(ByteStreamServer.class.getName());
  private final SimpleBlobStoreActionCache cache;

  static @Nullable Digest parseDigestFromResourceName(String resourceName) {
    try {
      String[] tokens = resourceName.split("/");
      if (tokens.length < 2) {
        return null;
      }
      String hash = tokens[tokens.length - 2];
      long size = Long.parseLong(tokens[tokens.length - 1]);
      return Digests.buildDigest(hash, size);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  public ByteStreamServer(SimpleBlobStoreActionCache cache) {
    this.cache = cache;
  }

  @Override
  public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
    Digest digest = parseDigestFromResourceName(request.getResourceName());

    if (digest == null) {
      responseObserver.onError(
          StatusUtils.invalidArgumentError(
              "resource_name",
              "Failed parsing digest from resource_name:" + request.getResourceName()));
    }

    if (!cache.containsKey(digest)) {
      responseObserver.onError(StatusUtils.notFoundError(digest));
      return;
    }

    try {
      // This still relies on the blob size to be small enough to fit in memory.
      // TODO(olaola): refactor to fix this if the need arises.
      Chunker c = new Chunker.Builder().addInput(cache.downloadBlob(digest)).build();
      while (c.hasNext()) {
        responseObserver.onNext(
            ReadResponse.newBuilder().setData(ByteString.copyFrom(c.next().getData())).build());
      }
      responseObserver.onCompleted();
    } catch (CacheNotFoundException e) {
      // This can only happen if an item gets evicted right after we check.
      responseObserver.onError(StatusUtils.notFoundError(digest));
    } catch (Exception e) {
      logger.log(WARNING, "Read request failed.", e);
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }

  @Override
  public StreamObserver<WriteRequest> write(final StreamObserver<WriteResponse> responseObserver) {
    return new StreamObserver<WriteRequest>() {
      byte[] blob;
      Digest digest;
      long offset;
      String resourceName;
      boolean closed;

      @Override
      public void onNext(WriteRequest request) {
        if (closed) {
          return;
        }

        if (digest == null) {
          resourceName = request.getResourceName();
          digest = parseDigestFromResourceName(resourceName);
          blob = new byte[(int) digest.getSizeBytes()];
        }

        if (digest == null) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "resource_name",
                  "Failed parsing digest from resource_name: " + request.getResourceName()));
          closed = true;
          return;
        }

        if (request.getWriteOffset() != offset) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "write_offset",
                  "Expected: " + offset + ", received: " + request.getWriteOffset()));
          closed = true;
          return;
        }

        if (!request.getResourceName().isEmpty()
            && !request.getResourceName().equals(resourceName)) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "resource_name",
                  "Expected: " + resourceName + ", received: " + request.getResourceName()));
          closed = true;
          return;
        }

        long size = request.getData().size();

        if (size > 0) {
          request.getData().copyTo(blob, (int) offset);
          offset += size;
        }

        boolean shouldFinishWrite = offset == digest.getSizeBytes();

        if (shouldFinishWrite != request.getFinishWrite()) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "finish_write",
                  "Expected:" + shouldFinishWrite + ", received: " + request.getFinishWrite()));
          closed = true;
        }
      }

      @Override
      public void onError(Throwable t) {
        logger.log(WARNING, "Write request failed remotely.", t);
        closed = true;
      }

      @Override
      public void onCompleted() {
        if (closed) {
          return;
        }

        if (digest == null || offset != digest.getSizeBytes()) {
          responseObserver.onError(
              StatusProto.toStatusRuntimeException(
                  Status.newBuilder()
                      .setCode(Code.FAILED_PRECONDITION.getNumber())
                      .setMessage("Request completed before all data was sent.")
                      .build()));
          closed = true;
          return;
        }

        try {
          Digest d = cache.uploadBlob(blob);

          if (!d.equals(digest)) {
            responseObserver.onError(
                StatusUtils.invalidArgumentError(
                    "resource_name",
                    "Received digest " + digest + " does not match computed digest " + d));
            closed = true;
            return;
          }

          responseObserver.onNext(WriteResponse.newBuilder().setCommittedSize(offset).build());
          responseObserver.onCompleted();
        } catch (Exception e) {
          logger.log(WARNING, "Write request failed.", e);
          responseObserver.onError(StatusUtils.internalError(e));
          closed = true;
        }
      }
    };
  }
}
