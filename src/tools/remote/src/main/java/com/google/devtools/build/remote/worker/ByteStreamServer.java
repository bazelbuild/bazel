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

package com.google.devtools.build.remote.worker;

import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.remote.Chunker;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.remote.worker.LiveOutputStreams.LiveOutput;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.UUID;
import javax.annotation.Nullable;

/** A basic implementation of a {@link ByteStreamImplBase} service. */
final class ByteStreamServer extends ByteStreamImplBase {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Largest chunk of live output sent in a single response. */
  private static final int LIVE_OUTPUT_CHUNK_SIZE = 64 * 1024;

  /** How long a live output read waits for new output before checking for cancellation. */
  private static final long LIVE_OUTPUT_POLL_MILLIS = 500;

  private final OnDiskBlobStoreCache cache;
  private final Path workPath;
  private final DigestUtil digestUtil;
  @Nullable private final LiveOutputStreams liveOutputStreams;

  @Nullable
  static Digest parseDigestFromResourceName(String resourceName) {
    try {
      String[] tokens = resourceName.split("/");
      if (tokens.length < 2) {
        return null;
      }
      String hash = tokens[tokens.length - 2];
      long size = Long.parseLong(tokens[tokens.length - 1]);
      return DigestUtil.buildDigest(hash, size);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  public ByteStreamServer(
      OnDiskBlobStoreCache cache,
      Path workPath,
      DigestUtil digestUtil,
      @Nullable LiveOutputStreams liveOutputStreams) {
    this.cache = cache;
    this.workPath = workPath;
    this.digestUtil = digestUtil;
    this.liveOutputStreams = liveOutputStreams;
  }

  @Override
  public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
    if (liveOutputStreams != null) {
      LiveOutput liveOutput = liveOutputStreams.get(request.getResourceName());
      if (liveOutput != null) {
        readLiveOutput(liveOutput, request, responseObserver);
        return;
      }
    }

    RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(meta);
    Digest digest = parseDigestFromResourceName(request.getResourceName());

    if (digest == null) {
      responseObserver.onError(
          StatusUtils.invalidArgumentError(
              "resource_name",
              "Failed parsing digest from resource_name:" + request.getResourceName()));
      return;
    }

    try {
      // This still relies on the blob size to be small enough to fit in memory.
      // TODO(olaola): refactor to fix this if the need arises.
      byte[] bytes = getFromFuture(cache.downloadBlob(context, digest));
      long readOffset = request.getReadOffset();
      long readLimit = request.getReadLimit();
      if (readOffset < 0 || readOffset > bytes.length) {
        responseObserver.onError(
            StatusUtils.outOfRangeError(
                "read_offset",
                "Expected a value in [0, %d], received: %d".formatted(bytes.length, readOffset)));
        return;
      }
      if (readLimit < 0) {
        responseObserver.onError(
            StatusUtils.invalidArgumentError(
                "read_limit", "Expected a non-negative value, received: %d".formatted(readLimit)));
        return;
      }
      int offset = (int) readOffset;
      int length =
          readLimit > 0
              ? (int) Math.min(bytes.length - readOffset, readLimit)
              : bytes.length - offset;
      try (Chunker c =
          Chunker.builder()
              .setInput(length, () -> new ByteArrayInputStream(bytes, offset, length))
              .build()) {
        while (c.hasNext()) {
          responseObserver.onNext(ReadResponse.newBuilder().setData(c.next().getData()).build());
        }
      }
      responseObserver.onCompleted();
    } catch (CacheNotFoundException e) {
      responseObserver.onError(StatusUtils.notFoundError(digest));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      logger.atWarning().withCause(e).log("Read request interrupted");
      responseObserver.onError(StatusUtils.interruptedError(digest));
    } catch (Exception e) {
      logger.atWarning().withCause(e).log("Read request failed");
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }

  /** Sends the output of an executing action as it is produced, until the action finishes. */
  private static void readLiveOutput(
      LiveOutput liveOutput, ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
    long offset = request.getReadOffset();
    long readLimit = request.getReadLimit();
    if (offset < 0) {
      responseObserver.onError(
          StatusUtils.outOfRangeError(
              "read_offset", "Expected a non-negative value, received: %d".formatted(offset)));
      return;
    }
    if (readLimit < 0) {
      responseObserver.onError(
          StatusUtils.invalidArgumentError(
              "read_limit", "Expected a non-negative value, received: %d".formatted(readLimit)));
      return;
    }
    long end = readLimit > 0 ? offset + readLimit : Long.MAX_VALUE;
    ServerCallStreamObserver<ReadResponse> serverObserver =
        (ServerCallStreamObserver<ReadResponse>) responseObserver;
    try {
      while (offset < end && !serverObserver.isCancelled()) {
        int maxBytes = (int) Math.min(LIVE_OUTPUT_CHUNK_SIZE, end - offset);
        ByteString data = liveOutput.read(offset, maxBytes, LIVE_OUTPUT_POLL_MILLIS);
        if (data == null) {
          break;
        }
        if (!data.isEmpty()) {
          responseObserver.onNext(ReadResponse.newBuilder().setData(data).build());
          offset += data.size();
        }
      }
      if (!serverObserver.isCancelled()) {
        responseObserver.onCompleted();
      }
    } catch (IllegalArgumentException e) {
      responseObserver.onError(StatusUtils.outOfRangeError("read_offset", e.getMessage()));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      responseObserver.onError(
          Status.CANCELLED.withDescription("live output read interrupted").asException());
    }
  }

  @Override
  public StreamObserver<WriteRequest> write(final StreamObserver<WriteResponse> responseObserver) {
    RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(meta);

    Path temp = workPath.getRelative("upload").getRelative(UUID.randomUUID().toString());
    try {
      temp.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(temp);
    } catch (IOException e) {
      logger.atSevere().withCause(e).log("Failed to create temporary file for upload");
      responseObserver.onError(StatusUtils.internalError(e));
      // We need to make sure that subsequent onNext or onCompleted calls don't make any further
      // calls on the responseObserver after the onError above, so we return a no-op observer.
      return new NoOpStreamObserver<>();
    }
    return new StreamObserver<WriteRequest>() {
      private Digest digest;
      private long offset;
      private String resourceName;
      private boolean closed;

      @Override
      public void onNext(WriteRequest request) {
        if (closed) {
          return;
        }

        if (digest == null) {
          resourceName = request.getResourceName();
          digest = parseDigestFromResourceName(resourceName);
        }

        if (digest == null) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "resource_name",
                  "Failed parsing digest from resource_name: " + request.getResourceName()));
          closed = true;
          return;
        }

        if (offset == 0) {
          boolean exists = false;
          try {
            exists = cache.refresh(digest);
          } catch (IOException e) {
            responseObserver.onError(StatusUtils.internalError(e));
            closed = true;
            return;
          }
          if (exists) {
            responseObserver.onNext(
                WriteResponse.newBuilder().setCommittedSize(digest.getSizeBytes()).build());
            responseObserver.onCompleted();
            closed = true;
            return;
          }
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
          try (OutputStream out = temp.getOutputStream(true)) {
            request.getData().writeTo(out);
          } catch (IOException e) {
            responseObserver.onError(StatusUtils.internalError(e));
            closed = true;
            return;
          }
          offset += size;
        }

        boolean shouldFinishWrite = offset == digest.getSizeBytes();

        if (shouldFinishWrite != request.getFinishWrite()) {
          responseObserver.onError(
              StatusUtils.invalidArgumentError(
                  "finish_write",
                  "Expected:" + shouldFinishWrite + ", received: " + request.getFinishWrite()));
          closed = true;
          return;
        }
      }

      @Override
      public void onError(Throwable t) {
        if (Status.fromThrowable(t).getCode() != Status.Code.CANCELLED) {
          logger.atWarning().withCause(t).log("Write request failed remotely");
        }
        closed = true;
        try {
          temp.delete();
        } catch (IOException e) {
          logger.atWarning().withCause(e).log("Could not delete temp file");
        }
      }

      @Override
      public void onCompleted() {
        if (closed) {
          return;
        }

        if (digest == null || offset != digest.getSizeBytes()) {
          responseObserver.onError(
              StatusProto.toStatusRuntimeException(
                  com.google.rpc.Status.newBuilder()
                      .setCode(Status.Code.FAILED_PRECONDITION.value())
                      .setMessage("Request completed before all data was sent.")
                      .build()));
          closed = true;
          return;
        }

        try {
          Digest d = digestUtil.compute(temp, temp.stat());
          getFromFuture(cache.uploadFile(context, d, temp));
          try {
            temp.delete();
          } catch (IOException e) {
            logger.atWarning().withCause(e).log("Could not delete temp file");
          }

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
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          logger.atWarning().withCause(e).log("Write request interrupted");
          responseObserver.onError(StatusUtils.interruptedError(digest));
          closed = true;
        } catch (Exception e) {
          logger.atWarning().withCause(e).log("Write request failed");
          responseObserver.onError(StatusUtils.internalError(e));
          closed = true;
        }
      }
    };
  }

  private static class NoOpStreamObserver<T> implements StreamObserver<T> {
    @Override
    public void onNext(T value) {}

    @Override
    public void onError(Throwable t) {}

    @Override
    public void onCompleted() {}
  }
}
