// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.logging;

import build.bazel.remote.execution.v2.ActionCacheGrpc;
import build.bazel.remote.execution.v2.CapabilitiesGrpc;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import build.bazel.remote.execution.v2.ExecutionGrpc;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.bytestream.ByteStreamGrpc;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.LogEntry;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.protobuf.Timestamp;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.time.Instant;
import javax.annotation.Nullable;

/** Client interceptor for logging details of certain gRPC calls. */
public class LoggingInterceptor implements ClientInterceptor {
  private final AsynchronousFileOutputStream rpcLogFile;
  private final Clock clock;

  /** Constructs a LoggingInterceptor which logs RPC calls to the given file. */
  public LoggingInterceptor(AsynchronousFileOutputStream rpcLogFile, Clock clock) {
    this.rpcLogFile = rpcLogFile;
    this.clock = clock;
  }

  /**
   * Returns a {@link LoggingHandler} to handle logging details for the specified method. If there
   * is no handler for the given method, returns {@code null}.
   *
   * @param method Method to return handler for.
   */
  @SuppressWarnings("rawtypes")
  protected <ReqT, RespT> @Nullable LoggingHandler selectHandler(
      MethodDescriptor<ReqT, RespT> method) {
    if (method == ExecutionGrpc.getExecuteMethod()) {
      return new ExecuteHandler();
    } else if (method == ExecutionGrpc.getWaitExecutionMethod()) {
      return new WaitExecutionHandler();
    } else if (method == ActionCacheGrpc.getGetActionResultMethod()) {
      return new GetActionResultHandler();
    } else if (method == ActionCacheGrpc.getUpdateActionResultMethod()) {
      return new UpdateActionResultHandler();
    } else if (method == ContentAddressableStorageGrpc.getFindMissingBlobsMethod()) {
      return new FindMissingBlobsHandler();
    } else if (method == ByteStreamGrpc.getReadMethod()) {
      return new ReadHandler();
    } else if (method == ByteStreamGrpc.getWriteMethod()) {
      return new WriteHandler();
    } else if (method == CapabilitiesGrpc.getGetCapabilitiesMethod()) {
      return new GetCapabilitiesHandler();
    }
    return null;
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    ClientCall<ReqT, RespT> call = next.newCall(method, callOptions);
    LoggingHandler<ReqT, RespT> handler = selectHandler(method);
    if (handler != null) {
      return new LoggingForwardingCall<>(call, handler, method);
    } else {
      return call;
    }
  }

  /** Get current time as a Timestamp. */
  private Timestamp getCurrentTimestamp() {
    Instant time = Instant.ofEpochMilli(clock.currentTimeMillis());
    return Timestamp.newBuilder()
        .setSeconds(time.getEpochSecond())
        .setNanos(time.getNano())
        .build();
  }

  /**
   * Wraps client call to log call details by building a {@link LogEntry} and writing it to the RPC
   * log file.
   */
  private class LoggingForwardingCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    private final LoggingHandler<ReqT, RespT> handler;
    private final LogEntry.Builder entryBuilder;

    protected LoggingForwardingCall(
        ClientCall<ReqT, RespT> delegate,
        LoggingHandler<ReqT, RespT> handler,
        MethodDescriptor<ReqT, RespT> method) {
      super(delegate);
      this.handler = handler;
      this.entryBuilder = LogEntry.newBuilder().setMethodName(method.getFullMethodName());
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      entryBuilder.setStartTime(getCurrentTimestamp());
      RequestMetadata metadata = TracingMetadataUtils.requestMetadataFromHeaders(headers);
      if (metadata != null) {
        entryBuilder.setMetadata(metadata);
      }
      super.start(
          new ForwardingClientCallListener.SimpleForwardingClientCallListener<RespT>(
              responseListener) {
            @Override
            public void onMessage(RespT message) {
              handler.handleResp(message);
              super.onMessage(message);
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              entryBuilder.setEndTime(getCurrentTimestamp());
              entryBuilder.setStatus(makeStatusProto(status));
              entryBuilder.setDetails(handler.getDetails());
              rpcLogFile.write(entryBuilder.build());
              super.onClose(status, trailers);
            }
          },
          headers);
    }

    @Override
    public void sendMessage(ReqT message) {
      handler.handleReq(message);
      super.sendMessage(message);
    }
  }

  /** Converts io.grpc.Status to com.google.rpc.Status proto for logging. */
  private static com.google.rpc.Status makeStatusProto(Status status) {
    String message = "";
    if (status.getCause() != null) {
      message = status.getCause().toString();
    } else if (status.getDescription() != null) {
      message = status.getDescription();
    }
    return com.google.rpc.Status.newBuilder()
        .setCode(status.getCode().value())
        .setMessage(message)
        .build();
  }
}
