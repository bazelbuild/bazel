package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.TracingMetadataUtils.extractRequestMetadata;

import com.google.devtools.build.lib.remote.logging.RpcLogEntry.LogEntry;
import com.google.devtools.build.lib.remote.logging.ExecuteHandler;
import com.google.devtools.build.lib.remote.logging.LoggingHandler;
import com.google.devtools.build.lib.remote.logging.WatchHandler;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc;
import com.google.watcher.v1.WatcherGrpc;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Client interceptor for logging details of certain gRPC calls.
 */
public class LoggingInterceptor implements ClientInterceptor {
  RpcLogger rpcLogger;

  /**
   * Creates a interceptor with this logger.
   * @param rpcLogger
   */
  public LoggingInterceptor(RpcLogger rpcLogger) {
    this.rpcLogger = rpcLogger;
  }

  /**
   * Returns a {@link LoggingHandler} to handle logging details for the specific method.
   * @param method Method to return handler for.
   * @return A handler for the given method. Returns null if the method has no specific handler.
   */
  protected @Nullable LoggingHandler selectHandler(MethodDescriptor method) {
    if (method == ExecutionGrpc.METHOD_EXECUTE) {
      return new ExecuteHandler();
    } else if (method == WatcherGrpc.METHOD_WATCH) {
      return new WatchHandler();
    } else {
      return null;
    }
  }

  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    LoggingHandler<ReqT, RespT> handler = selectHandler(method);
    if (handler != null) {
      return new LoggingForwardingCall<>(
          next.newCall(method, callOptions), handler, rpcLogger, method);
    } else {
      return next.newCall(method, callOptions);
    }
  }

  /**
   * Wraps client call to log call details by building a {@link LogEntry} and writing it with the
   * given {@link RpcLogger}.
   */
  private static class LoggingForwardingCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    private LoggingHandler<ReqT, RespT> handler;
    private RpcLogger rpcLogger;
    private LogEntry.Builder entryBuilder = LogEntry.newBuilder();
    protected LoggingForwardingCall(
        ClientCall<ReqT, RespT> delegate, LoggingHandler<ReqT, RespT> handler,
        RpcLogger rpcLogger, MethodDescriptor<ReqT, RespT> method) {
      super(delegate);
      this.handler = handler;
      this.rpcLogger = rpcLogger;
      entryBuilder.setMethodName(method.getFullMethodName());
    }

    @Override
    public void start(Listener<RespT> responseListener, Metadata headers) {
      try {
        entryBuilder.setMetadata(extractRequestMetadata(headers));
      } catch (IllegalStateException e) {
        // Don't set RequestMetadata field if it is not present.
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
              entryBuilder.setStatus(makeStatusProto(status));
              try {
                rpcLogger.writeEntry(entryBuilder.build());
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
              LogEntry entry = entryBuilder.mergeFrom(handler.getEntry()).build();
              // TODO(cdlee): Actually log the entry.
              System.out.println(entryBuilder.build());
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

  // Converts grpc.io.Status to com.google.rpc.Status for logging.
  private static com.google.rpc.Status makeStatusProto(Status status) {
    String message = "";
    if (status.getCause() != null) {
      message = status.getCause().toString();
    } else if (status.getDescription() != null){
      message = status.getDescription();
    }
    return com.google.rpc.Status.newBuilder()
        .setCode(status.getCode().value())
        .setMessage(message)
        .build();
  }
}