package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.TracingMetadataUtils.extractRequestMetadata;

import com.google.devtools.build.lib.remote.RpcLogEntry.LogEntry;
import com.google.devtools.build.lib.remote.RpcLogEntry.LogEntry.ExecuteEntry;
import com.google.devtools.build.lib.remote.RpcLogEntry.LogEntry.WatchEntry;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc;
import com.google.longrunning.Operation;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class LoggingInterceptor implements ClientInterceptor {
  @Override
  public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
      MethodDescriptor<ReqT, RespT> method, CallOptions callOptions, Channel next) {
    if (method == ExecutionGrpc.METHOD_EXECUTE) {
      return new ExecuteLoggingForwardingCall<>(next.newCall(method, callOptions), method);
    } else if (method == WatcherGrpc.METHOD_WATCH) {
      return new WatchLoggingForwardingCall<>(next.newCall(method, callOptions), method);
    } else {
      return next.newCall(method, callOptions);
    }
  }

  public static abstract class AbstractLoggingForwardingCall<ReqT, RespT>
      extends ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT> {
    protected LogEntry.Builder entryBuilder = LogEntry.newBuilder();
    protected AbstractLoggingForwardingCall(
        ClientCall<ReqT, RespT> delegate, MethodDescriptor<ReqT, RespT> method) {
      super(delegate);
      this.entryBuilder.setMethodName(method.getFullMethodName());
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
              logResponse(message);
              super.onMessage(message);
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              logClose(status, trailers);
              entryBuilder.setStatus(makeStatusProto(status));
              try {
                entryBuilder.build().writeDelimitedTo(new FileOutputStream(new File("xd")));
                System.out.flush();
              } catch (IOException e) {
                 // ?????
              }

              super.onClose(status, trailers);
            }
          },
          headers);
    }

    @Override
    public void sendMessage(ReqT message) {
      logRequest(message);
      super.sendMessage(message);
    }

    protected abstract void logRequest(ReqT message);

    protected abstract void logResponse(RespT message);

    protected abstract void logClose(Status status, Metadata trailers);

  }

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

  static class ExecuteLoggingForwardingCall<ReqT, RespT>
      extends AbstractLoggingForwardingCall<ReqT, RespT> {
    ExecuteEntry.Builder executeEntryBuilder = ExecuteEntry.newBuilder();

    protected ExecuteLoggingForwardingCall(
        ClientCall<ReqT, RespT> delegate, MethodDescriptor<ReqT, RespT> method) {
      super(delegate, method);
    }

    @Override
    protected void logRequest(ReqT message) {
      executeEntryBuilder.setRequest((ExecuteRequest) message);
    }

    @Override
    protected void logResponse(RespT message) {
      executeEntryBuilder.setOperation((Operation) message);
    }

    @Override
    protected void logClose(Status status, Metadata trailers) {
      entryBuilder.setExecuteEntry(executeEntryBuilder);
    }
  }

  static class WatchLoggingForwardingCall<ReqT, RespT>
      extends AbstractLoggingForwardingCall<ReqT, RespT> {
    WatchEntry.Builder watchEntryBuilder = WatchEntry.newBuilder();

    protected WatchLoggingForwardingCall(
        ClientCall<ReqT, RespT> delegate, MethodDescriptor<ReqT, RespT> method) {
      super(delegate, method);
    }

    @Override
    protected void logRequest(ReqT message) {
      watchEntryBuilder.setRequest((Request) message);
    }

    @Override
    protected void logResponse(RespT message) {
      watchEntryBuilder.addChangeBatch((ChangeBatch) message);
    }

    @Override
    protected void logClose(Status status, Metadata trailers) {
      entryBuilder.setWatchEntry(watchEntryBuilder);
    }
  }

}