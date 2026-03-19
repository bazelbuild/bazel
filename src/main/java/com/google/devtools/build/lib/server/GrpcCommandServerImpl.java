// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.server;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.util.OS;
import io.grpc.Context;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.unix.Socket;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

/**
 * The {@link GrpcCommandServer} implementation.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 *
 * <p>Every gRPC call is transferred to a separate thread in {@code commandExecutorPool} so that
 * long-lived calls don't block the event loop. We do this instead of setting an executor on the
 * server object because gRPC insists on serializing calls within a single RPC call, which means
 * that the Runnable passed to {@code setOnReadyHandler} doesn't get called while the main RPC
 * method is running, which means we can't use flow control, which we need so that gRPC doesn't
 * buffer an unbounded amount of outgoing data.
 */
public class GrpcCommandServerImpl extends CommandServerGrpc.CommandServerImplBase
    implements GrpcCommandServer {

  /**
   * A wrapper for {@link StreamObserver} that blocks on {@link #onNext} calls if the underlying
   * observer is not ready.
   *
   * <p>It does not react to the interrupt flag in order to allow Bazel to complete the current
   * command while printing output as well as sending the final exit code to the client. However, it
   * maintains the interrupt flag if it is already set.
   */
  // TODO(tjgq): Avoid surfacing StatusRuntimeException (and possibly other unchecked exceptions)
  // from onNext() and onCompleted() by wrapping it into an IOException.
  @VisibleForTesting
  static class BlockingStreamObserver<T> implements GrpcCommandServer.Responder<T> {
    private final ServerCallStreamObserver<T> observer;

    BlockingStreamObserver(StreamObserver<T> observer) {
      this((ServerCallStreamObserver<T>) observer);
    }

    BlockingStreamObserver(ServerCallStreamObserver<T> observer) {
      this.observer = observer;
      this.observer.setOnReadyHandler(this::notifyWaiters);
      this.observer.setOnCancelHandler(this::notifyWaiters);
    }

    private synchronized void notifyWaiters() {
      // This class does not restrict the number of concurrent calls to onNext, so we call notifyAll
      // here. In practice we'll usually only see one concurrent call; the ExperimentalEventHandler
      // uses synchronization to prevent multiple concurrent calls, but let's not rely on that here.
      notifyAll();
    }

    @Override
    public synchronized void onNext(T response) {
      boolean interrupted = false;
      while (!observer.isReady() && !observer.isCancelled()) {
        try {
          wait();
        } catch (InterruptedException e) {
          // We intentionally do not break or return here. The interrupt signal can be due the user
          // pressing ctrl-c: it can take Bazel a while to shut down (e.g., it is not currently
          // possible to interrupt persistent workers), and we must allow it to continue printing
          // output until the current operation comes to a finish.
          interrupted = true;
        }
      }
      try {
        // According to the documentation, if onNext is called in a canceled stream, it will be
        // silently ignored.
        observer.onNext(response);
      } finally {
        // onNext does not specify whether it can throw unchecked exceptions. We use a finally block
        // here to make sure that the interrupt bit is not lost even if it does.
        if (interrupted || observer.isCancelled()) {
          Thread.currentThread().interrupt();
        }
      }
    }

    @Override
    public void onCompleted() {
      observer.onCompleted();
    }
  }

  private final Executor callbackExecutorPool =
      Context.currentContextExecutor(
          Executors.newCachedThreadPool(
              new ThreadFactoryBuilder().setNameFormat("grpc-command-%d").setDaemon(true).build()));

  @Nullable private Server server = null;
  @Nullable private Callback callback = null;

  private Server bindWithRetries(InetSocketAddress address, int maxRetries) throws IOException {
    Server server = null;
    for (int attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        server =
            NettyServerBuilder.forAddress(address)
                .addService(this)
                .directExecutor()
                .build()
                .start();
        break;
      } catch (IOException | RuntimeException e) {
        // NettyServerBuilder.build() can throw a RuntimeException on epoll failures.
        if (attempt == maxRetries) {
          throw e;
        }
      }
    }
    return server;
  }

  @SuppressWarnings("AddressSelection") // intentional use of [::1] and 127.0.0.1
  protected Server bind(int port) throws IOException {
    // For reasons only Apple knows, you cannot bind to IPv4-localhost when you run in a sandbox
    // that only allows loopback traffic, but binding to IPv6-localhost works fine. This would
    // however break on systems that don't support IPv6. So what we'll do is to try to bind to IPv6
    // and if that fails, try again with IPv4.
    InetSocketAddress address = new InetSocketAddress("[::1]", port);
    try {
      // TODO(bazel-team): Remove the following check after upgrading netty to a version with a fix
      //   for https://github.com/netty/netty/issues/10402
      if (Epoll.isAvailable() && !Socket.isIPv6Preferred()) {
        throw new IOException("ipv6 is not preferred on the system.");
      }
      // For some strange reasons, Bazel server sometimes fails to bind to IPv6 localhost when
      // running in macOS sandbox-exec with internet blocked. Retrying seems to help.
      // See https://github.com/bazelbuild/bazel/issues/20743
      return bindWithRetries(address, OS.getCurrent() == OS.DARWIN ? 3 : 1);
    } catch (IOException ipv6Exception) {
      address = new InetSocketAddress("127.0.0.1", port);
      try {
        return NettyServerBuilder.forAddress(address)
            .addService(this)
            .directExecutor()
            .build()
            .start();
      } catch (IOException | RuntimeException ipv4Exception) {
        // NettyServerBuilder.build() can throw a RuntimeException on epoll failures.
        throw new IOException(
            "gRPC server failed to bind to localhost on port %d:\n[IPv4] %s\n[IPv6] %s"
                .formatted(port, ipv4Exception.getMessage(), ipv6Exception.getMessage()));
      }
    }
  }

  @Override
  public SocketAddress serve(int port, GrpcCommandServer.Callback callback) throws IOException {
    checkState(server == null, "serve() already called");
    this.callback = callback;
    this.server = bind(port);
    return Iterables.getOnlyElement(server.getListenSockets());
  }

  @Override
  public void shutdown() {
    checkNotNull(server, "shutdown() called before serve()");
    if (server != null) {
      server.shutdown();
    }
  }

  @Override
  public void shutdownNow() {
    checkNotNull(server, "shutdownNow() called before serve()");
    if (server != null) {
      server.shutdownNow();
    }
  }

  @Override
  public void awaitTermination() throws InterruptedException {
    checkNotNull(server, "awaitTermination() called before serve()");
    server.awaitTermination();
  }

  @Override
  public void run(RunRequest request, StreamObserver<RunResponse> streamObserver) {
    checkNotNull(callback, "run() called before serve()");
    BlockingStreamObserver<RunResponse> blockingObserver =
        new BlockingStreamObserver<>(streamObserver);
    callbackExecutorPool.execute(() -> callback.run(request, blockingObserver));
  }

  @Override
  public void ping(PingRequest pingRequest, StreamObserver<PingResponse> streamObserver) {
    checkNotNull(callback, "ping() called before serve()");
    BlockingStreamObserver<PingResponse> blockingObserver =
        new BlockingStreamObserver<>(streamObserver);
    callbackExecutorPool.execute(() -> callback.ping(pingRequest, blockingObserver));
  }

  @Override
  public void cancel(CancelRequest cancelRequest, StreamObserver<CancelResponse> streamObserver) {
    checkNotNull(callback, "cancel() called before serve()");
    BlockingStreamObserver<CancelResponse> blockingObserver =
        new BlockingStreamObserver<>(streamObserver);
    callbackExecutorPool.execute(() -> callback.cancel(cancelRequest, blockingObserver));
  }
}
