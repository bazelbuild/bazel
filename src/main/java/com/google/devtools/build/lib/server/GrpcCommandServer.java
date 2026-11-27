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
import com.google.devtools.build.lib.server.CommandProtos.TerminalSizeRequest;
import com.google.devtools.build.lib.server.CommandProtos.TerminalSizeResponse;
import com.google.devtools.build.lib.util.OS;
import io.grpc.Context;
import io.grpc.Server;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * The gRPC command server.
 *
 * <p>Every gRPC call is transferred to a separate thread in {@code callbackExecutorPool} so that
 * long-lived calls don't block the event loop. We do this instead of setting an executor on the
 * server object because gRPC insists on serializing calls within a single RPC call, which means
 * that the Runnable passed to {@code setOnReadyHandler} doesn't get called while the main RPC
 * method is running, which means we can't use flow control, which we need so that gRPC doesn't
 * buffer an unbounded amount of outgoing data.
 */
public class GrpcCommandServer extends CommandServerGrpc.CommandServerImplBase {

  /** The interface for responding to an RPC. */
  public interface Responder<T> {
    /**
     * Sends a response.
     *
     * <p>May be called multiple times for a streaming RPC.
     *
     * @throws IOException if an I/O error occurs
     */
    void onNext(T value) throws IOException;

    /**
     * Signals that the response is complete.
     *
     * <p>Must be called exactly once, after which {@link #onNext} must not be called again.
     *
     * @throws IOException if an I/O error occurs
     */
    void onCompleted() throws IOException;
  }

  /** The callback interface for handling RPCs. */
  public interface Callback {
    /**
     * Handles a Run RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void run(RunRequest request, Responder<RunResponse> responder);

    /**
     * Handles a Cancel RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void cancel(CancelRequest request, Responder<CancelResponse> responder);

    /**
     * Handles an UpdateTerminalSize RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void updateTerminalSize(TerminalSizeRequest request, Responder<TerminalSizeResponse> responder);

    /**
     * Handles a Ping RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void ping(PingRequest request, Responder<PingResponse> responder);
  }

  /**
   * A wrapper for {@link StreamObserver} that blocks on {@link #onNext} calls if the underlying
   * observer is not ready.
   *
   * <p>It does not react to the interrupt flag in order to allow Bazel to complete the current
   * command while printing output as well as sending the final exit code to the client. However, it
   * maintains the interrupt flag if it is already set.
   *
   * <p>When the client connection closes prematurely, the command thread is interrupted exactly
   * once, allowing it to terminate even if it produces no further output. Interrupting the thread
   * calling {@link #onNext} instead may lose the signal when {@code cli-update-thread} writes first
   * because that thread ignores interrupts. Also, interrupting on every write may repeatedly re-arm
   * the interrupt and prevent a retry of an interruptible step from ever converging, see
   * https://github.com/bazelbuild/bazel/issues/30435.
   */
  @VisibleForTesting
  static class BlockingStreamObserver<T> implements Responder<T> {
    private final ServerCallStreamObserver<T> observer;

    /**
     * Taken from the first {@link #onNext} call, which {@link CommandServer} performs before any
     * output can reach this observer.
     */
    @GuardedBy("this")
    @Nullable
    private Thread commandThread;

    @GuardedBy("this")
    private boolean commandInterruptible = true;

    BlockingStreamObserver(StreamObserver<T> observer) {
      this((ServerCallStreamObserver<T>) observer);
    }

    BlockingStreamObserver(ServerCallStreamObserver<T> observer) {
      this.observer = observer;
      this.observer.setOnReadyHandler(this::notifyWaiters);
      this.observer.setOnCancelHandler(this::notifyWaitersAndInterruptCommand);
    }

    private synchronized void notifyWaiters() {
      // This class does not restrict the number of concurrent calls to onNext, so we call notifyAll
      // here. In practice we'll usually only see one concurrent call; the ExperimentalEventHandler
      // uses synchronization to prevent multiple concurrent calls, but let's not rely on that here.
      notifyAll();
    }

    private synchronized void notifyWaitersAndInterruptCommand() {
      notifyAll(); // for the reason given in notifyWaiters
      if (commandInterruptible && commandThread != null) {
        commandThread.interrupt(); // the client went away after the first write
        commandInterruptible = false;
      }
    }

    @Override
    public synchronized void onNext(T response) throws IOException {
      if (commandThread == null) {
        commandThread = Thread.currentThread();
      }
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
      } catch (StatusRuntimeException e) {
        throw new IOException(e.getMessage(), e);
      } finally {
        // Restore the interrupt bit.
        if (interrupted) {
          Thread.currentThread().interrupt();
        }
        if (commandInterruptible && observer.isCancelled()) {
          commandThread.interrupt(); // the client was already gone at the first write
          commandInterruptible = false;
        }
      }
    }

    @Override
    public synchronized void onCompleted() throws IOException {
      commandInterruptible = false; // a late event may otherwise interrupt the pooled commandThread
      try {
        observer.onCompleted();
      } catch (StatusRuntimeException e) {
        throw new IOException(e.getMessage(), e);
      }
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

  /**
   * Starts serving.
   *
   * @param port the port to serve on (0 to pick an arbitrary unused port)
   * @param callback the callback to be called for incoming requests
   * @throws IOException if the server failed to bind to the given port
   * @return the local address the server is listening on
   */
  public SocketAddress serve(int port, Callback callback) throws IOException {
    checkState(server == null, "serve() already called");
    this.callback = callback;
    this.server = bind(port);
    return Iterables.getOnlyElement(server.getListenSockets());
  }

  /**
   * Initiates an orderly shutdown in which preexisting calls continue but new calls are rejected.
   */
  public void shutdown() {
    checkNotNull(server, "shutdown() called before serve()");
    server.shutdown();
  }

  /** Initiates a forceful shutdown in which preexisting and new calls will be rejected. */
  public void shutdownNow() {
    checkNotNull(server, "shutdownNow() called before serve()");
    server.shutdownNow();
  }

  /**
   * Waits for the server to terminate.
   *
   * @throws InterruptedException if the thread is interrupted while waiting
   */
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

  @Override
  public void updateTerminalSize(
      TerminalSizeRequest request, StreamObserver<TerminalSizeResponse> streamObserver) {
    checkNotNull(callback, "updateTerminalSize() called before serve()");
    BlockingStreamObserver<TerminalSizeResponse> blockingObserver =
        new BlockingStreamObserver<>(streamObserver);
    callbackExecutorPool.execute(() -> callback.updateTerminalSize(request, blockingObserver));
  }
}
