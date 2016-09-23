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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.net.InetAddresses;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ThreadUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.Charset;
import java.security.SecureRandom;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Exchanger;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import javax.annotation.concurrent.GuardedBy;

/**
 * gRPC server class.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 *
 * <p>This class is a little complicated and rich in multithreading, so an explanation of its
 * innards follows.
 *
 * <p>We use the direct executor for gRPC so that it calls our methods directly on its event handler
 * threads (which it creates itself). This is acceptable for {@code ping()} and {@code cancel()}
 * because they run very quickly. For {@code run()}, we transfer the call to our own threads in
 * {@code commandExecutorPool}. We do this instead of setting an executor on the server object
 * because gRPC insists on serializing calls within a single RPC call, which means that the Runnable
 * passed to {@code setOnReadyHandler} doesn't get called while the main RPC method is running,
 * which means we can't use flow control, which we need so that gRPC doesn't buffer an unbounded
 * amount of outgoing data.
 *
 * <p>Two threads are spawned for each command: one that handles the command in {@code
 * commandExecutorPool} and one that streams the result back to the client in {@code
 * streamExecutorPool}.
 *
 * <p>In addition to these threads, we maintain one extra thread for handling the server timeout and
 * an interrupt watcher thread is started for each interrupt request that logs if it takes too long
 * to take effect.
 *
 * <p>Each running RPC has a UUID associated with it that is used to identify it when a client wants
 * to cancel it. Cancellation is done by the client sending the server a {@code cancel()} RPC call
 * which results in the main thread of the command being interrupted.
 */
public class GrpcServerImpl extends RPCServer {
  private static final Logger log = Logger.getLogger(GrpcServerImpl.class.getName());

  // UTF-8 won't do because we want to be able to pass arbitrary binary strings.
  // Not that the internals of Bazel handle that correctly, but why not make at least this little
  // part correct?
  private static final Charset CHARSET = Charset.forName("ISO-8859-1");

  private static final long NANOSECONDS_IN_MS = TimeUnit.MILLISECONDS.toNanos(1);

  private class RunningCommand implements AutoCloseable {
    private final Thread thread;
    private final String id;

    private RunningCommand() {
      thread = Thread.currentThread();
      id = UUID.randomUUID().toString();
      synchronized (runningCommands) {
        runningCommands.put(id, this);
        runningCommands.notify();
      }

      log.info(String.format("Starting command %s on thread %s", id, thread.getName()));
    }

    @Override
    public void close() {
      synchronized (runningCommands) {
        runningCommands.remove(id);
        runningCommands.notify();
      }

      log.info(String.format("Finished command %s on thread %s", id, thread.getName()));
    }
  }

  /**
   * Factory class. Instantiated by reflection.
   *
   * <p>Used so that method calls using reflection are as simple as possible.
   */
  public static class Factory implements RPCServer.Factory {
    @Override
    public RPCServer create(CommandExecutor commandExecutor, Clock clock, int port,
      Path serverDirectory, int maxIdleSeconds) throws IOException {
      return new GrpcServerImpl(commandExecutor, clock, port, serverDirectory, maxIdleSeconds);
    }
  }

  @VisibleForTesting
  enum StreamType {
    STDOUT,
    STDERR,
  }

  /** Actions {@link GrpcSink} can do. */
  private enum SinkThreadAction {
    DISCONNECT,
    FINISH,
    READY,
    SEND,
  }

  /**
   * Sent back and forth between threads wanting to write to the client stream and the stream
   * handler thread.
   */
  @Immutable
  private static final class SinkThreadItem {
    private final boolean success;
    private final RunResponse message;

    private SinkThreadItem(boolean success, RunResponse message) {
      this.success = success;
      this.message = message;
    }
  }

  /**
   * A class that handles communicating through a gRPC interface.
   *
   * <p>It can do four things:
   * <li>Send a response message over the wire. If the channel is ready, it's sent immediately, if
   *     it's not, blocks until it is. Note that there can always be only one thread in {@link
   *     #offer(RunResponse)} because it's synchronized. This results in the associated streams
   *     blocking if gRPC is not ready, which is how we implement pushback.
   * <li>Be notified that gRPC is ready. If there is a pending message, it is then sent.
   * <li>Be notified that the client disconnected. In this case, an {@link IOException} is reported
   *     and the thread from which the stream was written to is interrupted so that the server
   *     becomes free as soon as possible.
   * <li>Processing can be terminated. It is reported whether the client disconnected before.
   */
  @VisibleForTesting
  static class GrpcSink {
    private final LinkedBlockingQueue<SinkThreadAction> actionQueue;
    private final Exchanger<SinkThreadItem> exchanger;
    private final ServerCallStreamObserver<RunResponse> observer;
    private final Future<?> future;
    private final AtomicReference<Thread> commandThread = new AtomicReference<>();
    private final AtomicBoolean disconnected = new AtomicBoolean(false);

    @VisibleForTesting
    GrpcSink(ServerCallStreamObserver<RunResponse> observer, ExecutorService executor) {
      // This queue is intentionally unbounded: we always act on it fairly quickly so filling up
      // RAM is not a concern but we don't want to block in the gRPC cancel/onready handlers.
      this.actionQueue = new LinkedBlockingQueue<>();
      this.exchanger = new Exchanger<>();
      this.observer = observer;
      this.observer.setOnCancelHandler(
          new Runnable() {
            @Override
            public void run() {
              Thread commandThread = GrpcSink.this.commandThread.get();
              if (commandThread != null) {
                log.info(
                    String.format(
                        "Interrupting thread %s due to gRPC cancel", commandThread.getName()));
                commandThread.interrupt();
              }

              actionQueue.offer(SinkThreadAction.DISCONNECT);
            }
          });
      this.observer.setOnReadyHandler(
          new Runnable() {
            @Override
            public void run() {
              actionQueue.offer(SinkThreadAction.READY);
            }
          });

      this.future =
          executor.submit(
              new Runnable() {
                @Override
                public void run() {
                  GrpcSink.this.call();
                }
              });
    }

    @VisibleForTesting
    void setCommandThread(Thread thread) {
      Thread old = commandThread.getAndSet(thread);
      if (old != null) {
        throw new IllegalStateException(String.format("Command state set twice (thread %s ->%s)",
            old.getName(), Thread.currentThread().getName()));
      }
    }

    /**
     * Sends an item to the client.
     *
     * @return true if the item was sent successfully, false if the connection to the client was
     *     lost
     */
    @VisibleForTesting
    synchronized boolean offer(RunResponse item) {
      SinkThreadItem queueItem = new SinkThreadItem(false, item);
      actionQueue.offer(SinkThreadAction.SEND);
      return exchange(queueItem, false).success;
    }

    private boolean disconnected() {
      return disconnected.get();
    }

    @VisibleForTesting
    boolean finish() {
      actionQueue.offer(SinkThreadAction.FINISH);
      try {
        Uninterruptibles.getUninterruptibly(future);
      } catch (ExecutionException e) {
        throw new IllegalStateException(e);
      }

      // Reset the interrupted bit so that it doesn't stay set for the next command that is handled
      // by this thread
      Thread.interrupted();
      return disconnected();
    }

    private SinkThreadItem exchange(SinkThreadItem item, boolean swallowInterrupts) {
      boolean interrupted = false;
      SinkThreadItem result;
      while (true) {
        try {
          result = exchanger.exchange(item);
          break;
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }

      if (interrupted && !swallowInterrupts) {
        Thread.currentThread().interrupt();
      }

      return result;
    }

    private void sendPendingItem() {
      SinkThreadItem item = exchange(new SinkThreadItem(true, null), true);
      try {
        observer.onNext(item.message);
      } catch (StatusRuntimeException e) {
        // The RPC was cancelled e.g. by the client terminating unexpectedly. We'll eventually get
        // notified about this and interrupt the command thread, but in the meantime, we can just
        // ignore the error; the client is dead, so there isn't anyone to talk to so swallowing the
        // output is fine.
        log.info(String.format("Client cancelled command for streamer thread %s",
            Thread.currentThread().getName()));
      }
    }

    /** Main function of the streamer thread. */
    private void call() {
      boolean itemPending = false;

      while (true) {
        SinkThreadAction action;
        action = Uninterruptibles.takeUninterruptibly(actionQueue);
        switch (action) {
          case FINISH:
            // Reset the interrupted bit so that it doesn't stay set for the next command that is
            // handled by this thread
            Thread.interrupted();
            return;

          case READY:
            if (itemPending) {
              sendPendingItem();
              itemPending = false;
            }
            break;

          case DISCONNECT:
            log.info("Client disconnected for stream thread " + Thread.currentThread().getName());
            disconnected.set(true);
            break;

          case SEND:
            if (disconnected()) {
              exchange(new SinkThreadItem(false, null), true);
            } else if (observer.isReady()) {
              sendPendingItem();
            } else {
              itemPending = true;
            }
        }
      }
    }
  }

  // TODO(lberki): Maybe we should implement line buffering?
  @VisibleForTesting
  static class RpcOutputStream extends OutputStream {
    private static final int CHUNK_SIZE = 8192;

    private final String commandId;
    private final String responseCookie;
    private final StreamType type;
    private final GrpcSink sink;

    RpcOutputStream(String commandId, String responseCookie, StreamType type, GrpcSink sink) {
      this.commandId = commandId;
      this.responseCookie = responseCookie;
      this.type = type;
      this.sink = sink;
    }

    @Override
    public synchronized void write(byte[] b, int off, int inlen) throws IOException {
      for (int i = 0; i < inlen; i += CHUNK_SIZE) {
        ByteString input = ByteString.copyFrom(b, off + i, Math.min(CHUNK_SIZE, inlen - i));
        RunResponse.Builder response = RunResponse
            .newBuilder()
            .setCookie(responseCookie)
            .setCommandId(commandId);

        switch (type) {
          case STDOUT: response.setStandardOutput(input); break;
          case STDERR: response.setStandardError(input); break;
          default: throw new IllegalStateException();
        }

        // Send the chunk to the streamer thread. May block.
        if (!sink.offer(response.build())) {
          // Client disconnected. Terminate the current command as soon as possible. Note that
          // throwing IOException is not enough because we are in the habit of swallowing it. Note
          // that when gRPC notifies us about the disconnection (see the call to setOnCancelHandler)
          // we interrupt the command thread, which should be enough to make the server come around
          // as soon as possible.
          log.info(
              String.format(
                  "Client disconnected received for command %s on thread %s",
                  commandId, Thread.currentThread().getName()));
          throw new IOException("Client disconnected");
        }
      }
    }

    @Override
    public void write(int byteAsInt) throws IOException {
      byte b = (byte) byteAsInt; // make sure we work with bytes in comparisons
      write(new byte[] {b}, 0, 1);
    }
  }

  // These paths are all relative to the server directory
  private static final String PORT_FILE = "command_port";
  private static final String REQUEST_COOKIE_FILE = "request_cookie";
  private static final String RESPONSE_COOKIE_FILE = "response_cookie";

  @GuardedBy("runningCommands")
  private final Map<String, RunningCommand> runningCommands = new HashMap<>();
  private final CommandExecutor commandExecutor;
  private final ExecutorService streamExecutorPool;
  private final ExecutorService commandExecutorPool;
  private final Clock clock;
  private final Path serverDirectory;
  private final String requestCookie;
  private final String responseCookie;
  private final AtomicLong interruptCounter = new AtomicLong(0);
  private final int maxIdleSeconds;

  private Server server;
  private final int port;
  boolean serving;

  public GrpcServerImpl(CommandExecutor commandExecutor, Clock clock, int port,
      Path serverDirectory, int maxIdleSeconds) throws IOException {
    super(serverDirectory);
    this.commandExecutor = commandExecutor;
    this.clock = clock;
    this.serverDirectory = serverDirectory;
    this.port = port;
    this.maxIdleSeconds = maxIdleSeconds;
    this.serving = false;

    this.streamExecutorPool =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("grpc-stream-%d").setDaemon(true).build());

    this.commandExecutorPool =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("grpc-command-%d").setDaemon(true).build());

    SecureRandom random = new SecureRandom();
    requestCookie = generateCookie(random, 16);
    responseCookie = generateCookie(random, 16);
  }

  private static String generateCookie(SecureRandom random, int byteCount) {
    byte[] bytes = new byte[byteCount];
    random.nextBytes(bytes);
    StringBuilder result = new StringBuilder();
    for (byte b : bytes) {
      result.append(Integer.toHexString(b + 128));
    }

    return result.toString();
  }

  private void startSlowInterruptWatcher(final ImmutableSet<String> commandIds) {
    if (commandIds.isEmpty()) {
      return;
    }

    Runnable interruptWatcher = new Runnable() {
      @Override
      public void run() {
        try {
          boolean ok;
          Thread.sleep(10 * 1000);
          synchronized (runningCommands) {
            ok = Collections.disjoint(commandIds, runningCommands.keySet());
          }
          if (!ok) {
            // At least one command was not interrupted. Interrupt took too long.
            ThreadUtils.warnAboutSlowInterrupt();
          }
        } catch (InterruptedException e) {
          // Ignore.
        }
      }
    };

    Thread interruptWatcherThread =
        new Thread(interruptWatcher, "interrupt-watcher-" + interruptCounter.incrementAndGet());
    interruptWatcherThread.setDaemon(true);
    interruptWatcherThread.start();
  }

  private void timeoutThread() {
    synchronized (runningCommands) {
      boolean idle = runningCommands.isEmpty();
      boolean wasIdle = false;
      long shutdownTime = -1;

      while (true) {
        if (!wasIdle && idle) {
          shutdownTime = BlazeClock.nanoTime() + maxIdleSeconds * 1000L * NANOSECONDS_IN_MS;
        }

        try {
          if (idle) {
            Verify.verify(shutdownTime > 0);
            long waitTime = shutdownTime - BlazeClock.nanoTime();
            if (waitTime > 0) {
              // Round upwards so that we don't busy-wait in the last millisecond
              runningCommands.wait((waitTime + NANOSECONDS_IN_MS - 1) / NANOSECONDS_IN_MS);
            }
          } else {
            runningCommands.wait();
          }
        } catch (InterruptedException e) {
          // Dealt with by checking the current time below.
        }

        wasIdle = idle;
        idle = runningCommands.isEmpty();
        if (wasIdle && idle && BlazeClock.nanoTime() >= shutdownTime) {
          break;
        }
      }
    }

    server.shutdown();
  }

  @Override
  public void interrupt() {
    synchronized (runningCommands) {
      for (RunningCommand command : runningCommands.values()) {
        command.thread.interrupt();
      }

      startSlowInterruptWatcher(ImmutableSet.copyOf(runningCommands.keySet()));
    }
  }

  @Override
  public void serve() throws IOException {
    Preconditions.checkState(!serving);

    // For reasons only Apple knows, you cannot bind to IPv4-localhost when you run in a sandbox
    // that only allows loopback traffic, but binding to IPv6-localhost works fine. This would
    // however break on systems that don't support IPv6. So what we'll do is to try to bind to IPv6
    // and if that fails, try again with IPv4.
    InetSocketAddress address = new InetSocketAddress("[::1]", port);
    try {
      server =
          NettyServerBuilder.forAddress(address)
              .addService(commandServer)
              .directExecutor()
              .build()
              .start();
    } catch (IOException e) {
      address = new InetSocketAddress("127.0.0.1", port);
      server =
          NettyServerBuilder.forAddress(address)
              .addService(commandServer)
              .directExecutor()
              .build()
              .start();
    }

    if (maxIdleSeconds > 0) {
      Thread timeoutThread =
          new Thread(
              new Runnable() {
                @Override
                public void run() {
                  timeoutThread();
                }
              });

      timeoutThread.setName("grpc-timeout");
      timeoutThread.setDaemon(true);
      timeoutThread.start();
    }
    serving = true;

    writeServerFile(
        PORT_FILE, InetAddresses.toUriString(address.getAddress()) + ":" + server.getPort());
    writeServerFile(REQUEST_COOKIE_FILE, requestCookie);
    writeServerFile(RESPONSE_COOKIE_FILE, responseCookie);

    try {
      server.awaitTermination();
    } catch (InterruptedException e) {
      // TODO(lberki): Handle SIGINT in a reasonable way
      throw new IllegalStateException(e);
    }
  }

  private void writeServerFile(String name, String contents) throws IOException {
    Path file = serverDirectory.getChild(name);
    FileSystemUtils.writeContentAsLatin1(file, contents);
    deleteAtExit(file, false);
  }

  private void executeCommand(
      RunRequest request, StreamObserver<RunResponse> observer, GrpcSink sink) {
    sink.setCommandThread(Thread.currentThread());

    if (!request.getCookie().equals(requestCookie) || request.getClientDescription().isEmpty()) {
      try {
        observer.onNext(
            RunResponse.newBuilder()
                .setExitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode())
                .build());
        observer.onCompleted();
      } catch (StatusRuntimeException e) {
        log.info("Client cancelled command while rejecting it: " + e.getMessage());
      }
      return;
    }

    // There is a small period of time between calling setOnCancelHandler() and setCommandThread()
    // during which the command thread is not interrupted when a cancel is signaled. Cover that
    // case by explicitly checking for disconnection here.
    if (sink.disconnected()) {
      return;
    }

    ImmutableList.Builder<String> args = ImmutableList.builder();
    for (ByteString requestArg : request.getArgList()) {
      args.add(requestArg.toString(CHARSET));
    }

    String commandId;
    int exitCode;

    try (RunningCommand command = new RunningCommand()) {
      commandId = command.id;

      try {
        // Send the client the command id as soon as we know it.
        observer.onNext(
            RunResponse.newBuilder()
                .setCookie(responseCookie)
                .setCommandId(commandId)
                .build());
      } catch (StatusRuntimeException e) {
        log.info(
            "The client cancelled the command before receiving the command id: " + e.getMessage());
      }

      OutErr rpcOutErr =
          OutErr.create(
              new RpcOutputStream(command.id, responseCookie, StreamType.STDOUT, sink),
              new RpcOutputStream(command.id, responseCookie, StreamType.STDERR, sink));

      exitCode =
          commandExecutor.exec(
              args.build(),
              rpcOutErr,
              request.getBlockForLock() ? LockingMode.WAIT : LockingMode.ERROR_OUT,
              request.getClientDescription(),
              clock.currentTimeMillis());

    } catch (InterruptedException e) {
      exitCode = ExitCode.INTERRUPTED.getNumericExitCode();
      commandId = ""; // The default value, the client will ignore it
    }

    if (sink.finish()) {
      // Client disconnected. Then we are not allowed to call any methods on the observer.
      return;
    }

    // There is a chance that an Uninterruptibles#getUninterruptibly() leaves us with the
    // interrupt bit set. So we just reset the interruption state here to make these cancel
    // requests not have any effect outside of command execution (after the try block above,
    // the cancel request won't find the thread to interrupt)
    Thread.interrupted();

    RunResponse response =
        RunResponse.newBuilder()
            .setCookie(responseCookie)
            .setCommandId(commandId)
            .setFinished(true)
            .setExitCode(exitCode)
            .build();

    try {
      observer.onNext(response);
      observer.onCompleted();
    } catch (StatusRuntimeException e) {
      // The client cancelled the call. Log an error and go on.
      log.info(String.format("Client cancelled command %s just right before its end: %s",
          commandId, e.getMessage()));
    }

    switch (commandExecutor.shutdown()) {
      case NONE:
        break;

      case CLEAN:
        server.shutdown();
        break;

      case EXPUNGE:
        disableShutdownHooks();
        server.shutdown();
        break;
    }
  }

  private final CommandServerGrpc.CommandServerImplBase commandServer =
      new CommandServerGrpc.CommandServerImplBase() {
        @Override
        public void run(final RunRequest request, final StreamObserver<RunResponse> observer) {
          final GrpcSink sink = new GrpcSink((ServerCallStreamObserver<RunResponse>) observer,
              streamExecutorPool);
          // Switch to our own threads so that onReadyStateHandler can be called (see class-level
          // comment)
          commandExecutorPool.execute(
              new Runnable() {
                @Override
                public void run() {
                  executeCommand(request, observer, sink);
                }
              });
        }

        @Override
        public void ping(PingRequest pingRequest, StreamObserver<PingResponse> streamObserver) {
          Preconditions.checkState(serving);

          try (RunningCommand command = new RunningCommand()) {
            PingResponse.Builder response = PingResponse.newBuilder();
            if (pingRequest.getCookie().equals(requestCookie)) {
              response.setCookie(responseCookie);
            }

            streamObserver.onNext(response.build());
            streamObserver.onCompleted();
          }
        }

        @Override
        public void cancel(
            final CancelRequest request, final StreamObserver<CancelResponse> streamObserver) {
          log.info("Got cancel message for " + request.getCommandId());
          if (!request.getCookie().equals(requestCookie)) {
            streamObserver.onCompleted();
            return;
          }

          // Actually performing the cancellation can result in some blocking which we don't want
          // to do on the dispatcher thread, instead offload to command pool.
          commandExecutorPool.execute(new Runnable() {
            @Override
            public void run() {
              doCancel(request, streamObserver);
            }
          });
        }

        private void doCancel(
            CancelRequest request, StreamObserver<CancelResponse> streamObserver) {
          try (RunningCommand cancelCommand = new RunningCommand()) {
            synchronized (runningCommands) {
              RunningCommand pendingCommand = runningCommands.get(request.getCommandId());
              if (pendingCommand != null) {
                log.info(
                    String.format(
                        "Interrupting command %s on thread %s",
                        request.getCommandId(), pendingCommand.thread.getName()));
                pendingCommand.thread.interrupt();
                startSlowInterruptWatcher(ImmutableSet.of(request.getCommandId()));
              } else {
                log.info("Cannot find command " + request.getCommandId() + " to interrupt");
              }
            }

            try {
              streamObserver.onNext(CancelResponse.newBuilder().setCookie(responseCookie).build());
              streamObserver.onCompleted();
            } catch (StatusRuntimeException e) {
              // There is no one to report the failure to
              log.info("Client cancelled RPC of cancellation request for "
                  + request.getCommandId());
            }
          }
        }
      };
}
