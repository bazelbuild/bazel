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
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.server.CommandProtos.StartupOption;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ThreadUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.InvocationPolicyParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.InetSocketAddress;
import java.nio.charset.Charset;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
public class GrpcServerImpl implements RPCServer {
  private static final Logger logger = Logger.getLogger(GrpcServerImpl.class.getName());

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
        if (runningCommands.isEmpty()) {
          busy();
        }
        runningCommands.put(id, this);
        runningCommands.notify();
      }

      logger.info(String.format("Starting command %s on thread %s", id, thread.getName()));
    }

    @Override
    public void close() {
      synchronized (runningCommands) {
        runningCommands.remove(id);
        if (runningCommands.isEmpty()) {
          idle();
        }
        runningCommands.notify();
      }

      logger.info(String.format("Finished command %s on thread %s", id, thread.getName()));
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
      Path workspace, Path serverDirectory, int maxIdleSeconds) throws IOException {
      return new GrpcServerImpl(
          commandExecutor, clock, port, workspace, serverDirectory, maxIdleSeconds);
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
   * A class that handles communicating through a gRPC interface for a streaming rpc call.
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
    private final AtomicLong receivedEventCount = new AtomicLong(0);

    @VisibleForTesting
    GrpcSink(
        final String rpcCommandName,
        ServerCallStreamObserver<RunResponse> observer,
        ExecutorService executor) {
      // This queue is intentionally unbounded: we always act on it fairly quickly so filling up
      // RAM is not a concern but we don't want to block in the gRPC cancel/onready handlers.
      this.actionQueue = new LinkedBlockingQueue<>();
      this.exchanger = new Exchanger<>();
      this.observer = observer;
      this.observer.setOnCancelHandler(
          () -> {
            Thread commandThread = GrpcSink.this.commandThread.get();
            if (commandThread != null) {
              logger.info(
                  String.format(
                      "Interrupting thread %s due to the streaming %s call being cancelled "
                          + "(likely client hang up or explicit gRPC-level cancellation)",
                      commandThread.getName(), rpcCommandName));
              commandThread.interrupt();
            }

            actionQueue.offer(SinkThreadAction.DISCONNECT);
          });
      this.observer.setOnReadyHandler(() -> actionQueue.offer(SinkThreadAction.READY));
      this.future = executor.submit(GrpcSink.this::call);
    }

    @VisibleForTesting
    long getReceivedEventCount() {
      return receivedEventCount.get();
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
        logger.info(
            String.format(
                "Client cancelled command for streamer thread %s",
                Thread.currentThread().getName()));
      }
    }

    /** Main function of the streamer thread. */
    private void call() {
      boolean itemPending = false;

      while (true) {
        SinkThreadAction action;
        action = Uninterruptibles.takeUninterruptibly(actionQueue);
        receivedEventCount.incrementAndGet();
        switch (action) {
          case FINISH:
            if (itemPending) {
              exchange(new SinkThreadItem(false, null), true);
              itemPending = false;
            }

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
            logger.info(
                "Client disconnected for stream thread " + Thread.currentThread().getName());
            disconnected.set(true);
            if (itemPending) {
              exchange(new SinkThreadItem(false, null), true);
              itemPending = false;
            }
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

  /**
   * An output stream that forwards the data written to it over the gRPC command stream.
   *
   * <p>Note that wraping this class with a {@code Channel} can cause a deadlock if there is an
   * {@link OutputStream} in between that synchronizes both on {@code #close()} and {@code #write()}
   * because then if an interrupt happens in {@link GrpcSink#exchange(SinkThreadItem, boolean)},
   * the thread on which {@code interrupt()} was called will wait until the {@code Channel} closes
   * itself while holding a lock for interrupting the thread on which {@code #exchange()} is
   * being executed and that thread will hold a lock that is needed for the {@code Channel} to be
   * closed and call {@code interrupt()} in {@code #exchange()}, which will in turn try to acquire
   * the interrupt lock.
   */
  @VisibleForTesting
  static class RpcOutputStream extends OutputStream {
    private static final int CHUNK_SIZE = 8192;

    // Store commandId and responseCookie as ByteStrings to avoid String -> UTF8 bytes conversion
    // for each serialized chunk of output.
    private final ByteString commandIdBytes;
    private final ByteString responseCookieBytes;

    private final StreamType type;
    private final GrpcSink sink;

    RpcOutputStream(String commandId, String responseCookie, StreamType type, GrpcSink sink) {
      this.commandIdBytes = ByteString.copyFromUtf8(commandId);
      this.responseCookieBytes = ByteString.copyFromUtf8(responseCookie);
      this.type = type;
      this.sink = sink;
    }

    @Override
    public synchronized void write(byte[] b, int off, int inlen) throws IOException {
      for (int i = 0; i < inlen; i += CHUNK_SIZE) {
        ByteString input = ByteString.copyFrom(b, off + i, Math.min(CHUNK_SIZE, inlen - i));
        RunResponse.Builder response = RunResponse
            .newBuilder()
            .setCookieBytes(responseCookieBytes)
            .setCommandIdBytes(commandIdBytes);

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
          logger.info(
              String.format(
                  "Client disconnected received for command %s on thread %s",
                  commandIdBytes.toStringUtf8(), Thread.currentThread().getName()));
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

  /**
   * A thread that watches if the PID file changes and shuts down the server immediately if so.
   */
  private class PidFileWatcherThread extends Thread {
    private boolean shuttingDown = false;

    private PidFileWatcherThread() {
      super("pid-file-watcher");
      setDaemon(true);
    }

    // The synchronized block is here so that if the "PID file deleted" timer kicks in during a
    // regular shutdown, they don't race.
    private synchronized void signalShutdown() {
      shuttingDown = true;
    }

    @Override
    public void run() {
      while (true) {
        Uninterruptibles.sleepUninterruptibly(5, TimeUnit.SECONDS);
        boolean ok = false;
        try {
          String pidFileContents = new String(FileSystemUtils.readContentAsLatin1(pidFile));
          ok = pidFileContents.equals(pidInFile);
        } catch (IOException e) {
          logger.info("Cannot read PID file: " + e.getMessage());
          // Handled by virtue of ok not being set to true
        }

        if (!ok) {
          synchronized (PidFileWatcherThread.this) {
            if (shuttingDown) {
              logger.warning("PID file deleted or overwritten but shutdown is already in progress");
              break;
            }

            shuttingDown = true;
            // Someone overwrote the PID file. Maybe it's another server, so shut down as quickly
            // as possible without even running the shutdown hooks (that would delete it)
            logger.severe("PID file deleted or overwritten, exiting as quickly as possible");
            Runtime.getRuntime().halt(ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode());
          }
        }
      }
    }
  }

  // These paths are all relative to the server directory
  private static final String PORT_FILE = "command_port";
  private static final String REQUEST_COOKIE_FILE = "request_cookie";
  private static final String RESPONSE_COOKIE_FILE = "response_cookie";

  private static final AtomicBoolean runShutdownHooks = new AtomicBoolean(true);

  @GuardedBy("runningCommands")
  private final Map<String, RunningCommand> runningCommands = new HashMap<>();
  private final CommandExecutor commandExecutor;
  private final ExecutorService streamExecutorPool;
  private final ExecutorService commandExecutorPool;
  private final Clock clock;
  private final Path serverDirectory;
  private final Path workspace;
  private final String requestCookie;
  private final String responseCookie;
  private final AtomicLong interruptCounter = new AtomicLong(0);
  private final int maxIdleSeconds;
  private final PidFileWatcherThread pidFileWatcherThread;
  private final Path pidFile;
  private final String pidInFile;
  private final List<Path> filesToDeleteAtExit = new ArrayList<>();
  private final int port;

  private Server server;
  private IdleServerTasks idleServerTasks;
  boolean serving;

  public GrpcServerImpl(CommandExecutor commandExecutor, Clock clock, int port,
      Path workspace, Path serverDirectory, int maxIdleSeconds) throws IOException {
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        shutdownHook();
      }
    });

    // server.pid was written in the C++ launcher after fork() but before exec() .
    // The client only accesses the pid file after connecting to the socket
    // which ensures that it gets the correct pid value.
    pidFile = serverDirectory.getRelative("server.pid.txt");
    pidInFile = new String(FileSystemUtils.readContentAsLatin1(pidFile));
    deleteAtExit(pidFile);

    this.commandExecutor = commandExecutor;
    this.clock = clock;
    this.serverDirectory = serverDirectory;
    this.workspace = workspace;
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

    pidFileWatcherThread = new PidFileWatcherThread();
    pidFileWatcherThread.start();
    idleServerTasks = new IdleServerTasks(workspace);
    idleServerTasks.idle();
  }

  private void idle() {
    Preconditions.checkState(idleServerTasks == null);
    idleServerTasks = new IdleServerTasks(workspace);
    idleServerTasks.idle();
  }

  private void busy() {
    Preconditions.checkState(idleServerTasks != null);
    idleServerTasks.busy();
    idleServerTasks = null;
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

    Runnable interruptWatcher = () -> {
        try {
          Thread.sleep(10 * 1000);
          boolean ok;
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

  /**
   * This is called when the server is shut down as a result of a "clean --expunge".
   *
   * <p>In this case, no files should be deleted on shutdown hooks, since clean also deletes the
   * lock file, and there is a small possibility of the following sequence of events:
   *
   * <ol>
   *   <li> Client 1 runs "blaze clean --expunge"
   *   <li> Client 2 runs a command and waits for client 1 to finish
   *   <li> The clean command deletes everything including the lock file
   *   <li> Client 2 starts running and since the output base is empty, starts up a new server,
   *     which creates its own socket and PID files
   *   <li> The server used by client runs its shutdown hooks, deleting the PID files created by
   *     the new server
   * </ol>
   *
   * It also disables the "die when the PID file changes" handler so that it doesn't kill the server
   * while the "clean --expunge" commmand is running.
   */

  @Override
  public void prepareForAbruptShutdown() {
    disableShutdownHooks();
    pidFileWatcherThread.signalShutdown();
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
      Thread timeoutThread = new Thread(this::timeoutThread);
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
    deleteAtExit(file);
  }

  protected void disableShutdownHooks() {
    runShutdownHooks.set(false);
  }

  private void shutdownHook() {
    if (!runShutdownHooks.get()) {
      return;
    }

    List<Path> files;
    synchronized (filesToDeleteAtExit) {
      files = new ArrayList<>(filesToDeleteAtExit);
    }
    for (Path path : files) {
      try {
        path.delete();
      } catch (IOException e) {
        printStack(e);
      }
    }
  }

  /**
   * Schedule the specified file for (attempted) deletion at JVM exit.
   */
  protected void deleteAtExit(final Path path) {
    synchronized (filesToDeleteAtExit) {
      filesToDeleteAtExit.add(path);
    }
  }

  static void printStack(IOException e) {
    /*
     * Hopefully this never happens. It's not very nice to just write this
     * to the user's console, but I'm not sure what better choice we have.
     */
    StringWriter err = new StringWriter();
    PrintWriter printErr = new PrintWriter(err);
    printErr.println("=======[BLAZE SERVER: ENCOUNTERED IO EXCEPTION]=======");
    e.printStackTrace(printErr);
    printErr.println("=====================================================");
    logger.severe(err.toString());
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
        logger.info("Client cancelled command while rejecting it: " + e.getMessage());
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

    // TODO(b/63925394): This information needs to be passed to the GotOptionsEvent, which does not
    // currently have the explicit startup options. See Improved Command Line Reporting design doc
    // for details.
    // Convert the startup options record to Java strings, source first.
    ImmutableList.Builder<Pair<String, String>> startupOptions = ImmutableList.builder();
    for (StartupOption option : request.getStartupOptionsList()) {
      startupOptions.add(
          new Pair<>(option.getSource().toString(CHARSET), option.getOption().toString(CHARSET)));
    }

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
        logger.info(
            "The client cancelled the command before receiving the command id: " + e.getMessage());
      }

      OutErr rpcOutErr = OutErr.create(
          new RpcOutputStream(command.id, responseCookie, StreamType.STDOUT, sink),
          new RpcOutputStream(command.id, responseCookie, StreamType.STDERR, sink));

      try {
        InvocationPolicy policy = InvocationPolicyParser.parsePolicy(request.getInvocationPolicy());
        exitCode =
            commandExecutor.exec(
                policy,
                args.build(),
                rpcOutErr,
                request.getBlockForLock() ? LockingMode.WAIT : LockingMode.ERROR_OUT,
                request.getClientDescription(),
                clock.currentTimeMillis(),
                Optional.of(startupOptions.build()));
      } catch (OptionsParsingException e) {
        rpcOutErr.printErrLn(e.getMessage());
        exitCode = ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
      }
    } catch (InterruptedException e) {
      exitCode = ExitCode.INTERRUPTED.getNumericExitCode();
      commandId = ""; // The default value, the client will ignore it
    }

    if (sink.finish()) {
      // Client disconnected. Then we are not allowed to call any methods on the observer.
      logger.info(
          String.format(
              "Client disconnected before we could send exit code for command %s", commandId));
      return;
    }

    // There is a chance that an Uninterruptibles#getUninterruptibly() leaves us with the
    // interrupt bit set. So we just reset the interruption state here to make these cancel
    // requests not have any effect outside of command execution (after the try block above,
    // the cancel request won't find the thread to interrupt)
    Thread.interrupted();

    boolean shutdown = commandExecutor.shutdown();
    if (shutdown) {
      server.shutdown();
    }
    RunResponse response =
        RunResponse.newBuilder()
            .setCookie(responseCookie)
            .setCommandId(commandId)
            .setFinished(true)
            .setExitCode(exitCode)
            .setTerminationExpected(shutdown)
            .build();

    try {
      observer.onNext(response);
      observer.onCompleted();
    } catch (StatusRuntimeException e) {
      // The client cancelled the call. Log an error and go on.
      logger.info(
          String.format(
              "Client cancelled command %s just right before its end: %s",
              commandId, e.getMessage()));
    }
  }

  private final CommandServerGrpc.CommandServerImplBase commandServer =
      new CommandServerGrpc.CommandServerImplBase() {
        @Override
        public void run(final RunRequest request, final StreamObserver<RunResponse> observer) {
          final GrpcSink sink =
              new GrpcSink(
                  "Run", (ServerCallStreamObserver<RunResponse>) observer, streamExecutorPool);
          // Switch to our own threads so that onReadyStateHandler can be called (see class-level
          // comment)
          commandExecutorPool.execute(() -> executeCommand(request, observer, sink));
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
          logger.info(String.format("Got CancelRequest for command id %s", request.getCommandId()));
          if (!request.getCookie().equals(requestCookie)) {
            streamObserver.onCompleted();
            return;
          }

          // Actually performing the cancellation can result in some blocking which we don't want
          // to do on the dispatcher thread, instead offload to command pool.
          commandExecutorPool.execute(() -> doCancel(request, streamObserver));
        }

        private void doCancel(
            CancelRequest request, StreamObserver<CancelResponse> streamObserver) {
          try (RunningCommand cancelCommand = new RunningCommand()) {
            synchronized (runningCommands) {
              RunningCommand pendingCommand = runningCommands.get(request.getCommandId());
              if (pendingCommand != null) {
                logger.info(
                    String.format(
                        "Interrupting command %s on thread %s",
                        request.getCommandId(), pendingCommand.thread.getName()));
                pendingCommand.thread.interrupt();
                startSlowInterruptWatcher(ImmutableSet.of(request.getCommandId()));
              } else {
                logger.info("Cannot find command " + request.getCommandId() + " to interrupt");
              }
            }

            try {
              streamObserver.onNext(CancelResponse.newBuilder().setCookie(responseCookie).build());
              streamObserver.onCompleted();
            } catch (StatusRuntimeException e) {
              // There is no one to report the failure to
              logger.info(
                  "Client cancelled RPC of cancellation request for " + request.getCommandId());
            }
          }
        }
      };
}
