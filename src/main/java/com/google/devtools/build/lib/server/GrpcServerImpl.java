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

import com.google.common.base.Optional;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.net.InetAddresses;
import com.google.common.util.concurrent.Uninterruptibles;
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
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.CallStreamObserver;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import javax.annotation.concurrent.GuardedBy;

/**
 * gRPC server class.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 */
public class GrpcServerImpl extends RPCServer {
  private static final Logger LOG = Logger.getLogger(GrpcServerImpl.class.getName());

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

      LOG.info(String.format("Starting command %s on thread %s", id, thread.getName()));
    }

    @Override
    public void close() {
      synchronized (runningCommands) {
        runningCommands.remove(id);
        runningCommands.notify();
      }

      LOG.info(String.format("Finished command %s on thread %s", id, thread.getName()));
    }
  }

  /**
   * Factory class. Instantiated by reflection.
   */
  public static class Factory implements RPCServer.Factory {
    @Override
    public RPCServer create(CommandExecutor commandExecutor, Clock clock, int port,
      Path serverDirectory, int maxIdleSeconds) throws IOException {
      return new GrpcServerImpl(commandExecutor, clock, port, serverDirectory, maxIdleSeconds);
    }
  }

  private enum StreamType {
    STDOUT,
    STDERR,
  }

  private static Runnable streamRunnable(
      final LinkedBlockingQueue<Optional<RunResponse>> queue,
      final CallStreamObserver<RunResponse> observer) {
    return new Runnable() {
      @Override
      public void run() {
        while (true) {
          Optional<RunResponse> item;
          try {
            item = queue.take();
          } catch (InterruptedException e) {
            // Ignore. This is running on its own thread to which interrupts are never delivered
            // except by explicit SIGINT to that thread, which is a case we can ignore.
            continue;
          }
          if (!item.isPresent()) {
            return;
          }

          observer.onNext(item.get());
        }
      }
    };
  }

  // TODO(lberki): Maybe we should implement line buffering?
  private class RpcOutputStream extends OutputStream {
    private final String commandId;
    private final StreamType type;
    private final LinkedBlockingQueue<Optional<RunResponse>> work;

    private RpcOutputStream(String commandId, StreamType type,
        LinkedBlockingQueue<Optional<RunResponse>> work) {
      this.commandId = commandId;
      this.type = type;
      this.work = work;
    }

    @Override
    public void write(byte[] b, int off, int inlen) {
      ByteString input = ByteString.copyFrom(b, off, inlen);
      RunResponse.Builder response = RunResponse
          .newBuilder()
          .setCookie(responseCookie)
          .setCommandId(commandId);

      switch (type) {
        case STDOUT: response.setStandardOutput(input); break;
        case STDERR: response.setStandardError(input); break;
        default: throw new IllegalStateException();
      }
      work.offer(Optional.of(response.build()));
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
  private final Clock clock;
  private final Path serverDirectory;
  private final String requestCookie;
  private final String responseCookie;
  private final AtomicLong interruptCounter = new AtomicLong(0);
  private final ExecutorService streamExecutor;
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

    final AtomicInteger counter = new AtomicInteger(1);
    this.streamExecutor = Executors.newCachedThreadPool(new ThreadFactory() {
      @Override
      public Thread newThread(Runnable r) {
        Thread result = new Thread(r);
        result.setName("streamer-" + counter.getAndAdd(1));
        result.setDaemon(true);
        return result;
      }
    });

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
      server = NettyServerBuilder.forAddress(address).addService(commandServer).build().start();
    } catch (IOException e) {
      address = new InetSocketAddress("127.0.0.1", port);
      server = NettyServerBuilder.forAddress(address).addService(commandServer).build().start();
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


  private final CommandServerGrpc.CommandServerImplBase commandServer =
      new CommandServerGrpc.CommandServerImplBase() {
        @Override
        public void run(RunRequest request, StreamObserver<RunResponse> observer) {
          if (!request.getCookie().equals(requestCookie)
              || request.getClientDescription().isEmpty()) {
            observer.onNext(
                RunResponse.newBuilder()
                    .setExitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode())
                    .build());
            observer.onCompleted();
            return;
          }

          ImmutableList.Builder<String> args = ImmutableList.builder();
          for (ByteString requestArg : request.getArgList()) {
            args.add(requestArg.toString(CHARSET));
          }

          String commandId;
          int exitCode;
          LinkedBlockingQueue<Optional<RunResponse>> work = new LinkedBlockingQueue<>();
          Future<?> streamFuture = streamExecutor.submit(streamRunnable(
              work, (CallStreamObserver<RunResponse>) observer));

          try (RunningCommand command = new RunningCommand()) {
            commandId = command.id;
            OutErr rpcOutErr =
                OutErr.create(
                    new RpcOutputStream(command.id, StreamType.STDOUT, work),
                    new RpcOutputStream(command.id, StreamType.STDERR, work));

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

          // Signal the streamer thread to exit. If we don't do this, streamFuture will never get
          // computed and we hang.
          work.offer(Optional.<RunResponse>absent());
          try {
            Uninterruptibles.getUninterruptibly(streamFuture);
          } catch (ExecutionException e) {
            throw new IllegalStateException(e);
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

          observer.onNext(response);
          observer.onCompleted();

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
        public void cancel(CancelRequest request, StreamObserver<CancelResponse> streamObserver) {
          if (!request.getCookie().equals(requestCookie)) {
            streamObserver.onCompleted();
            return;
          }

          try (RunningCommand cancelCommand = new RunningCommand()) {
            synchronized (runningCommands) {
              RunningCommand pendingCommand = runningCommands.get(request.getCommandId());
              if (pendingCommand != null) {
                LOG.info(String.format("Interrupting command %s on thread %s",
                        request.getCommandId(), pendingCommand.thread.getName()));
                pendingCommand.thread.interrupt();
              } else {
                LOG.info("Cannot find command " + request.getCommandId() + " to interrupt");
              }

              startSlowInterruptWatcher(ImmutableSet.of(request.getCommandId()));
            }

            streamObserver.onNext(CancelResponse.newBuilder().setCookie(responseCookie).build());
            streamObserver.onCompleted();
          }
        }
      };
}
