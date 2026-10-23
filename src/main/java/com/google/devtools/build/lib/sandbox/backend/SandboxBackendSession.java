// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import build.bazel.remote.execution.v2.CapabilitiesGrpc;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionGrpc;
import build.bazel.remote.execution.v2.GetCapabilitiesRequest;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.ChannelConnectionWithServerCapabilitiesFactory;
import com.google.devtools.build.lib.remote.GrpcCacheClient;
import com.google.devtools.build.lib.remote.ReferenceCountedChannel;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.longrunning.Operation;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptors;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.ClientCalls;
import io.reactivex.rxjava3.core.Single;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Iterator;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * A running sandbox backend and everything Bazel needs to talk to it.
 *
 * <p>The backend is launched as {@code <binary> serve --socket=<path> <options>...}. It has to
 * create the socket at {@code <path>} once it accepts connections, serve the REAPI services {@code
 * Execution}, {@code ByteStream} and {@code ContentAddressableStorage} (the latter only {@code
 * FindMissingBlobs}) on it, and exit when Bazel kills it. Bazel never closes the standard input
 * of the backend and ignores its standard output and error except to report them if the backend
 * dies.
 *
 * <p>The state of a session, in particular what the backend is known to hold, is tied to the
 * backend process: a restarted backend gets a new session.
 */
final class SandboxBackendSession implements AutoCloseable {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Duration STARTUP_TIMEOUT = Duration.ofSeconds(30);
  private static final Duration CAPABILITIES_TIMEOUT = Duration.ofSeconds(30);
  private static final Duration OUTPUT_DRAIN_TIMEOUT = Duration.ofSeconds(1);
  private static final int OUTPUT_TAIL_LIMIT = 8 * 1024;

  private final String name;
  @Nullable private final Subprocess process;
  @Nullable private final SandboxBackendTransport transport;
  private final ReferenceCountedChannel channel;
  private final ListeningScheduledExecutorService retryScheduler;
  private final GrpcCacheClient cacheClient;
  private final SandboxBackendCache cache;
  private final DigestUtil digestUtil;
  private final StringBuilder outputTail = new StringBuilder();
  private final AtomicBoolean channelReleased = new AtomicBoolean();
  private volatile boolean alive = true;
  @Nullable private volatile String deathMessage;
  @Nullable private MerkleTreeComputer merkleTreeComputer;
  @Nullable private String merkleTreeComputerCommandId;

  private SandboxBackendSession(
      String name,
      @Nullable Subprocess process,
      @Nullable SandboxBackendTransport transport,
      ChannelSupplier channelSupplier,
      DigestUtil digestUtil)
      throws IOException {
    this.name = name;
    this.process = process;
    this.transport = transport;
    this.digestUtil = digestUtil;
    this.channel =
        new ReferenceCountedChannel(
            new ChannelConnectionWithServerCapabilitiesFactory() {
              @Override
              public Single<ChannelConnectionWithServerCapabilities> create() {
                return Single.fromCallable(
                    () -> {
                      ManagedChannel connection = channelSupplier.get();
                      return new ChannelConnectionWithServerCapabilities(
                          connection, Single.fromCallable(() -> getCapabilities(connection)));
                    });
              }

              @Override
              public int maxConcurrency() {
                return 100;
              }
            });
    this.retryScheduler =
        MoreExecutors.listeningDecorator(
            Executors.newScheduledThreadPool(
                1, Thread.ofPlatform().daemon().name("sandbox-backend-retry-" + name).factory()));
    RemoteOptions options;
    try {
      // The backend is local and trusted. Retrying doesn't help if it is gone, and it can't be
      // slow in the way that a network service can be.
      options = Options.parse(RemoteOptions.class, "--remote_retries=0").getOptions();
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
    RemoteRetrier retrier =
        new RemoteRetrier(
            options,
            RemoteRetrier.EXPERIMENTAL_GRPC_RESULT_CLASSIFIER,
            retryScheduler,
            Retrier.ALLOW_ALL_CALLS);
    this.cacheClient =
        new GrpcCacheClient(
            channel, CallCredentialsProvider.NO_CREDENTIALS, options, retrier, digestUtil);
    this.cache = new SandboxBackendCache(cacheClient, digestUtil);
  }

  private ServerCapabilities getCapabilities(ManagedChannel connection) throws IOException {
    try {
      return CapabilitiesGrpc.newBlockingStub(connection)
          .withDeadlineAfter(CAPABILITIES_TIMEOUT.toSeconds(), TimeUnit.SECONDS)
          .getCapabilities(GetCapabilitiesRequest.getDefaultInstance());
    } catch (StatusRuntimeException e) {
      throw new IOException(
          e.getStatus().getCode() == Status.Code.UNIMPLEMENTED
              // Treated as a failure rather than as "supports everything": a backend has to
              // declare that it executes, and which digest functions and API versions it speaks.
              ? String.format(
                  "sandbox backend '%s' does not implement GetCapabilities, which it has to to"
                      + " declare that it supports execution",
                  name)
              : String.format("sandbox backend '%s' failed GetCapabilities: %s", name, e.getStatus()),
          e);
    }
  }

  /**
   * Asks the backend what it supports, once per session, and fails if that isn't enough for Bazel.
   */
  private void checkCapabilities() throws IOException {
    ServerCapabilities capabilities;
    try {
      capabilities = channel.getServerCapabilities();
    } catch (IOException e) {
      throw !alive && deathMessage != null ? new IOException(deathMessage, e) : e;
    }
    String problem =
        SandboxBackendCapabilities.check(name, capabilities, digestUtil.getDigestFunction());
    if (problem != null) {
      throw new IOException(problem);
    }
  }

  @FunctionalInterface
  interface ChannelSupplier {
    ManagedChannel get() throws IOException;
  }

  /**
   * Launches the backend and waits for it to create its socket.
   *
   * @throws IOException if the backend can't be launched, exits or doesn't become ready in time
   */
  static SandboxBackendSession start(
      String name,
      BackendConfig config,
      ImmutableMap<String, String> clientEnv,
      DigestUtil digestUtil)
      throws IOException, InterruptedException {
    return start(name, config, clientEnv, digestUtil, STARTUP_TIMEOUT);
  }

  /** Like {@link #start(String, BackendConfig, ImmutableMap, DigestUtil)}, with a custom timeout. */
  static SandboxBackendSession start(
      String name,
      BackendConfig config,
      ImmutableMap<String, String> clientEnv,
      DigestUtil digestUtil,
      Duration startupTimeout)
      throws IOException, InterruptedException {
    String binary = config.resolveBinary(clientEnv);
    if (binary == null) {
      throw new IOException(
          String.format(
              "sandbox backend '%s': binary '%s' doesn't exist or isn't executable",
              name, config.binary()));
    }
    SandboxBackendTransport transport = SandboxBackendTransport.create();
    Subprocess process = null;
    SandboxBackendSession session = null;
    try {
      process =
          new SubprocessBuilder(clientEnv)
              .setArgv(
                  ImmutableList.<String>builder()
                      .add(binary, "serve", "--socket=" + transport.socketPath())
                      .addAll(config.options())
                      .build())
              .setStdout(SubprocessBuilder.StreamAction.STREAM)
              .redirectErrorStream(true)
              .setWorkingDirectory(new File("."))
              .start();
      session = new SandboxBackendSession(name, process, transport, transport::connect, digestUtil);
      session.watchProcess();
      session.awaitReady(startupTimeout);
      session.checkCapabilities();
      return session;
    } catch (IOException | InterruptedException | RuntimeException e) {
      if (session != null) {
        session.close();
      } else {
        if (process != null) {
          process.destroy();
        }
        transport.close();
      }
      throw e;
    }
  }

  /** Creates a session for a backend that is served on {@code channel}. */
  static SandboxBackendSession forChannel(
      String name, ChannelSupplier channel, DigestUtil digestUtil) throws IOException {
    SandboxBackendSession session =
        new SandboxBackendSession(
            name, /* process= */ null, /* transport= */ null, channel, digestUtil);
    try {
      session.checkCapabilities();
    } catch (IOException | RuntimeException e) {
      session.close();
      throw e;
    }
    return session;
  }

  private void watchProcess() {
    Thread reader =
        new Thread(
            () -> {
              try (Reader in =
                  new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8)) {
                char[] buffer = new char[1024];
                int n;
                while ((n = in.read(buffer)) != -1) {
                  synchronized (outputTail) {
                    outputTail.append(buffer, 0, n);
                    if (outputTail.length() > OUTPUT_TAIL_LIMIT) {
                      outputTail.delete(0, outputTail.length() - OUTPUT_TAIL_LIMIT);
                    }
                  }
                }
              } catch (IOException e) {
                // The process is gone, which is handled by the waiting thread.
              }
            },
            "sandbox-backend-output-" + name);
    reader.setDaemon(true);
    reader.start();
    Thread waiter =
        new Thread(
            () -> {
              process.waitForUninterruptibly();
              // The output is complete once the reader sees the end of the stream, unless a child
              // of the backend keeps it open, in which case what was read so far has to do.
              try {
                reader.join(OUTPUT_DRAIN_TIMEOUT.toMillis());
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
              if (alive) {
                // Set before alive is cleared, as whoever sees the backend dead reads it.
                deathMessage =
                    String.format(
                        "sandbox backend '%s' exited with status %d%s",
                        name, process.exitValue(), formatTail());
                alive = false;
                logger.atWarning().log("%s", deathMessage);
                // Fails all calls that are waiting for the backend.
                releaseChannel();
              }
            },
            "sandbox-backend-waiter-" + name);
    waiter.setDaemon(true);
    waiter.start();
  }

  private String formatTail() {
    synchronized (outputTail) {
      return outputTail.length() == 0 ? "" : ", its output:\n" + outputTail;
    }
  }

  private void awaitReady(Duration timeout) throws IOException, InterruptedException {
    long deadline = System.nanoTime() + timeout.toNanos();
    while (!transport.isSocketCreated()) {
      if (!alive) {
        throw new IOException(deathMessage);
      }
      if (System.nanoTime() > deadline) {
        throw new IOException(
            String.format(
                "sandbox backend '%s' didn't create its socket within %d seconds%s",
                name, timeout.toSeconds(), formatTail()));
      }
      Thread.sleep(10);
    }
  }

  /** Whether the backend is still running. */
  boolean isAlive() {
    return alive;
  }

  /** A description of why the backend died, or null if it is still running. */
  @Nullable
  String deathMessage() {
    return deathMessage;
  }

  SandboxBackendCache cache() {
    return cache;
  }

  GrpcCacheClient cacheClient() {
    return cacheClient;
  }

  DigestUtil digestUtil() {
    return digestUtil;
  }

  /**
   * Returns the computer of Merkle trees for the given command. Its knowledge of what has already
   * been uploaded is shared by all commands for as long as this session lives.
   */
  synchronized MerkleTreeComputer merkleTreeComputer(
      String buildRequestId, String commandId, String workspaceName) {
    if (!commandId.equals(merkleTreeComputerCommandId)) {
      merkleTreeComputer =
          new MerkleTreeComputer(
              digestUtil, cache, buildRequestId, commandId, workspaceName, /* uploadScope= */ this);
      merkleTreeComputerCommandId = commandId;
    }
    return merkleTreeComputer;
  }

  /**
   * Runs the action and returns its result.
   *
   * <p>If the thread is interrupted, the call is cancelled, which the backend takes as the request
   * to kill the process of the action.
   *
   * @throws StatusRuntimeException if the call fails
   */
  ExecuteResponse execute(RequestMetadata metadata, ExecuteRequest request)
      throws IOException, InterruptedException {
    return channel.withChannelBlocking(
        ch -> {
          Channel intercepted =
              ClientInterceptors.intercept(
                  ch, TracingMetadataUtils.attachMetadataInterceptor(metadata));
          ClientCall<ExecuteRequest, Operation> call =
              intercepted.newCall(ExecutionGrpc.getExecuteMethod(), CallOptions.DEFAULT);
          try {
            Iterator<Operation> operations =
                ClientCalls.blockingServerStreamingCall(call, request);
            while (operations.hasNext()) {
              Operation operation = operations.next();
              if (!operation.getDone()) {
                continue;
              }
              return switch (operation.getResultCase()) {
                case ERROR ->
                    ExecuteResponse.newBuilder().setStatus(operation.getError()).build();
                case RESPONSE -> operation.getResponse().unpack(ExecuteResponse.class);
                case RESULT_NOT_SET ->
                    throw new IOException("sandbox backend returned a finished operation without a result");
              };
            }
            throw new IOException("sandbox backend ended the Execute stream without a result");
          } catch (StatusRuntimeException e) {
            if (e.getStatus().getCode() == Status.Code.CANCELLED && Thread.interrupted()) {
              throw new InterruptedException();
            }
            throw e;
          } finally {
            // Does nothing if the call has completed.
            call.cancel("Execute finished", null);
          }
        });
  }

  private void releaseChannel() {
    if (channelReleased.compareAndSet(false, true)) {
      channel.release();
    }
  }

  @Override
  public void close() {
    alive = false;
    if (process != null) {
      process.destroyAndWait();
    }
    releaseChannel();
    retryScheduler.shutdownNow();
    if (transport != null) {
      transport.close();
    }
  }

  // Referenced to keep the status utilities used by callers in one place.
  static com.google.rpc.Status statusOf(Throwable t) {
    com.google.rpc.Status status = StatusProto.fromThrowable(t);
    return status != null
        ? status
        : com.google.rpc.Status.newBuilder()
            .setCode(Status.fromThrowable(t).getCode().value())
            .setMessage(String.valueOf(t.getMessage()))
            .build();
  }
}
