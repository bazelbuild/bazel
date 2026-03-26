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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.CommandDispatcher;
import com.google.devtools.build.lib.runtime.CommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.CommandDispatcher.UiVerbosity;
import com.google.devtools.build.lib.runtime.SafeRequestLogging;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.CommandManager.RunningCommand;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.server.CommandProtos.ServerInfo;
import com.google.devtools.build.lib.server.CommandProtos.StartupOption;
import com.google.devtools.build.lib.server.FailureDetails.Command;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Filesystem;
import com.google.devtools.build.lib.server.FailureDetails.Filesystem.Code;
import com.google.devtools.build.lib.server.FailureDetails.GrpcServer;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.CommandExtensionReporter;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.InvocationPolicyParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Inet4Address;
import java.net.Inet6Address;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Manages the client request/response loop.
 *
 * <p>In addition to the request threads (managed by {@link GrpcCommandServer}), we maintain one
 * extra thread for handling the server timeout, and an interrupt watcher thread is started for each
 * interrupt request that logs if it takes too long to take effect.
 *
 * <p>Each running RPC has a UUID associated with it that is used to identify it when a client wants
 * to cancel it. Cancellation is done by the client sending the server a Cancel RPC, which results
 * in the main thread of the command being interrupted.
 */
public class CommandServer implements GrpcCommandServer.Callback {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static CommandServer create(
      GrpcCommandServer grpcCommandServer,
      CommandDispatcher dispatcher,
      ShutdownHooks shutdownHooks,
      PidFileWatcher pidFileWatcher,
      Clock clock,
      int port,
      Path serverDirectory,
      int serverPid,
      int maxIdleSeconds,
      boolean shutdownOnLowSysMem,
      boolean idleServerTasks,
      @Nullable String slowInterruptMessageSuffix) {
    SecureRandom random = new SecureRandom();
    return new CommandServer(
        grpcCommandServer,
        dispatcher,
        shutdownHooks,
        pidFileWatcher,
        clock,
        port,
        generateCookie(random, 16),
        generateCookie(random, 16),
        serverDirectory,
        serverPid,
        maxIdleSeconds,
        shutdownOnLowSysMem,
        idleServerTasks,
        slowInterruptMessageSuffix);
  }

  @VisibleForTesting
  enum StreamType {
    STDOUT,
    STDERR,
  }

  /** Command extension reporter that packs the protobuf into a RunResponse and sends it. */
  private static class RpcCommandExtensionReporter implements CommandExtensionReporter {

    // Store commandId and responseCookie as ByteStrings to avoid String -> UTF8 bytes conversion
    // for each serialized chunk of output.
    private final ByteString commandIdBytes;
    private final ByteString responseCookieBytes;

    private final GrpcCommandServer.Responder<RunResponse> responder;

    RpcCommandExtensionReporter(
        String commandId,
        String responseCookie,
        GrpcCommandServer.Responder<RunResponse> responder) {
      this.commandIdBytes = ByteString.copyFromUtf8(commandId);
      this.responseCookieBytes = ByteString.copyFromUtf8(responseCookie);
      this.responder = responder;
    }

    @Override
    public synchronized void report(Any commandExtension) throws IOException {
      responder.onNext(
          RunResponse.newBuilder()
              .setCookieBytes(responseCookieBytes)
              .setCommandIdBytes(commandIdBytes)
              .setStandardOutput(ByteString.EMPTY)
              .addCommandExtensions(commandExtension)
              .build());
    }
  }

  /**
   * An output stream that forwards the data written to it over the gRPC command stream.
   *
   * <p>Note that wraping this class with a {@code Channel} can cause a deadlock if there is an
   * {@link OutputStream} in between that synchronizes both on {@code #close()} and {@code #write()}
   * because then if an interrupt happens in {@code FlowControl#onNext}, the thread on which {@code
   * interrupt()} was called will wait until the {@code Channel} closes itself while holding a lock
   * for interrupting the thread on which {@code FlowControl#onNext} is being executed and that
   * thread will hold a lock that is needed for the {@code Channel} to be closed and call {@code
   * interrupt()} in {@code FlowControl#onNext}, which will in turn try to acquire the interrupt
   * lock.
   */
  private static class RpcOutputStream extends OutputStream {
    private static final int CHUNK_SIZE = 8192;

    // Store commandId and responseCookie as ByteStrings to avoid String -> UTF8 bytes conversion
    // for each serialized chunk of output.
    private final ByteString commandIdBytes;
    private final ByteString responseCookieBytes;

    private final StreamType type;
    private final GrpcCommandServer.Responder<RunResponse> responder;

    RpcOutputStream(
        String commandId,
        String responseCookie,
        StreamType type,
        GrpcCommandServer.Responder<RunResponse> responder) {
      this.commandIdBytes = ByteString.copyFromUtf8(commandId);
      this.responseCookieBytes = ByteString.copyFromUtf8(responseCookie);
      this.type = type;
      this.responder = responder;
    }

    @Override
    public void write(byte[] b, int off, int inlen) throws IOException {
      for (int i = 0; i < inlen; i += CHUNK_SIZE) {
        ByteString input = ByteString.copyFrom(b, off + i, Math.min(CHUNK_SIZE, inlen - i));
        RunResponse.Builder response =
            RunResponse.newBuilder()
                .setCookieBytes(responseCookieBytes)
                .setCommandIdBytes(commandIdBytes);

        switch (type) {
          case STDOUT -> response.setStandardOutput(input);
          case STDERR -> response.setStandardError(input);
        }

        try {
          // This can block waiting for the client to read the available data.
          responder.onNext(response.build());
        } catch (IOException e) {
          // I am not sure whether there are any circumstances under which this call could throw an
          // exception, but I'd rather it be logged than that we crash silently. The documentation
          // only says that onNext does not throw a CancelledException if the stream is canceled,
          // but otherwise does not say anything about exceptions that can be thrown from onNext.
          // Note that Blaze redirects System.{out,err} to this output stream, so attempting to call
          // printStackTrace() from here could go into an infinite loop.
          BugReport.sendBugReport(e);
          Thread.currentThread().interrupt();
        }
      }
    }

    @Override
    public void write(int byteAsInt) throws IOException {
      write(new byte[] {(byte) byteAsInt}, 0, 1);
    }
  }

  // These paths are all relative to the server directory
  private static final String PORT_FILE = "command_port";
  private static final String REQUEST_COOKIE_FILE = "request_cookie";
  private static final String RESPONSE_COOKIE_FILE = "response_cookie";
  private static final String SERVER_INFO_FILE = "server_info.rawproto";

  private final GrpcCommandServer grpcCommandServer;
  private final CommandManager commandManager;
  private final CommandDispatcher dispatcher;
  private final ShutdownHooks shutdownHooks;
  private final Clock clock;
  private final Path serverDirectory;
  private final String requestCookie;
  private final String responseCookie;
  private final int maxIdleSeconds;
  private final boolean shutdownOnLowSysMem;
  private final PidFileWatcher pidFileWatcher;
  private final int serverPid;
  private final int port;

  @VisibleForTesting
  CommandServer(
      GrpcCommandServer grpcCommandServer,
      CommandDispatcher dispatcher,
      ShutdownHooks shutdownHooks,
      PidFileWatcher pidFileWatcher,
      Clock clock,
      int port,
      String requestCookie,
      String responseCookie,
      Path serverDirectory,
      int serverPid,
      int maxIdleSeconds,
      boolean shutdownOnLowSysMem,
      boolean doIdleServerTasks,
      @Nullable String slowInterruptMessageSuffix) {
    this.grpcCommandServer = grpcCommandServer;
    this.dispatcher = dispatcher;
    this.shutdownHooks = shutdownHooks;
    this.pidFileWatcher = pidFileWatcher;
    this.clock = clock;
    this.port = port;
    this.requestCookie = requestCookie;
    this.responseCookie = responseCookie;
    this.serverDirectory = serverDirectory;
    this.serverPid = serverPid;
    this.maxIdleSeconds = maxIdleSeconds;
    this.shutdownOnLowSysMem = shutdownOnLowSysMem;

    commandManager = new CommandManager(doIdleServerTasks, slowInterruptMessageSuffix);
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

  /**
   * This is called when the server is shut down as a result of a "clean --expunge".
   *
   * <p>In this case, no files should be deleted on shutdown hooks, since clean also deletes the
   * lock file, and there is a small possibility of the following sequence of events:
   *
   * <ol>
   *   <li>Client 1 runs "blaze clean --expunge"
   *   <li>Client 2 runs a command and waits for client 1 to finish
   *   <li>The clean command deletes everything including the lock file
   *   <li>Client 2 starts running and since the output base is empty, starts up a new server, which
   *       creates its own socket and PID files
   *   <li>The server used by client runs its shutdown hooks, deleting the PID files created by the
   *       new server
   * </ol>
   *
   * It also disables the "die when the PID file changes" handler so that it doesn't kill the server
   * while the "clean --expunge" command is running.
   */
  public void prepareForAbruptShutdown() {
    shutdownHooks.disable();
    pidFileWatcher.signalShutdown();
  }

  /** Interrupts (cancels) in-flight commands. */
  public void interrupt() {
    commandManager.interruptInflightCommands();
  }

  /**
   * Starts serving, writes the server status files, and blocks until the shutdown command is
   * received.
   */
  public void serveAndAwaitTermination() throws AbruptExitException {
    SocketAddress address;
    try {
      address = serve();
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              createFailureDetail(e.getMessage(), GrpcServer.Code.SERVER_BIND_FAILURE)),
          e);
    }

    if (maxIdleSeconds > 0) {
      Thread timeoutAndMemoryCheckingThread =
          new Thread(
              new ServerWatcherRunnable(
                  grpcCommandServer, maxIdleSeconds, shutdownOnLowSysMem, commandManager));
      timeoutAndMemoryCheckingThread.setName("grpc-timeout-and-memory");
      timeoutAndMemoryCheckingThread.setDaemon(true);
      timeoutAndMemoryCheckingThread.start();
    }

    writeServerStatusFiles(address);

    try {
      awaitTermination();
    } catch (InterruptedException e) {
      // TODO(lberki): Handle SIGINT in a reasonable way
      throw new IllegalStateException(e);
    }
  }

  @VisibleForTesting
  @CanIgnoreReturnValue
  SocketAddress serve() throws IOException {
    return grpcCommandServer.serve(port, this);
  }

  @VisibleForTesting
  void shutdown() {
    grpcCommandServer.shutdown();
  }

  @VisibleForTesting
  void shutdownNow() {
    grpcCommandServer.shutdownNow();
  }

  @VisibleForTesting
  void awaitTermination() throws InterruptedException {
    grpcCommandServer.awaitTermination();
  }

  private void writeServerStatusFiles(SocketAddress address) throws AbruptExitException {
    String addressString = formatAddress(address);

    writeServerFile(PORT_FILE, addressString);
    writeServerFile(REQUEST_COOKIE_FILE, requestCookie);
    writeServerFile(RESPONSE_COOKIE_FILE, responseCookie);

    ServerInfo info =
        ServerInfo.newBuilder()
            .setPid(serverPid)
            .setAddress(addressString)
            .setRequestCookie(requestCookie)
            .setResponseCookie(responseCookie)
            .build();

    // Write then mv so the user never sees incomplete contents.
    Path serverInfoTmpFile = serverDirectory.getChild(SERVER_INFO_FILE + ".tmp");
    try {
      try (OutputStream out = serverInfoTmpFile.getOutputStream()) {
        info.writeTo(out);
      }
      Path serverInfoFile = serverDirectory.getChild(SERVER_INFO_FILE);
      serverInfoTmpFile.renameTo(serverInfoFile);
      shutdownHooks.deleteAtExit(serverInfoFile);
    } catch (IOException e) {
      throw createFilesystemFailureException("Failed to write server info file", e);
    }
  }

  private String formatAddress(SocketAddress address) {
    if (address instanceof InetSocketAddress inetAddress) {
      if (inetAddress.getAddress() instanceof Inet4Address inet4Addr) {
        return inet4Addr.getHostAddress() + ":" + inetAddress.getPort();
      } else if (inetAddress.getAddress() instanceof Inet6Address inet6Addr) {
        return "[" + inet6Addr.getHostAddress() + "]:" + inetAddress.getPort();
      }
    }
    // Can only happen in tests using an in-memory implementation; representation doesn't matter.
    return address.toString();
  }

  private void writeServerFile(String name, String contents) throws AbruptExitException {
    Path file = serverDirectory.getChild(name);
    try {
      FileSystemUtils.writeContentAsLatin1(file, contents);
    } catch (IOException e) {
      throw createFilesystemFailureException("Server file (" + file + ") write failed", e);
    }
    shutdownHooks.deleteAtExit(file);
  }

  @Override
  public void run(RunRequest request, GrpcCommandServer.Responder<RunResponse> responder) {
    boolean badCookie = !isValidRequestCookie(request.getCookie());
    if (badCookie || request.getClientDescription().isEmpty()) {
      try {
        FailureDetail failureDetail =
            badCookie
                ? createFailureDetail("Invalid RunRequest: bad cookie", GrpcServer.Code.BAD_COOKIE)
                : createFailureDetail(
                    "Invalid RunRequest: no client description",
                    GrpcServer.Code.NO_CLIENT_DESCRIPTION);
        responder.onNext(
            RunResponse.newBuilder()
                .setFinished(true)
                .setExitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode())
                .setFailureDetail(failureDetail)
                .build());
        responder.onCompleted();
      } catch (IOException e) {
        logger.atInfo().withCause(e).log("Error while sending RunResponse");
      }
      return;
    }

    String commandId;
    BlazeCommandResult result;

    // TODO(b/63925394): This information needs to be passed to the GotOptionsEvent, which does not
    // currently have the explicit startup options. See Improved Command Line Reporting design doc
    // for details.
    // Convert the startup options record to Java strings, source first.
    ImmutableList.Builder<Pair<String, String>> startupOptions = ImmutableList.builder();
    for (StartupOption option : request.getStartupOptionsList()) {
      // UTF-8 won't do because we want to be able to pass arbitrary binary strings.
      startupOptions.add(
          new Pair<>(
              platformBytesToInternalString(option.getSource()),
              platformBytesToInternalString(option.getOption())));
    }

    commandManager.preemptEligibleCommands();

    try (RunningCommand command =
        request.getPreemptible()
            ? commandManager.createPreemptibleCommand()
            : commandManager.createCommand()) {
      commandId = command.getId();

      try {
        // Send the client the command id as soon as we know it.
        responder.onNext(
            RunResponse.newBuilder().setCookie(responseCookie).setCommandId(commandId).build());
      } catch (IOException e) {
        logger.atInfo().withCause(e).log("Error while sending initial RunResponse");
      }

      OutErr rpcOutErr =
          OutErr.create(
              new RpcOutputStream(command.getId(), responseCookie, StreamType.STDOUT, responder),
              new RpcOutputStream(command.getId(), responseCookie, StreamType.STDERR, responder));

      try {
        // Transform args into Bazel's internal string representation.
        ImmutableList<String> args =
            request.getArgList().stream()
                .map(CommandServer::platformBytesToInternalString)
                .collect(ImmutableList.toImmutableList());

        InvocationPolicy policy = InvocationPolicyParser.parsePolicy(request.getInvocationPolicy());
        logger.atInfo().log("Executing command %s", SafeRequestLogging.getRequestLogString(args));
        result =
            dispatcher.exec(
                policy,
                args,
                rpcOutErr,
                request.getBlockForLock() ? LockingMode.WAIT : LockingMode.ERROR_OUT,
                request.getQuiet() ? UiVerbosity.QUIET : UiVerbosity.NORMAL,
                request.getClientDescription(),
                clock.currentTimeMillis(),
                Optional.of(startupOptions.build()),
                commandManager::getIdleTaskResults,
                request.getCommandExtensionsList(),
                new RpcCommandExtensionReporter(command.getId(), responseCookie, responder));
      } catch (OptionsParsingException e) {
        rpcOutErr.printErrLn(e.getMessage());
        result =
            BlazeCommandResult.detailedExitCode(
                DetailedExitCode.of(
                    FailureDetail.newBuilder()
                        .setMessage("Invocation policy parsing failed: " + e.getMessage())
                        .setCommand(
                            Command.newBuilder()
                                .setCode(Command.Code.INVOCATION_POLICY_PARSE_FAILURE))
                        .build()));
      }

      // Record tasks to be run by IdleTaskManager. This is triggered in RunningCommand#close()
      // (as a Closeable), as we go out of scope immediately after this.
      command.setIdleTasks(result.getIdleTasks());
    } catch (InterruptedException e) {
      result =
          BlazeCommandResult.detailedExitCode(
              InterruptedFailureDetails.detailedExitCode("Command dispatch interrupted"));
      commandId = ""; // The default value, the client will ignore it
    }
    RunResponse.Builder response =
        RunResponse.newBuilder()
            .setCookie(responseCookie)
            .setCommandId(commandId)
            .setFinished(true)
            .setTerminationExpected(result.shutdown());

    if (result.getExecRequest() != null) {
      response.setExitCode(0);
      response.setExecRequest(result.getExecRequest());
    } else {
      response.setExitCode(result.getExitCode().getNumericExitCode());
      if (result.getFailureDetail() != null) {
        response.setFailureDetail(result.getFailureDetail());
      }
    }

    try {
      responder.onNext(response.addAllCommandExtensions(result.getResponseExtensions()).build());
      responder.onCompleted();

    } catch (IOException e) {
      logger.atInfo().withCause(e).log("Error while sending RunResponse");
    }
    if (result.shutdown()) {
      grpcCommandServer.shutdown();
    }
  }

  @Override
  public void ping(PingRequest request, GrpcCommandServer.Responder<PingResponse> responder) {
    logger.atInfo().log("Got PingRequest");
    try (RunningCommand command = commandManager.createCommand()) {
      PingResponse.Builder response = PingResponse.newBuilder();
      if (isValidRequestCookie(request.getCookie())) {
        response.setCookie(responseCookie);
      }
      responder.onNext(response.build());
      responder.onCompleted();
    } catch (IOException e) {
      // There is no one to report the failure to.
      logger.atInfo().withCause(e).log("Error while sending PingResponse");
    }
  }

  @Override
  public void cancel(CancelRequest request, GrpcCommandServer.Responder<CancelResponse> responder) {
    logger.atInfo().log("Got CancelRequest for command id %s", request.getCommandId());
    try {
      if (isValidRequestCookie(request.getCookie())) {
        commandManager.doCancel(request);
        responder.onNext(CancelResponse.newBuilder().setCookie(responseCookie).build());
      }
      responder.onCompleted();
    } catch (IOException e) {
      // There is no one to report the failure to.
      logger.atInfo().withCause(e).log("Error while sending CancelResponse");
    }
  }

  /**
   * Returns whether or not the provided cookie is valid for this server using a constant-time
   * comparison in order to guard against timing attacks.
   */
  private boolean isValidRequestCookie(String incomingRequestCookie) {
    // Note that cookie file was written as latin-1, so use that here.
    return MessageDigest.isEqual(
        incomingRequestCookie.getBytes(StandardCharsets.ISO_8859_1),
        requestCookie.getBytes(StandardCharsets.ISO_8859_1));
  }

  private static AbruptExitException createFilesystemFailureException(
      String message, IOException e) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(
                    message + (Strings.isNullOrEmpty(e.getMessage()) ? "" : ": " + e.getMessage()))
                .setFilesystem(Filesystem.newBuilder().setCode(Code.SERVER_FILE_WRITE_FAILURE))
                .build()),
        e);
  }

  private static FailureDetail createFailureDetail(String message, GrpcServer.Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setGrpcServer(GrpcServer.newBuilder().setCode(detailedCode))
        .build();
  }

  private static String platformBytesToInternalString(ByteString bytes) {
    return bytes.toString(ISO_8859_1);
  }
}
