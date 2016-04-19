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

import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.nio.channels.ServerSocketChannel;
import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * gRPC server class.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 */
public class GrpcServerImpl extends RPCServer implements CommandServerGrpc.CommandServer {
  private class RunningCommand implements AutoCloseable {
    private final Thread thread;
    private final String id;

    private RunningCommand() {
      thread = Thread.currentThread();
      id = UUID.randomUUID().toString();
      synchronized (runningCommands) {
        runningCommands.put(id, this);
      }
    }

    @Override
    public void close() {
      synchronized (runningCommands) {
        runningCommands.remove(id);
      }
    }
  }

  /**
   * Factory class. Instantiated by reflection.
   */
  public static class Factory implements RPCServer.Factory {
    @Override
    public RPCServer create(CommandExecutor commandExecutor, Clock clock, int port,
      Path serverDirectory) throws IOException {
      return new GrpcServerImpl(commandExecutor, clock, port, serverDirectory);
    }
  }

  private enum StreamType {
    STDOUT,
    STDERR,
  }

  // TODO(lberki): Maybe we should implement line buffering?
  private class RpcOutputStream extends OutputStream {
    private final StreamObserver<RunResponse> observer;
    private final String commandId;
    private final StreamType type;

    private RpcOutputStream(
        StreamObserver<RunResponse> observer, String commandId, StreamType type) {
      this.observer = observer;
      this.commandId = commandId;
      this.type = type;
    }

    @Override
    public synchronized void write(byte[] b, int off, int inlen) {
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

      observer.onNext(response.build());
    }

    @Override
    public void write(int byteAsInt) throws IOException {
      byte b = (byte) byteAsInt; // make sure we work with bytes in comparisons
      write(new byte[] {b}, 0, 1);
    }
  }

  // These paths are all relative to the server directory
  private static final String PORT_FILE = "grpc_port";
  private static final String REQUEST_COOKIE_FILE = "request_cookie";
  private static final String RESPONSE_COOKIE_FILE = "response_cookie";

  private final Map<String, RunningCommand> runningCommands = new HashMap<>();
  private final CommandExecutor commandExecutor;
  private final Clock clock;
  private final Path serverDirectory;
  private final String requestCookie;
  private final String responseCookie;

  private Server server;
  private int port;  // mutable so that we can overwrite it if port 0 is passed in
  boolean serving;

  public GrpcServerImpl(CommandExecutor commandExecutor, Clock clock, int port,
      Path serverDirectory) throws IOException {
    super(serverDirectory);
    this.commandExecutor = commandExecutor;
    this.clock = clock;
    this.serverDirectory = serverDirectory;
    this.port = port;
    this.serving = false;

    SecureRandom random = new SecureRandom();
    requestCookie = generateCookie(random, 16);
    responseCookie = generateCookie(random, 16);
  }

  private static String generateCookie(SecureRandom random, int byteCount) {
    byte[] bytes = new byte[byteCount];
    random.nextBytes(bytes);
    StringBuilder result = new StringBuilder();
    for (byte b : bytes) {
      result.append(Integer.toHexString(((int) b) + 128));
    }

    return result.toString();
  }

  @Override
  public void serve() throws IOException {
    Preconditions.checkState(!serving);
    server = NettyServerBuilder.forAddress(new InetSocketAddress("localhost", port))
        .addService(CommandServerGrpc.bindService(this))
        .build();

    server.start();
    serving = true;

    if (port == 0) {
      port = getActualServerPort();
    }

    writeServerFile(PORT_FILE, Integer.toString(port));
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

  /**
   * Gets the server port the kernel bound our server to if port 0 was passed in.
   *
   * <p>The implementation is awful, but gRPC doesn't provide an official way to do this:
   * https://github.com/grpc/grpc-java/issues/72
   */
  private int getActualServerPort() {
    try {
      ServerSocketChannel channel =
          (ServerSocketChannel) getField(server, "transportServer", "channel", "ch");
      InetSocketAddress address = (InetSocketAddress) channel.getLocalAddress();
      return address.getPort();
    } catch (IllegalAccessException | NullPointerException | IOException e) {
      throw new IllegalStateException("Cannot read server socket address from gRPC");
    }
  }

  private static Object getField(Object instance, String... fieldNames)
    throws IllegalAccessException, NullPointerException {
    for (String fieldName : fieldNames) {
      Field field = null;
      for (Class<?> clazz = instance.getClass(); clazz != null; clazz = clazz.getSuperclass()) {
        try {
          field = clazz.getDeclaredField(fieldName);
          break;
        } catch (NoSuchFieldException e) {
          // Try again with the superclass
        }
      }
      field.setAccessible(true);
      instance = field.get(instance);
    }

    return instance;
  }

  @Override
  public void run(
      RunRequest request, StreamObserver<RunResponse> observer) {
    if (!request.getCookie().equals(requestCookie)) {
      observer.onNext(RunResponse.newBuilder()
          .setExitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode())
          .build());
      observer.onCompleted();
      return;
    }

    String commandId;
    int exitCode;
    try (RunningCommand command = new RunningCommand()) {
      commandId = command.id;
      OutErr rpcOutErr = OutErr.create(
          new RpcOutputStream(observer, command.id, StreamType.STDOUT),
          new RpcOutputStream(observer, command.id, StreamType.STDERR));

      exitCode = commandExecutor.exec(request.getArgList(), rpcOutErr, clock.currentTimeMillis());
    }

    // There is a chance that a cancel request comes in after commandExecutor#exec() has finished
    // and no one calls Thread.interrupted() to receive the interrupt. So we just reset the
    // interruption state here to make these cancel requests not have any effect outside of command
    // execution.
    Thread.interrupted();

    RunResponse response = RunResponse.newBuilder()
        .setCookie(responseCookie)
        .setCommandId(commandId)
        .setFinished(true)
        .setExitCode(exitCode)
        .build();

    observer.onNext(response);
    observer.onCompleted();

    if (commandExecutor.shutdown()) {
      server.shutdownNow();
    }
  }

  @Override
  public void ping(PingRequest pingRequest, StreamObserver<PingResponse> streamObserver) {
    Preconditions.checkState(serving);

    PingResponse.Builder response = PingResponse.newBuilder();
    if (pingRequest.getCookie().equals(requestCookie)) {
      response.setCookie(responseCookie);
    }

    streamObserver.onNext(response.build());
    streamObserver.onCompleted();
  }

  @Override
  public void cancel(CancelRequest request, StreamObserver<CancelResponse> streamObserver) {
    if (!request.getCookie().equals(requestCookie)) {
      streamObserver.onCompleted();
      return;
    }

    synchronized (runningCommands) {
      RunningCommand command = runningCommands.get(request.getCommandId());
      if (command != null) {
        command.thread.interrupt();
      }
    }

    streamObserver.onNext(CancelResponse.newBuilder().setCookie(responseCookie).build());
    streamObserver.onCompleted();
  }
}
