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
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.protobuf.ByteString;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.nio.channels.ServerSocketChannel;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;

/**
 * gRPC server class.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 */
public class GrpcServerImpl implements CommandServerGrpc.CommandServer, GrpcServer {
  /**
   * Factory class. Instantiated by reflection.
   */
  public static class Factory implements GrpcServer.Factory {
    @Override
    public GrpcServer create(CommandExecutor commandExecutor, Clock clock, int port,
      String outputBase) {
      return new GrpcServerImpl(commandExecutor, clock, port, outputBase);
    }
  }

  private enum StreamType {
    STDOUT,
    STDERR,
  }

  // TODO(lberki): Maybe we should implement line buffering?
  private class RpcOutputStream extends OutputStream {
    private final StreamObserver<RunResponse> observer;
    private final StreamType type;

    private RpcOutputStream(StreamObserver<RunResponse> observer, StreamType type) {
      this.observer = observer;
      this.type = type;
    }

    @Override
    public synchronized void write(byte[] b, int off, int inlen) {
      ByteString input = ByteString.copyFrom(b, off, inlen);
      RunResponse.Builder response = RunResponse
          .newBuilder()
          .setCookie(responseCookie);

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

  // These paths are all relative to the output base
  private static final String PORT_FILE = "server/grpc_port";
  private static final String REQUEST_COOKIE_FILE = "server/request_cookie";
  private static final String RESPONSE_COOKIE_FILE = "server/response_cookie";

  private final CommandExecutor commandExecutor;
  private final Clock clock;
  private final String outputBase;
  private final String requestCookie;
  private final String responseCookie;

  private Server server;
  private int port;  // mutable so that we can overwrite it if port 0 is passed in
  boolean serving;

  public GrpcServerImpl(CommandExecutor commandExecutor, Clock clock, int port,
      String outputBase) {
    this.commandExecutor = commandExecutor;
    this.clock = clock;
    this.outputBase = outputBase;
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

  public void serve() throws IOException {
    Preconditions.checkState(!serving);
    server = ServerBuilder.forPort(port)
        .addService(CommandServerGrpc.bindService(this))
        .build();

    server.start();
    serving = true;

    if (port == 0) {
      port = getActualServerPort();
    }

    writeFile(PORT_FILE, Integer.toString(port));
    writeFile(REQUEST_COOKIE_FILE, requestCookie);
    writeFile(RESPONSE_COOKIE_FILE, responseCookie);

  }

  private void writeFile(String path, String contents) throws IOException {
    OutputStreamWriter writer = new OutputStreamWriter(
        new FileOutputStream(new File(outputBase + "/" + path)), StandardCharsets.UTF_8);
    writer.write(contents);
    writer.close();
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

  public void terminate() {
    server.shutdownNow();
    // This is Uninterruptibles#callUninterruptibly. Calling that method properly is about the same
    // amount of code as implementing it ourselves.
    boolean interrupted = false;
    try {
      while (true) {
        try {
          server.awaitTermination();
          serving = false;
          return;
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
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

    OutErr rpcOutErr = OutErr.create(
        new RpcOutputStream(observer, StreamType.STDOUT),
        new RpcOutputStream(observer, StreamType.STDERR));

    int exitCode = commandExecutor.exec(
        request.getArgList(), rpcOutErr, clock.currentTimeMillis());

    RunResponse response = RunResponse.newBuilder()
        .setCookie(responseCookie)
        .setFinished(true)
        .setExitCode(exitCode)
        .build();

    observer.onNext(response);
    observer.onCompleted();
  }

  @Override
  public void ping(PingRequest pingRequest, StreamObserver<PingResponse> streamObserver) {
    Preconditions.checkState(serving);

    CommandProtos.PingResponse.Builder response = CommandProtos.PingResponse.newBuilder();
    if (pingRequest.getCookie().equals(requestCookie)) {
      response.setCookie(responseCookie);
    }

    streamObserver.onNext(response.build());
    streamObserver.onCompleted();
  }
}
