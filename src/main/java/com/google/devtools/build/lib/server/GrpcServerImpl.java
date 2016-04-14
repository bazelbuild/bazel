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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.nio.channels.ServerSocketChannel;

/**
 * gRPC server class.
 *
 * <p>Only this class should depend on gRPC so that we only need to exclude this during
 * bootstrapping.
 */
public class GrpcServerImpl implements CommandServerGrpc.CommandServer, GrpcServer {
  private static final String PORT_FILE = "server/grpc_port";  // relative to the output base
  private final CommandExecutor commandExecutor;
  private final Clock clock;
  private final File portFile;
  private Server server;
  private int port;  // mutable so that we can overwrite it if port 0 is passed in
  boolean serving;

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

  private GrpcServerImpl(CommandExecutor commandExecutor, Clock clock, int port,
      String outputBase) {
    this.commandExecutor = commandExecutor;
    this.clock = clock;
    this.port = port;
    this.portFile = new File(outputBase + "/" + PORT_FILE);
    this.serving = false;
  }

  public void serve() throws IOException {
    Preconditions.checkState(!serving);
    server = ServerBuilder.forPort(port)
        .addService(CommandServerGrpc.bindService(this))
        .build();

    server.start();
    if (port == 0) {
      port = getActualServerPort();
    }

    PrintWriter portWriter = new PrintWriter(portFile);
    portWriter.print(port);
    portWriter.close();
    serving = true;
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
    Preconditions.checkState(serving);
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
  public void run(CommandProtos.Request request, StreamObserver<CommandProtos.Response> observer) {
    Preconditions.checkState(serving);
    commandExecutor.exec(
        ImmutableList.of("version"), OutErr.SYSTEM_OUT_ERR, clock.currentTimeMillis());
    CommandProtos.Response response = CommandProtos.Response.newBuilder()
        .setNumber(request.getNumber() + 1)
        .build();
    observer.onNext(response);
    observer.onCompleted();
  }
}
