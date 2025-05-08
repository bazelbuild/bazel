// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdebug.server;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugRequest;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Manages the connection to and communication to/from the debugger client. Reading and writing are
 * internally synchronized by {@link DebugServerTransport}.
 */
final class DebugServerTransport {

  @SuppressWarnings("NonFinalStaticField")
  @VisibleForTesting
  static Consumer<Integer> onListenPortCallbackForTests = null;

  /** Sets up the server transport and blocks while waiting for an incoming connection. */
  static DebugServerTransport createAndWaitForClient(
      EventHandler eventHandler, ServerSocket serverSocket, boolean verboseLogging)
      throws IOException {
    // TODO(bazel-team): reject all connections after the first
    eventHandler.handle(Event.progress("Waiting for debugger..."));
    if (onListenPortCallbackForTests != null) {
      onListenPortCallbackForTests.accept(serverSocket.getLocalPort());
    }
    Socket clientSocket = serverSocket.accept();
    eventHandler.handle(Event.info("Debugger connection successfully established."));
    return new DebugServerTransport(
        eventHandler,
        serverSocket,
        clientSocket,
        clientSocket.getInputStream(),
        clientSocket.getOutputStream(),
        verboseLogging);
  }

  private final EventHandler eventHandler;
  private final ServerSocket serverSocket;
  private final Socket clientSocket;
  private final InputStream requestStream;
  private final OutputStream eventStream;
  private final boolean verboseLogging;

  private DebugServerTransport(
      EventHandler eventHandler,
      ServerSocket serverSocket,
      Socket clientSocket,
      InputStream requestStream,
      OutputStream eventStream,
      boolean verboseLogging) {
    this.eventHandler = eventHandler;
    this.serverSocket = serverSocket;
    this.clientSocket = clientSocket;
    this.requestStream = requestStream;
    this.eventStream = eventStream;
    this.verboseLogging = verboseLogging;
  }

  /**
   * Blocks waiting for a properly-formed client request. Returns null if the client connection is
   * closed.
   */
  @Nullable
  DebugRequest readClientRequest() {
    synchronized (requestStream) {
      try {
        DebugRequest request = DebugRequest.parseDelimitedFrom(requestStream);
        if (verboseLogging) {
          eventHandler.handle(Event.debug("Received debug client request:\n" + request));
        }
        return request;
      } catch (IOException e) {
        handleParsingError(e);
        return null;
      }
    }
  }

  private void handleParsingError(IOException e) {
    if (isClosed()) {
      // an IOException is expected when the client disconnects -- no need to log an error
      return;
    }
    String message = "Error parsing debug request: " + e.getMessage();
    postEvent(DebugEventHelper.error(message));
    eventHandler.handle(Event.error(message));
  }

  /** Posts a debug event. */
  void postEvent(DebugEvent event) {
    if (verboseLogging) {
      eventHandler.handle(Event.debug("Sending debug event:\n" + event));
    }
    synchronized (eventStream) {
      try {
        event.writeDelimitedTo(eventStream);
      } catch (IOException e) {
        eventHandler.handle(Event.error("Failed to send debug event to client: " + e.getMessage()));
      }
    }
  }

  void close() throws IOException {
    clientSocket.close();
    serverSocket.close();
  }

  boolean isClosed() {
    return serverSocket.isClosed() || clientSocket.isClosed();
  }
}
