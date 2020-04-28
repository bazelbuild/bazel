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
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos;
import com.google.devtools.build.lib.syntax.Debugger;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.List;
import javax.annotation.Nullable;

/** Manages the network socket and debugging state for threads running Starlark code. */
public final class StarlarkDebugServer implements Debugger {

  /**
   * Initializes debugging support, setting up any debugging-specific overrides, then opens the
   * debug server socket and blocks waiting for an incoming connection.
   *
   * @param port the port on which the server should listen for connections
   * @param verboseLogging if true, debug-level events will be logged
   * @throws IOException if an I/O error occurs while opening the socket or waiting for a connection
   */
  public static StarlarkDebugServer createAndWaitForConnection(
      EventHandler eventHandler, int port, boolean verboseLogging) throws IOException {
    ServerSocket serverSocket = new ServerSocket(port, /* backlog */ 1);
    return createAndWaitForConnection(eventHandler, serverSocket, verboseLogging);
  }

  /**
   * Initializes debugging support, setting up any debugging-specific overrides, then opens the
   * debug server socket and blocks waiting for an incoming connection.
   *
   * @param verboseLogging if true, debug-level events will be logged
   * @throws IOException if an I/O error occurs while waiting for a connection
   */
  @VisibleForTesting
  static StarlarkDebugServer createAndWaitForConnection(
      EventHandler eventHandler, ServerSocket serverSocket, boolean verboseLogging)
      throws IOException {
    DebugServerTransport transport =
        DebugServerTransport.createAndWaitForClient(eventHandler, serverSocket, verboseLogging);
    return new StarlarkDebugServer(eventHandler, transport, verboseLogging);
  }

  private final EventHandler eventHandler;
  /** Handles all thread-related state. */
  private final ThreadHandler threadHandler;
  /** The server socket for the debug server. */
  private final DebugServerTransport transport;

  private final boolean verboseLogging;

  private StarlarkDebugServer(
      EventHandler eventHandler, DebugServerTransport transport, boolean verboseLogging) {
    this.eventHandler = eventHandler;
    this.threadHandler = new ThreadHandler();
    this.transport = transport;
    this.verboseLogging = verboseLogging;
    listenForClientRequests();
  }

  /**
   * Starts a worker thread to listen for and handle incoming client requests, returning any
   * relevant responses.
   */
  private void listenForClientRequests() {
    Thread clientThread =
        new Thread(
            () -> {
              try {
                while (true) {
                  StarlarkDebuggingProtos.DebugRequest request = transport.readClientRequest();
                  if (request == null) {
                    return;
                  }
                  StarlarkDebuggingProtos.DebugEvent response = handleClientRequest(request);
                  if (response != null) {
                    transport.postEvent(response);
                  }
                }
              } catch (Throwable e) {
                if (!transport.isClosed()) {
                  eventHandler.handle(
                      Event.error(
                          "Debug server listener thread died: "
                              + Throwables.getStackTraceAsString(e)));
                }
              } finally {
                eventHandler.handle(
                    Event.info(
                        "Debug server listener thread closed; shutting down debug server and "
                            + "resuming all threads"));
                close();
              }
            });

    clientThread.setDaemon(true);
    clientThread.start();
  }

  @Override
  public void close() {
    try {
      if (verboseLogging) {
        eventHandler.handle(Event.debug("Closing debug server"));
      }
      transport.close();

    } catch (IOException e) {
      eventHandler.handle(
          Event.error(
              "Error shutting down the debug server: " + Throwables.getStackTraceAsString(e)));
    } finally {
      // ensure no threads are left paused, otherwise the build command will never complete
      threadHandler.resumeAllThreads();
    }
  }

  /**
   * Called by the interpreter before execution of the code at the specified location. Pauses the
   * execution of the current thread if there are conditions that should cause it to be paused, such
   * as a breakpoint being reached.
   *
   * @param location the location of the statement or expression currently being executed
   */
  @Override
  public void before(StarlarkThread thread, Location location) {
    if (!transport.isClosed()) {
      threadHandler.pauseIfNecessary(thread, location, transport);
    }
  }

  /** Handles a request from the client, and returns the response, where relevant. */
  @Nullable
  private StarlarkDebuggingProtos.DebugEvent handleClientRequest(
      StarlarkDebuggingProtos.DebugRequest request) {
    long sequenceNumber = request.getSequenceNumber();
    try {
      switch (request.getPayloadCase()) {
        case START_DEBUGGING:
          threadHandler.resumeAllThreads();
          return DebugEventHelper.startDebuggingResponse(sequenceNumber);
        case LIST_FRAMES:
          return listFrames(sequenceNumber, request.getListFrames());
        case SET_BREAKPOINTS:
          return setBreakpoints(sequenceNumber, request.getSetBreakpoints());
        case CONTINUE_EXECUTION:
          return continueExecution(sequenceNumber, request.getContinueExecution());
        case PAUSE_THREAD:
          return pauseThread(sequenceNumber, request.getPauseThread());
        case EVALUATE:
          return evaluate(sequenceNumber, request.getEvaluate());
        case GET_CHILDREN:
          return getChildren(sequenceNumber, request.getGetChildren());
        case PAYLOAD_NOT_SET:
          DebugEventHelper.error(sequenceNumber, "No request payload found");
      }
      return DebugEventHelper.error(
          sequenceNumber, "Unhandled request type: " + request.getPayloadCase());
    } catch (DebugRequestException e) {
      return DebugEventHelper.error(sequenceNumber, e.getMessage());
    }
  }

  /** Handles a {@code ListFramesRequest} and returns its response. */
  private StarlarkDebuggingProtos.DebugEvent listFrames(
      long sequenceNumber, StarlarkDebuggingProtos.ListFramesRequest request)
      throws DebugRequestException {
    List<StarlarkDebuggingProtos.Frame> frames = threadHandler.listFrames(request.getThreadId());
    return DebugEventHelper.listFramesResponse(sequenceNumber, frames);
  }

  /** Handles a {@code SetBreakpointsRequest} and returns its response. */
  private StarlarkDebuggingProtos.DebugEvent setBreakpoints(
      long sequenceNumber, StarlarkDebuggingProtos.SetBreakpointsRequest request) {
    threadHandler.setBreakpoints(request.getBreakpointList());
    return DebugEventHelper.setBreakpointsResponse(sequenceNumber);
  }

  /** Handles a {@code EvaluateRequest} and returns its response. */
  private StarlarkDebuggingProtos.DebugEvent evaluate(
      long sequenceNumber, StarlarkDebuggingProtos.EvaluateRequest request)
      throws DebugRequestException {
    return DebugEventHelper.evaluateResponse(
        sequenceNumber, threadHandler.evaluate(request.getThreadId(), request.getStatement()));
  }

  /** Handles a {@code GetChildrenRequest} and returns its response. */
  private StarlarkDebuggingProtos.DebugEvent getChildren(
      long sequenceNumber, StarlarkDebuggingProtos.GetChildrenRequest request)
      throws DebugRequestException {
    return DebugEventHelper.getChildrenResponse(
        sequenceNumber,
        threadHandler.getChildrenForValue(request.getThreadId(), request.getValueId()));
  }

  /** Handles a {@code ContinueExecutionRequest} and returns its response. */
  private StarlarkDebuggingProtos.DebugEvent continueExecution(
      long sequenceNumber, StarlarkDebuggingProtos.ContinueExecutionRequest request)
      throws DebugRequestException {
    long threadId = request.getThreadId();
    if (threadId == 0) {
      threadHandler.resumeAllThreads();
      return DebugEventHelper.continueExecutionResponse(sequenceNumber);
    }
    threadHandler.resumeThread(threadId, request.getStepping());
    return DebugEventHelper.continueExecutionResponse(sequenceNumber);
  }

  private StarlarkDebuggingProtos.DebugEvent pauseThread(
      long sequenceNumber, StarlarkDebuggingProtos.PauseThreadRequest request)
      throws DebugRequestException {
    long threadId = request.getThreadId();
    if (threadId == 0) {
      threadHandler.pauseAllThreads();
    } else {
      threadHandler.pauseThread(threadId);
    }
    return DebugEventHelper.pauseThreadResponse(sequenceNumber);
  }
}
