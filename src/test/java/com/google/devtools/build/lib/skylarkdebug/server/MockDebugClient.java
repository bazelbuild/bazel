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

package com.google.devtools.build.lib.skylarkdebug.server;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.DebugRequest;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** A basic implementation of a skylark debugging client, for use in integration tests. */
class MockDebugClient {

  private static final int RESPONSE_TIMEOUT_MILLIS = 10000;
  private static final ExecutorService readTaskExecutor = Executors.newFixedThreadPool(1);

  private Socket clientSocket;

  final List<DebugEvent> unnumberedEvents = new ArrayList<>();
  final Map<Long, DebugEvent> responses = new HashMap<>();

  private Future<?> readTask;

  /** Connects to the debug server, and starts listening for events. */
  void connect(ServerSocket serverSocket, Duration timeout) {
    long startTimeMillis = System.currentTimeMillis();
    IOException exception = null;
    while (System.currentTimeMillis() - startTimeMillis < timeout.toMillis()) {
      try {
        clientSocket = new Socket();
        clientSocket.connect(
            new InetSocketAddress(serverSocket.getInetAddress(), serverSocket.getLocalPort()), 100);
        readTask =
            readTaskExecutor.submit(
                () -> {
                  while (true) {
                    eventReceived(DebugEvent.parseDelimitedFrom(clientSocket.getInputStream()));
                  }
                });
        return;
      } catch (IOException e) {
        exception = e;
      }
    }
    throw new RuntimeException("Couldn't connect to the debug server", exception);
  }

  void close() throws IOException {
    if (clientSocket != null) {
      clientSocket.close();
    }
    if (readTask != null) {
      readTask.cancel(true);
    }
  }

  /**
   * Blocks waiting for an unnumbered event (not a direct response to a request). Returns null if no
   * event arrives before the timeout.
   */
  @Nullable
  DebugEvent waitForEvent(Predicate<DebugEvent> predicate, Duration timeout) {
    waitForEvents(list -> list.stream().anyMatch(predicate), timeout);
    return unnumberedEvents.stream().filter(predicate).findFirst().orElse(null);
  }

  /**
   * Blocks waiting for a condition on all unnumbered events to be satisfied. Returns true if the
   * condition was satisfied before the timeout.
   */
  boolean waitForEvents(Predicate<List<DebugEvent>> predicate, Duration timeout) {
    long startTime = System.currentTimeMillis();
    synchronized (unnumberedEvents) {
      while (!predicate.test(ImmutableList.copyOf(unnumberedEvents))
          && System.currentTimeMillis() - startTime < timeout.toMillis()) {
        try {
          unnumberedEvents.wait(timeout.toMillis());
        } catch (InterruptedException e) {
          throw new AssertionError(e);
        }
      }
    }
    return predicate.test(ImmutableList.copyOf(unnumberedEvents));
  }

  /**
   * Sends a {@link DebugRequest} to the server, and blocks waiting for a response.
   *
   * @return the {@link DebugEvent} response from the server, or null if no response was received.
   */
  @Nullable
  DebugEvent sendRequestAndWaitForResponse(DebugRequest request) throws IOException {
    request.writeDelimitedTo(clientSocket.getOutputStream());
    clientSocket.getOutputStream().flush();
    return waitForResponse(request.getSequenceNumber());
  }

  private void eventReceived(DebugEvent event) {
    if (event.getSequenceNumber() == 0) {
      synchronized (unnumberedEvents) {
        unnumberedEvents.add(event);
        unnumberedEvents.notifyAll();
      }
      return;
    }
    synchronized (responses) {
      DebugEvent existing = responses.put(event.getSequenceNumber(), event);
      if (existing != null) {
        throw new AssertionError(
            "There's already an event in the response queue corresponding to sequence number "
                + event.getSequenceNumber());
      }
      responses.notifyAll();
    }
  }

  /**
   * Wait for a response from the debug server. Returns null if no response was received, or this
   * thread was interrupted.
   */
  @Nullable
  private DebugEvent waitForResponse(long sequence) {
    DebugEvent response = null;
    long startTime = System.currentTimeMillis();
    synchronized (responses) {
      while (response == null && shouldWaitForResponse(startTime)) {
        try {
          responses.wait(1000);
        } catch (InterruptedException e) {
          throw new AssertionError(e);
        }
        response = responses.remove(sequence);
      }
    }
    return response;
  }

  private boolean shouldWaitForResponse(long startTime) {
    return clientSocket.isConnected()
        && !readTask.isDone()
        && System.currentTimeMillis() - startTime < RESPONSE_TIMEOUT_MILLIS;
  }
}
