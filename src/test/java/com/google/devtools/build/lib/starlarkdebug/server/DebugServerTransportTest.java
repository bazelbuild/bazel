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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.ContinueExecutionResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugRequest;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.StartDebuggingRequest;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DebugServerTransport}. */
@RunWith(JUnit4.class)
public class DebugServerTransportTest {

  private static final ExecutorService executor = Executors.newFixedThreadPool(1);

  private final EventCollectionApparatus events =
      new EventCollectionApparatus(EventKind.ALL_EVENTS);

  /** A simple debug client for testing purposes. */
  private static class MockDebugClient {

    Socket clientSocket;

    void connect(Duration timeout, ServerSocket serverSocket) {
      long startTimeMillis = System.currentTimeMillis();
      IOException exception = null;
      while (System.currentTimeMillis() - startTimeMillis < timeout.toMillis()) {
        try {
          clientSocket = new Socket();
          clientSocket.connect(
              new InetSocketAddress(serverSocket.getInetAddress(), serverSocket.getLocalPort()),
              100);
          return;
        } catch (IOException e) {
          exception = e;
        }
      }
      throw new RuntimeException("Couldn't connect to the debug server", exception);
    }

    List<DebugEvent> readEvents() throws Exception {
      List<DebugEvent> events = new ArrayList<>();
      while (clientSocket.getInputStream().available() != 0) {
        events.add(DebugEvent.parseDelimitedFrom(clientSocket.getInputStream()));
      }
      return events;
    }

    void sendRequest(DebugRequest request) throws IOException {
      request.writeDelimitedTo(clientSocket.getOutputStream());
    }
  }

  private static ServerSocket getServerSocket() throws IOException {
    // For reasons only Apple knows, you cannot bind to IPv4-localhost when you run in a sandbox
    // that only allows loopback traffic, but binding to IPv6-localhost works fine. This would
    // however break on systems that don't support IPv6. So what we'll do is to try to bind to IPv6
    // and if that fails, try again with IPv4.
    try {
      return new ServerSocket(0, 1, InetAddress.getByName("[::1]"));
    } catch (SocketException e) {
      return new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"));
    }
  }

  @Test
  public void testConnectAndReceiveRequest() throws Exception {
    ServerSocket serverSocket = getServerSocket();
    Future<DebugServerTransport> future =
        executor.submit(
            () ->
                DebugServerTransport.createAndWaitForClient(
                    events.reporter(), serverSocket, false));
    MockDebugClient client = new MockDebugClient();
    client.connect(Duration.ofSeconds(10), serverSocket);

    DebugServerTransport serverTransport = future.get(10, TimeUnit.SECONDS);
    assertThat(serverTransport).isNotNull();
    DebugRequest request =
        DebugRequest.newBuilder()
            .setSequenceNumber(10)
            .setStartDebugging(StartDebuggingRequest.getDefaultInstance())
            .build();
    client.sendRequest(request);

    assertThat(serverTransport.readClientRequest()).isEqualTo(request);
    serverTransport.close();
  }

  @Test
  public void testConnectAndPostEvent() throws Exception {
    ServerSocket serverSocket = getServerSocket();
    Future<DebugServerTransport> future =
        executor.submit(
            () ->
                DebugServerTransport.createAndWaitForClient(
                    events.reporter(), serverSocket, false));
    MockDebugClient client = new MockDebugClient();
    client.connect(Duration.ofSeconds(10), serverSocket);

    DebugServerTransport serverTransport = future.get(10, TimeUnit.SECONDS);
    assertThat(serverTransport).isNotNull();
    DebugEvent event =
        DebugEvent.newBuilder()
            .setSequenceNumber(10)
            .setContinueExecution(ContinueExecutionResponse.getDefaultInstance())
            .build();
    serverTransport.postEvent(event);

    assertThat(client.readEvents()).containsExactly(event);
    serverTransport.close();
  }
}
