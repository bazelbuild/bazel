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
package com.google.devtools.build.lib.server;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.io.IOException;
import java.net.SocketAddress;

/** The gRPC command server interface. */
@SkybridgeInterface
public interface GrpcCommandServer {
  /** The interface for responding to an RPC. */
  public interface Responder {
    /**
     * Sends a response.
     *
     * <p>May be called multiple times for a streaming RPC.
     *
     * @param response the response to send in wire format
     * @throws IOException if an I/O error occurs
     */
    void onNext(byte[] response) throws IOException;

    /**
     * Signals that the response is complete.
     *
     * <p>Must be called exactly once, after which {@link #onNext} must not be called again.
     *
     * @throws IOException if an I/O error occurs
     */
    void onCompleted() throws IOException;
  }

  /** The callback interface for handling RPCs. */
  public interface Callback {
    /**
     * Handles a Run RPC.
     *
     * @param request a serialized {@link com.google.devtools.build.lib.server.RunRequest}
     * @param responder a responder accepting serialized {@link
     *     com.google.devtools.build.lib.server.RunResponse}
     */
    void run(byte[] request, Responder responder);

    /**
     * Handles a Cancel RPC.
     *
     * @param request a serialized {@link com.google.devtools.build.lib.server.CancelRequest}
     * @param responder a responder accepting serialized {@link
     *     com.google.devtools.build.lib.server.CancelResponse}
     */
    void cancel(byte[] request, Responder responder);

    /**
     * Handles a Ping RPC.
     *
     * @param request a serialized {@link com.google.devtools.build.lib.server.PingRequest}
     * @param responder a responder accepting serialized {@link
     *     com.google.devtools.build.lib.server.PingResponse}
     */
    void ping(byte[] request, Responder responder);
  }

  /**
   * Starts serving.
   *
   * @param port the port to serve on (0 to pick an arbitrary unused port)
   * @param callback the callback to be called for incoming requests
   * @throws IOException if the server failed to bind to the given port
   * @return the local address the server is listening on
   */
  SocketAddress serve(int port, Callback callback) throws IOException;

  /**
   * Initiates an orderly shutdown in which preexisting calls continue but new calls are rejected.
   */
  void shutdown();

  /** Initiates a forceful shutdown in which preexisting and new calls will be rejected. */
  void shutdownNow();

  /**
   * Waits for the server to terminate.
   *
   * @throws InterruptedException if the thread is interrupted while waiting
   */
  void awaitTermination() throws InterruptedException;
}
