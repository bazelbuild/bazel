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

import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.CommandProtos.CancelResponse;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.PingResponse;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import java.io.IOException;
import java.net.SocketAddress;

/** The gRPC command server interface. */
public interface GrpcCommandServer {
  /** The interface for responding to an RPC. */
  public interface Responder<T> {
    /**
     * Sends a response.
     *
     * <p>May be called multiple times for a streaming RPC.
     *
     * @throws IOException if an I/O error occurs
     */
    void onNext(T value) throws IOException;

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
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void run(RunRequest request, Responder<RunResponse> responder);

    /**
     * Handles a Cancel RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void cancel(CancelRequest request, Responder<CancelResponse> responder);

    /**
     * Handles a Ping RPC.
     *
     * @param request the request to handle
     * @param responder the responder to send responses to
     */
    void ping(PingRequest request, Responder<PingResponse> responder);
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
