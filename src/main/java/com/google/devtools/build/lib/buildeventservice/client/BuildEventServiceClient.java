// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice.client;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import io.grpc.Status;
import io.grpc.StatusException;

/** Interface used to abstract the Stubby and gRPC client implementations. */
public interface BuildEventServiceClient {

  /** Callback for ACKed build events. */
  @FunctionalInterface
  interface AckCallback {

    /**
     * Called whenever an ACK from the BES server is received. ACKs are expected to be received in
     * sequence. Implementations need to be thread-safe.
     */
    void apply(PublishBuildToolEventStreamResponse ack);
  }

  /** A handle to a bidirectional stream. */
  interface StreamContext {

    /**
     * The completed status of the stream. The future will never fail, but in case of error will
     * contain a corresponding status.
     */
    ListenableFuture<Status> getStatus();

    /**
     * Sends an event over the currently open stream. In case of error, this method will fail
     * silently and report the error via the {@link ListenableFuture} returned by {@link
     * #getStatus()}.
     *
     * <p>This method may block due to flow control.
     */
    void sendOverStream(PublishBuildToolEventStreamRequest buildEvent) throws InterruptedException;

    /**
     * Half closes the currently opened stream. This method does not block. Callers should block on
     * the future returned by {@link #getStatus()} in order to make sure that all {@code
     * ackCallback} calls have been received.
     */
    void halfCloseStream();

    /**
     * Closes the currently opened stream with an error. This method does not block. Callers should
     * block on the future returned by {@link #getStatus()} in order to make sure that all
     * ackCallback calls have been received. This method is NOOP if the stream was already finished.
     */
    void abortStream(Status status);
  }

  /** Makes a blocking RPC call that publishes a {@code lifecycleEvent}. */
  void publish(PublishLifecycleEventRequest lifecycleEvent)
      throws StatusException, InterruptedException;

  /**
   * Starts a new stream with the given {@code ackCallback}. Callers must wait on the returned
   * future contained in the {@link StreamContext} in order to guarantee that all callback calls
   * have been received.
   */
  StreamContext openStream(AckCallback callback) throws InterruptedException;

  /**
   * Called once to dispose resources that this client might be holding (such as thread pools). This
   * should be the last method called on this object.
   */
  void shutdown();

  /**
   * If possible, returns a user readable error message for a given {@link Throwable}.
   *
   * <p>As a last resort, it's valid to return {@link Throwable#getMessage()}.
   */
  String userReadableError(Throwable t);
}
