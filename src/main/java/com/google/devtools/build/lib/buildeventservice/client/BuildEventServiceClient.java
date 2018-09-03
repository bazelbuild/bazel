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
import java.util.function.Function;

/** Interface used to abstract the Stubby and gRPC client implementations. */
public interface BuildEventServiceClient {

  /** Makes a blocking RPC call that publishes a {@code lifecycleEvent}. */
  void publish(PublishLifecycleEventRequest lifecycleEvent)
      throws StatusException, InterruptedException;

  /**
   * Starts a new stream with the given {@code ackCallback}. Callers must wait on the returned
   * future in order to guarantee that all callback calls have been received. The returned future
   * will never fail, but in case of error will contain a corresponding status.
   */
  ListenableFuture<Status> openStream(
      Function<PublishBuildToolEventStreamResponse, Void> ackCallback)
      throws StatusException, InterruptedException;

  /**
   * Sends an event over the currently open stream. This method may block due to flow control.
   */
  void sendOverStream(PublishBuildToolEventStreamRequest buildEvent)
      throws StatusException, InterruptedException;

  /**
   * Closes the currently opened stream. This method does not block. Callers should block on
   * the future returned by {@link #openStream(Function)} in order to make sure that all
   * {@code ackCallback} calls have been received.
   */
  void closeStream();

  /**
   * Closes the currently opened stream with error. This method does not block. Callers should block
   * on the future returned by {@link #openStream(Function)} if in order to make sure that all
   * ackCallback calls have been received. This method is NOOP if the stream was already finished.
   */
  void abortStream(Status status);

  /**
   * Checks if there is a currently an active stream.
   *
   * @return {@code true} if the current stream is active, false otherwise.
   */
  boolean isStreamActive();

  /**
   * Called once to dispose resources that this client might be holding (such as thread pools). This
   * should be the last method called on this object.
   *
   * @throws InterruptedException
   */
  void shutdown() throws InterruptedException;

  /**
   * If possible, returns a user readable error message for a given {@link Throwable}.
   *
   * <p>As a last resort, it's valid to return {@link Throwable#getMessage()}.
   */
  String userReadableError(Throwable t);
}
