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

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/** Interface used to abstract the Stubby and gRPC client implementations. */
@SkybridgeInterface
public interface BuildEventServiceClient {



  /** Callback for ACKed build events. */
  @FunctionalInterface
  interface AckCallback {
    /**
     * Called whenever an ACK from the BES server is received. ACKs are expected to be received in
     * sequence. Implementations must be thread-safe.
     */
    void apply(long sequenceNumber);
  }

  /** The status of a stream. */
  public interface StreamStatus {
    /** Returns whether the status is successful. */
    boolean isOk();

    /** Returns whether the status is retriable. */
    boolean isRetriable();

    /** Returns whether the status indicates a failed precondition. */
    boolean isFailedPrecondition();

    /** Returns an error message for this status. */
    String getErrorMessage();
  }

  /** An exception with an underlying {@link StreamStatus}. */
  public class StreamException extends Exception {
    private final StreamStatus status;

    public StreamException(StreamStatus status, @Nullable Throwable cause) {
      super(status.getErrorMessage(), cause);
      this.status = status;
    }

    /** Returns the underlying {@link StreamStatus}. */
    public StreamStatus getStatus() {
      return status;
    }
  }

  /** The reason why a stream is being aborted. */
  public final class AbortReason {
    private final String name;

    private AbortReason(String name) {
      this.name = name;
    }

    /** The operation was cancelled. */
    public static final AbortReason CANCELLED = new AbortReason("CANCELLED");

    /** A precondition was failed. */
    public static final AbortReason FAILED_PRECONDITION = new AbortReason("FAILED_PRECONDITION");

    @Override
    public String toString() {
      return name;
    }
  }

  /** A handle to a bidirectional stream. */
  interface StreamContext {

    /**
     * The completed status of the stream. The future will never fail, but in case of error will
     * contain a corresponding status.
     */
    Future<StreamStatus> getStatus();

    /**
     * Sends a {@link StreamEvent} over the currently open stream. In case of error, this method
     * will fail silently and report the error via the {@link Future} returned by {@link
     * #getStatus()}.
     *
     * <p>This method may block due to flow control.
     */
    void sendOverStream(StreamEvent streamEvent) throws InterruptedException;

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
    void abortStream(AbortReason reason, @Nullable String description);
  }

  /** Makes a blocking RPC call that publishes a {@link LifecycleEvent}. */
  void publish(CommandContext commandContext, LifecycleEvent lifecycleEvent)
      throws StreamException, InterruptedException;

  /**
   * Starts a new stream with the given {@link CommandContext} and {@link AckCallback}. Callers must
   * wait on the returned future contained in the {@link StreamContext} in order to guarantee that
   * all callback calls have been received.
   */
  StreamContext openStream(CommandContext commandContext, AckCallback callback)
      throws InterruptedException;

  /**
   * Called once to dispose resources that this client might be holding (such as thread pools). This
   * should be the last method called on this object.
   */
  void shutdown();
}
