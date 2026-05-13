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
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Instant;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/** Interface used to abstract the Stubby and gRPC client implementations. */
@SkybridgeInterface
public interface BuildEventServiceClient {

  /** Context for a build command. */
  public record CommandContext(
      String buildId,
      String invocationId,
      int attemptNumber,
      Set<String> keywords,
      @Nullable String projectId,
      boolean checkPrecedingLifecycleEvents) {
    public CommandContext {
      Objects.requireNonNull(buildId, "buildId");
      Objects.requireNonNull(invocationId, "invocationId");
      Objects.requireNonNull(keywords, "keywords");
      if (attemptNumber < 1) {
        throw new IllegalArgumentException("attemptNumber must be >= 1");
      }
    }

    public static Builder builder() {
      return new Builder();
    }

    /** Builder for {@link CommandContext}. */
    public static final class Builder {
      private String buildId;
      private String invocationId;
      private int attemptNumber;
      private Set<String> keywords;
      private String projectId;
      private boolean checkPrecedingLifecycleEvents;

      private Builder() {}

      @CanIgnoreReturnValue
      public Builder setBuildId(String buildId) {
        this.buildId = buildId;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setInvocationId(String invocationId) {
        this.invocationId = invocationId;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setAttemptNumber(int attemptNumber) {
        this.attemptNumber = attemptNumber;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setKeywords(Set<String> keywords) {
        this.keywords = keywords;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setProjectId(@Nullable String projectId) {
        this.projectId = projectId;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setCheckPrecedingLifecycleEvents(boolean checkPrecedingLifecycleEvents) {
        this.checkPrecedingLifecycleEvents = checkPrecedingLifecycleEvents;
        return this;
      }

      public CommandContext build() {
        return new CommandContext(
            buildId,
            invocationId,
            attemptNumber,
            keywords,
            projectId,
            checkPrecedingLifecycleEvents);
      }
    }
  }

  /** The status of an invocation. */
  enum InvocationStatus {
    /** No information is available about the invocation status. */
    UNKNOWN,
    /** The invocation succeeded. */
    SUCCEEDED,
    /** The invocation failed. */
    FAILED,
  }

  /** A lifecycle event. */
  sealed interface LifecycleEvent {
    /** The time at which the event occurred. */
    Instant eventTime();

    /** The lifecycle event signalling that the build was enqueued. */
    record BuildEnqueued(Instant eventTime) implements LifecycleEvent {}

    /** The lifecycle event signalling that the invocation was started. */
    record InvocationStarted(Instant eventTime) implements LifecycleEvent {}

    /**
     * The lifecycle event signalling that the invocation was finished.
     *
     * @param status the invocation status
     */
    record InvocationFinished(Instant eventTime, InvocationStatus status)
        implements LifecycleEvent {}

    /**
     * The lifecycle event signalling that the build was finished.
     *
     * @param status the invocation status
     */
    record BuildFinished(Instant eventTime, InvocationStatus status) implements LifecycleEvent {}
  }

  /** An event sent over a {@link StreamContext}. */
  sealed interface StreamEvent {
    /** The time at which the event occurred. */
    Instant eventTime();

    /** The sequence number of the event. */
    long sequenceNumber();

    /**
     * An event containing a {@link BuildEventStreamProtos.BuildEvent}.
     *
     * @param payload the {@link BuildEventStreamProtos.BuildEvent} in wire format
     */
    @SuppressWarnings("ArrayRecordComponent")
    record BazelEvent(Instant eventTime, long sequenceNumber, byte[] payload)
        implements StreamEvent {}

    /** An event signalling the end of the stream. */
    record StreamFinished(Instant eventTime, long sequenceNumber) implements StreamEvent {}
  }

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
  enum AbortReason {
    /** The operation was cancelled. */
    CANCELLED,
    /** A precondition was failed. */
    FAILED_PRECONDITION,
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
