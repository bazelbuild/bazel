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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.auto.value.AutoBuilder;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.StatusException;
import java.time.Instant;
import java.util.Set;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/** Interface used to abstract the Stubby and gRPC client implementations. */
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
      checkNotNull(buildId);
      checkNotNull(invocationId);
      checkArgument(attemptNumber >= 1);
      checkNotNull(keywords);
    }

    public static Builder builder() {
      return new AutoBuilder_BuildEventServiceClient_CommandContext_Builder();
    }

    /** Builder for {@link CommandContext}. */
    @AutoBuilder
    public abstract static class Builder {
      public abstract Builder setBuildId(String buildId);

      public abstract Builder setInvocationId(String invocationId);

      public abstract Builder setAttemptNumber(int attemptNumber);

      public abstract Builder setKeywords(Set<String> keywords);

      public abstract Builder setProjectId(@Nullable String projectId);

      public abstract Builder setCheckPrecedingLifecycleEvents(
          boolean checkPrecedingLifecycleEvents);

      public abstract CommandContext build();
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
    /** The command context for the event. */
    CommandContext commandContext();

    /** The time at which the event occurred. */
    Instant eventTime();

    /** The lifecycle event signalling that the build was enqueued. */
    record BuildEnqueued(CommandContext commandContext, Instant eventTime)
        implements LifecycleEvent {}

    /** The lifecycle event signalling that the invocation was started. */
    record InvocationStarted(CommandContext commandContext, Instant eventTime)
        implements LifecycleEvent {}

    /**
     * The lifecycle event signalling that the invocation was finished.
     *
     * @param status the invocation status
     */
    record InvocationFinished(
        CommandContext commandContext, Instant eventTime, InvocationStatus status)
        implements LifecycleEvent {}

    /**
     * The lifecycle event signalling that the build was finished.
     *
     * @param status the invocation status
     */
    record BuildFinished(CommandContext commandContext, Instant eventTime, InvocationStatus status)
        implements LifecycleEvent {}
  }

  /** An event sent over a {@link StreamContext}. */
  sealed interface StreamEvent {
    /** The command context for the event. */
    CommandContext commandContext();

    /** The time at which the event occurred. */
    Instant eventTime();

    /** The sequence number of the event. */
    long sequenceNumber();

    /**
     * An event containing a {@link BuildEventStreamProtos.BuildEvent}.
     *
     * @param payload the {@link BuildEventStreamProtos.BuildEvent} in wire format
     */
    record BazelEvent(
        CommandContext commandContext, Instant eventTime, long sequenceNumber, ByteString payload)
        implements StreamEvent {}

    /** An event signalling the end of the stream. */
    record StreamFinished(CommandContext commandContext, Instant eventTime, long sequenceNumber)
        implements StreamEvent {}
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

  /** A handle to a bidirectional stream. */
  interface StreamContext {

    /**
     * The completed status of the stream. The future will never fail, but in case of error will
     * contain a corresponding status.
     */
    Future<Status> getStatus();

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
    void abortStream(Status status);
  }

  /** Makes a blocking RPC call that publishes a {@link LifecycleEvent}. */
  void publish(LifecycleEvent lifecycleEvent) throws StatusException, InterruptedException;

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
