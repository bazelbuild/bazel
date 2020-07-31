// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.events;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.FileOutErr.OutputReference;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * An event is a situation encountered by the build system that's worth
 * reporting: A 3-tuple of ({@link EventKind}, {@link Location}, message).
 */
@Immutable
public final class Event implements Serializable {
  private final EventKind kind;
  private final Location location;
  private final String message;
  @Nullable private final FileOutErr outErr;

  /**
   * An alternative representation for message.
   *
   * <p>Exactly one of message or messageBytes will be non-null. If messageBytes is non-null, then
   * it contains the UTF-8-encoded bytes of the message. We do this to avoid converting back and
   * forth between Strings and bytes.
   */
  private final byte[] messageBytes;

  @Nullable
  private final String tag;

  private int hashCode;

  private Event(
      EventKind kind,
      @Nullable Location location,
      String message,
      @Nullable String tag,
      @Nullable FileOutErr outErr) {
    this.kind = Preconditions.checkNotNull(kind);
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.messageBytes = null;
    this.tag = tag;
    this.outErr = outErr;
  }

  private Event(
      EventKind kind,
      @Nullable Location location,
      byte[] messageBytes,
      @Nullable String tag,
      @Nullable FileOutErr outErr) {
    this.kind = Preconditions.checkNotNull(kind);
    this.location = location;
    this.message = null;
    this.messageBytes = Preconditions.checkNotNull(messageBytes);
    this.tag = tag;
    this.outErr = outErr;
  }

  public Event withTag(String tag) {
    if (Objects.equals(tag, this.tag)) {
      return this;
    }
    if (this.message != null) {
      return new Event(this.kind, this.location, this.message, tag, this.outErr);
    } else {
      return new Event(this.kind, this.location, this.messageBytes, tag, this.outErr);
    }
  }

  public Event withStdoutStderr(FileOutErr outErr) {
    if (this.message != null) {
      return new Event(this.kind, this.location, this.message, this.tag, outErr);
    } else {
      return new Event(this.kind, this.location, this.messageBytes, this.tag, outErr);
    }
  }

  public String getMessage() {
    return message != null ? message : new String(messageBytes, UTF_8);
  }

  public byte[] getMessageBytes() {
    return messageBytes != null ? messageBytes : message.getBytes(UTF_8);
  }

  /** Provide the message as a reference object. */
  public OutputReference getMessageReference() {
    // The message is short and we have it in memory anyway; so just wrap it into
    // the common interface.
    return new ArrayOutputReference(getMessageBytes());
  }

  public EventKind getKind() {
    return kind;
  }

  /**
   * the tag is typically the action that generated the event.
   */
  @Nullable
  public String getTag() {
    return tag;
  }

  /** Indicate if any output is associated with this event. */
  public boolean hasStdoutStderr() {
    return outErr != null;
  }

  /**
   * Get the stdout bytes associated with this event; typically, the event will report where the
   * output originated from.
   */
  @Nullable
  public byte[] getStdOut() {
    if (outErr == null) {
      return null;
    }
    return outErr.outAsBytes();
  }

  /**
   * Get the stdout bytes associated with this event; typically, the event will report where the
   * output originated from.
   */
  @Nullable
  public byte[] getStdErr() {
    if (outErr == null) {
      return null;
    }
    return outErr.errAsBytes();
  }

  /** Get a reference to the associated output. */
  public OutputReference getStdOutReference() {
    return outErr.getOutReference();
  }

  /** Get a reference to the associated output. */
  public OutputReference getStdErrReference() {
    return outErr.getErrReference();
  }

  /**
   * Returns the location of this event, if any.  Returns null iff the event
   * wasn't associated with any particular location, for example, a progress
   * message.
   */
  @Nullable public Location getLocation() {
    return location;
  }

  /** Returns the event formatted as {@code "ERROR foo.bzl:1:2: oops"}. */
  @Override
  public String toString() {
    // TODO(adonovan): <no location> is just noise.
    return kind
        + " "
        + (location != null ? location.toString() : "<no location>")
        + ": "
        + getMessage();
  }

  @Override
  public int hashCode() {
    // We defer the computation of hashCode until it is needed to avoid the overhead of computing it
    // and then never using it. In particular, we use Event for streaming stdout and stderr, which
    // are both large and the hashCode is never used.
    //
    // This uses the same construction as String.hashCode. We don't lock, so reads and writes to the
    // field can race. However, integer reads and writes are atomic, and this code guarantees that
    // all writes have the same value, so the memory location can only be either 0 or the final
    // value. Note that a reader could see the final value on the first read and 0 on the second
    // read, so we must take care to only read the field once.
    int h = hashCode;
    if (h == 0) {
      h = Objects.hash(kind, location, message, tag, Arrays.hashCode(messageBytes), outErr);
      hashCode = h;
    }
    return h;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (other == null || !other.getClass().equals(getClass())) {
      return false;
    }
    Event that = (Event) other;
    return Objects.equals(this.kind, that.kind)
        && Objects.equals(this.location, that.location)
        && Objects.equals(this.tag, that.tag)
        && Objects.equals(this.message, that.message)
        && Objects.equals(this.outErr, that.outErr)
        && Arrays.equals(this.messageBytes, that.messageBytes);
  }

  /** Replays a sequence of events on {@code handler}. */
  public static void replayEventsOn(EventHandler handler, Iterable<Event> events) {
    for (Event event : events) {
      handler.handle(event);
    }
  }

  /** Converts a list of SyntaxErrors to Events and replay on {@code handler}. */
  public static void replayEventsOn(EventHandler handler, List<SyntaxError> errors) {
    for (SyntaxError error : errors) {
      handler.handle(Event.error(error.location(), error.message()));
    }
  }

  public static Event of(EventKind kind, @Nullable Location location, String message) {
    return new Event(kind, location, message, null, null);
  }

  /**
   * Construct an event by passing in the {@code byte[]} array instead of a String.
   *
   * The bytes must be decodable as UTF-8 text.
   */
  public static Event of(EventKind kind, @Nullable Location location, byte[] messageBytes) {
    return new Event(kind, location, messageBytes, null, null);
  }

  /** Reports an error. */
  public static Event error(@Nullable Location location, String message) {
    return new Event(EventKind.ERROR, location, message, null, null);
  }

  /** Reports an error. */
  public static Event error(String message) {
    return error(null, message);
  }

  /** Reports a warning. */
  public static Event warn(@Nullable Location location, String message) {
    return new Event(EventKind.WARNING, location, message, null, null);
  }

  /** Reports a warning. */
  public static Event warn(String message) {
    return warn(null, message);
  }

  /**
   * Reports atemporal statements about the build, i.e. they're true for the duration of execution.
   */
  public static Event info(@Nullable Location location, String message) {
    return new Event(EventKind.INFO, location, message, null, null);
  }

  /**
   * Reports atemporal statements about the build, i.e. they're true for the duration of execution.
   */
  public static Event info(String message) {
    return info(null, message);
  }

  /** Reports a temporal statement about the build. */
  public static Event progress(@Nullable Location location, String message) {
    return new Event(EventKind.PROGRESS, location, message, null, null);
  }

  /** Reports a temporal statement about the build. */
  public static Event progress(String message) {
    return progress(null, message);
  }

  /** Reports a debug message. */
  public static Event debug(@Nullable Location location, String message) {
    return new Event(EventKind.DEBUG, location, message, null, null);
  }

  /**
   * Reports a debug message.
   */
  public static Event debug(String message) {
    return debug(null, message);
  }

  private static class ArrayOutputReference implements OutputReference {
    private final byte[] message;

    ArrayOutputReference(byte[] message) {
      this.message = message;
    }

    @Override
    public long getLength() {
      return message.length;
    }

    @Override
    public byte[] getFinalBytes(int count) {
      if (count >= message.length) {
        return message;
      } else {
        return Arrays.copyOfRange(message, message.length - count, message.length);
      }
    }
  }

  /** Returns a StarlarkThread PrintHandler that sends DEBUG events to the provided EventHandler. */
  public static StarlarkThread.PrintHandler makeDebugPrintHandler(EventHandler h) {
    return (thread, msg) -> h.handle(Event.debug(thread.getCallerLocation(), msg));
  }
}
