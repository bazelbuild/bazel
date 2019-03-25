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
import java.io.Serializable;
import java.util.Arrays;
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
  @Nullable private final String stdout;
  @Nullable private final String stderr;

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

  private final int hashCode;

  private Event(
      EventKind kind,
      @Nullable Location location,
      String message,
      @Nullable String tag,
      @Nullable String stdout,
      @Nullable String stderr) {
    this.kind = Preconditions.checkNotNull(kind);
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.messageBytes = null;
    this.tag = tag;
    this.stdout = stdout;
    this.stderr = stderr;
    this.hashCode =
        Objects.hash(kind, location, message, tag, Arrays.hashCode(messageBytes), stdout, stderr);
  }

  private Event(
      EventKind kind,
      @Nullable Location location,
      byte[] messageBytes,
      @Nullable String tag,
      @Nullable String stdout,
      @Nullable String stderr) {
    this.kind = Preconditions.checkNotNull(kind);
    this.location = location;
    this.message = null;
    this.messageBytes = Preconditions.checkNotNull(messageBytes);
    this.tag = tag;
    this.stdout = stdout;
    this.stderr = stderr;
    this.hashCode =
        Objects.hash(kind, location, message, tag, Arrays.hashCode(messageBytes), stdout, stderr);
  }

  public Event withTag(String tag) {
    if (Objects.equals(tag, this.tag)) {
      return this;
    }
    if (this.message != null) {
      return new Event(this.kind, this.location, this.message, tag, this.stdout, this.stderr);
    } else {
      return new Event(this.kind, this.location, this.messageBytes, tag, this.stdout, this.stderr);
    }
  }

  public Event withStdoutStderr(String stdout, String stderr) {
    if (this.message != null) {
      return new Event(this.kind, this.location, this.message, this.tag, stdout, stderr);
    } else {
      return new Event(this.kind, this.location, this.messageBytes, this.tag, stdout, stderr);
    }
  }

  public String getMessage() {
    return message != null ? message : new String(messageBytes, UTF_8);
  }

  public byte[] getMessageBytes() {
    return messageBytes != null ? messageBytes : message.getBytes(UTF_8);
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

  /**
   * Get the stdout bytes associated with this event; typically, the event will report where the
   * output originated from.
   */
  @Nullable
  public String getStdOut() {
    return stdout;
  }

  /**
   * Get the stdout bytes associated with this event; typically, the event will report where the
   * output originated from.
   */
  @Nullable
  public String getStdErr() {
    return stderr;
  }

  /**
   * Returns the location of this event, if any.  Returns null iff the event
   * wasn't associated with any particular location, for example, a progress
   * message.
   */
  @Nullable public Location getLocation() {
    return location;
  }

  /**
   * Returns <i>some</i> moderately sane representation of the event. Should never be used in
   * user-visible places, only for debugging and testing.
   */
  @Override
  public String toString() {
    return kind + " " + (location != null ? location.print() : "<no location>") + ": "
        + getMessage();
  }

  @Override
  public int hashCode() {
    return hashCode;
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
        && Objects.equals(this.stdout, that.stdout)
        && Objects.equals(this.stderr, that.stderr)
        && Arrays.equals(this.messageBytes, that.messageBytes);
  }

  /**
   * Replay a sequence of events on an {@link EventHandler}.
   */
  public static void replayEventsOn(EventHandler eventHandler, Iterable<Event> events) {
    for (Event event : events) {
      eventHandler.handle(event);
    }
  }

  public static Event of(EventKind kind, @Nullable Location location, String message) {
    return new Event(kind, location, message, null, null, null);
  }

  /**
   * Construct an event by passing in the {@code byte[]} array instead of a String.
   *
   * The bytes must be decodable as UTF-8 text.
   */
  public static Event of(EventKind kind, @Nullable Location location, byte[] messageBytes) {
    return new Event(kind, location, messageBytes, null, null, null);
  }

  /** Reports an error. */
  public static Event error(@Nullable Location location, String message) {
    return new Event(EventKind.ERROR, location, message, null, null, null);
  }

  /** Reports an error. */
  public static Event error(String message) {
    return error(null, message);
  }

  /** Reports a warning. */
  public static Event warn(@Nullable Location location, String message) {
    return new Event(EventKind.WARNING, location, message, null, null, null);
  }

  /** Reports a warning. */
  public static Event warn(String message) {
    return warn(null, message);
  }

  /**
   * Reports atemporal statements about the build, i.e. they're true for the duration of execution.
   */
  public static Event info(@Nullable Location location, String message) {
    return new Event(EventKind.INFO, location, message, null, null, null);
  }

  /**
   * Reports atemporal statements about the build, i.e. they're true for the duration of execution.
   */
  public static Event info(String message) {
    return info(null, message);
  }

  /** Reports a temporal statement about the build. */
  public static Event progress(@Nullable Location location, String message) {
    return new Event(EventKind.PROGRESS, location, message, null, null, null);
  }

  /** Reports a temporal statement about the build. */
  public static Event progress(String message) {
    return progress(null, message);
  }

  /** Reports a debug message. */
  public static Event debug(@Nullable Location location, String message) {
    return new Event(EventKind.DEBUG, location, message, null, null, null);
  }

  /**
   * Reports a debug message.
   */
  public static Event debug(String message) {
    return debug(null, message);
  }
}
