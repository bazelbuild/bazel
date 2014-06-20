// Copyright 2014 Google Inc. All rights reserved.
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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * An event is a situation encountered by the build system that's worth
 * reporting: A 3-tuple of ({@link EventKind}, {@link Location}, message).
 */
@Immutable
public final class Event {

  private final EventKind kind;
  private final Location location;
  private final String message;
  /**
   * An alternative representation for message.
   * Exactly one of message or messageBytes will be non-null.
   * If messageBytes is non-null, then it contains the bytes
   * of the message, encoded using the platform's default charset.
   * We do this to avoid converting back and forth between Strings
   * and bytes.
   */
  private final byte[] messageBytes;

  public Event(EventKind kind, @Nullable Location location, String message) {
    this.kind = kind;
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.messageBytes = null;
  }

  public Event(EventKind kind, @Nullable Location location, byte[] messageBytes) {
    this.kind = kind;
    this.location = location;
    this.message = null;
    this.messageBytes = Preconditions.checkNotNull(messageBytes);
  }

  public String getMessage() {
    return message != null ? message : new String(messageBytes);
  }

  public byte[] getMessageBytes() {
    return messageBytes != null ? messageBytes : message.getBytes(ISO_8859_1);
  }

  public EventKind getKind() {
    return kind;
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

  /**
   * Replay a sequence of events on an {@link ErrorEventListener}.
   */
  public static void replayEventsOn(ErrorEventListener listener, Iterable<Event> events) {
    for (Event event : events) {
      switch (event.getKind()) {
        case WARNING :
          listener.warn(event.getLocation(), event.getMessage());
          break;
        case ERROR :
          listener.error(event.getLocation(), event.getMessage());
          break;
        case INFO :
          listener.info(event.getLocation(), event.getMessage());
          break;
        case PROGRESS :
          listener.progress(event.getLocation(), event.getMessage());
          break;
        default :
          throw new IllegalStateException("Can't happen!");
      }
    }
  }
}
