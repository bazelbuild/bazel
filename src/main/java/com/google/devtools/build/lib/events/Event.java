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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Stream;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.SyntaxError;

/**
 * A situation encountered by the build system that's worth reporting.
 *
 * <p>An event specifies an {@link EventKind}, a message, and (optionally) additional properties.
 *
 * <p>Although this is serializable, use caution if {@link Class}es with the same name and different
 * classloaders are expected to be found in the same event's properties. Common class object
 * serialization techniques ignore classloaders, so duplicate entry keys may be deserialized,
 * violating assumptions.
 */
@Immutable
@CheckReturnValue
public final class Event implements Serializable {

  private final EventKind kind;

  /**
   * This field has type {@link String} or {@link byte[]}.
   *
   * <p>If this field is a byte array then it contains the UTF-8-encoded bytes of a message. This
   * optimization avoids converting back and forth between strings and bytes.
   */
  private final Object message;

  /**
   * This map's entries are ordered by {@link Class#getName}.
   *
   * <p>That is not a total ordering because of classloaders. The order of entries whose key names
   * are equal is not deterministic.
   */
  private final ImmutableClassToInstanceMap<Object> properties;

  private int hashCode;

  private Event(EventKind kind, Object message, ImmutableClassToInstanceMap<Object> properties) {
    this.kind = checkNotNull(kind);
    this.message = checkNotNull(message);
    this.properties = checkNotNull(properties);
  }

  public EventKind getKind() {
    return kind;
  }

  public String getMessage() {
    return message instanceof String ? (String) message : new String((byte[]) message, UTF_8);
  }

  /**
   * Returns this event's message as a {@link byte[]}. If this event was instantiated using a {@link
   * String}, the returned byte array is encoded using {@link
   * java.nio.charset.StandardCharsets#UTF_8}.
   */
  public byte[] getMessageBytes() {
    return message instanceof byte[] ? (byte[]) message : ((String) message).getBytes(UTF_8);
  }

  /** Returns the property value associated with {@code type} if any, and {@code null} otherwise. */
  @Nullable
  public <T> T getProperty(Class<T> type) {
    return properties.getInstance(type);
  }

  /**
   * Returns an {@link Event} instance that has the same type, message, and properties as the event
   * this is called on, and additionally associates {@code propertyValue} (if non-{@code null}) with
   * {@code type}.
   *
   * <p>If the event this is called on already has a property associated with {@code type} and
   * {@code propertyValue} is non-{@code null}, the returned event will have {@code propertyValue}
   * associated with it instead. If {@code propertyValue} is non-{@code null}, the returned event
   * will have no property associated with {@code type}.
   *
   * <p>If the event this is called on has no property associated with {@code type}, and {@code
   * propertyValue} is {@code null}, then this returns that event (it does not create a new {@link
   * Event} instance).
   *
   * <p>In any case, the event this is called on does not change.
   */
  // This implementation would be inefficient if #withProperty is called repeatedly because it may
  // copy and sort the key collection. In practice we expect it to be called a small number of times
  // per event (e.g. fewer than 5; usually 0).
  //
  // If that changes then consider an Event.Builder strategy instead.
  public <T> Event withProperty(Class<T> type, @Nullable T propertyValue) {
    Iterable<Class<?>> orderedKeys;
    boolean containsKey = properties.containsKey(type);
    if (!containsKey && propertyValue != null) {
      orderedKeys =
          Stream.concat(properties.keySet().stream(), Stream.of(type))
              .sorted(comparing(Class::getName))
              .collect(toImmutableList());
    } else if (containsKey) {
      orderedKeys = properties.keySet();
    } else {
      // !containsKey and propertyValue is null, so there's nothing to change.
      return this;
    }

    ImmutableClassToInstanceMap.Builder<Object> newProperties =
        new ImmutableClassToInstanceMap.Builder<>();
    for (Class<?> key : orderedKeys) {
      if (key.equals(type)) {
        if (propertyValue != null) {
          newProperties.put(type, propertyValue);
        }
      } else {
        addToBuilder(newProperties, key);
      }
    }

    return new Event(kind, message, newProperties.build());
  }

  /**
   * This type-parameterized method solves a problem where a {@code properties.getInstance(key)}
   * expression would have type {@link Object} when {@code key} is a wildcard-parameterized {@link
   * Class}. That {@link Object}-typed expression would then fail to type check in a {@code
   * builder.put(key, properties.getInstance(key))} statement.
   */
  private <T> void addToBuilder(ImmutableClassToInstanceMap.Builder<Object> builder, Class<T> key) {
    builder.put(key, checkNotNull(properties.getInstance(key)));
  }

  /**
   * Like {@link #withProperty(Class, Object)}, with {@code type.equals(String.class)}.
   *
   * <p>Additionally, if the event this is called on already has a {@link String} property with
   * value {@code tag}, or if {@code tag} is {@code null} and the event has no {@link String}
   * property, then this returns that event (it does not create a new {@link Event} instance).
   */
  public Event withTag(@Nullable String tag) {
    if (Objects.equals(tag, getProperty(String.class))) {
      return this;
    }
    return withProperty(String.class, tag);
  }

  /** Like {@link #withProperty(Class, Object)}, with {@code type.equals(FileOutErr.class)}. */
  public Event withStdoutStderr(FileOutErr outErr) {
    return withProperty(FileOutErr.class, outErr);
  }

  /**
   * Returns the {@link String} property, if any, asssociated with the event. When non-null, this
   * value typically describes some property of the action that generated the event.
   */
  // TODO(mschaller): change code which relies on this to rely on a more structured value, using
  //  types less prone to interference.
  @Nullable
  public String getTag() {
    return getProperty(String.class);
  }

  /** Indicates if a {@link FileOutErr} property is associated with this event. */
  public boolean hasStdoutStderr() {
    return getProperty(FileOutErr.class) != null;
  }

  /**
   * Gets the path to the stdout associated with this event (which the caller must not access), or
   * null if there is no such path.
   */
  @Nullable
  public PathFragment getStdOutPathFragment() {
    FileOutErr outErr = getProperty(FileOutErr.class);
    return outErr == null ? null : outErr.getOutputPathFragment();
  }

  /** Gets the size of the stdout associated with this event without reading it. */
  public long getStdOutSize() throws IOException {
    FileOutErr outErr = getProperty(FileOutErr.class);
    return outErr == null ? 0 : outErr.outSize();
  }

  /** Returns the stdout bytes associated with this event if any, and {@code null} otherwise. */
  @Nullable
  public byte[] getStdOut() {
    FileOutErr outErr = getProperty(FileOutErr.class);
    if (outErr == null) {
      return null;
    }
    return outErr.outAsBytes();
  }

  /**
   * Gets the path to the stderr associated with this event (which the caller must not access), or
   * null if there is no such path.
   */
  @Nullable
  public PathFragment getStdErrPathFragment() {
    FileOutErr outErr = getProperty(FileOutErr.class);
    return outErr == null ? null : outErr.getErrorPathFragment();
  }

  /** Gets the size of the stderr associated with this event without reading it. */
  public long getStdErrSize() throws IOException {
    FileOutErr outErr = getProperty(FileOutErr.class);
    return outErr == null ? 0 : outErr.errSize();
  }

  /** Returns the stderr bytes associated with this event if any, and {@code null} otherwise. */
  @Nullable
  public byte[] getStdErr() {
    FileOutErr outErr = getProperty(FileOutErr.class);
    if (outErr == null) {
      return null;
    }
    return outErr.errAsBytes();
  }

  /**
   * Returns the location of this event, if any. Returns null iff the event wasn't associated with
   * any particular location, for example, a progress message.
   */
  @Nullable
  public Location getLocation() {
    return getProperty(Location.class);
  }

  /** Returns the event formatted as {@code "ERROR foo.bzl:1:2: oops"}. */
  @Override
  public String toString() {
    Location location = getLocation();
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
      h =
          Objects.hash(
              kind,
              message instanceof String ? message : Arrays.hashCode((byte[]) message),
              properties);
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
        && this.message.getClass().equals(that.message.getClass())
        && (this.message instanceof String
            ? Objects.equals(this.message, that.message)
            : Arrays.equals((byte[]) this.message, (byte[]) that.message))
        && Objects.equals(this.properties, that.properties);
  }

  /** Constructs an event with the provided {@link EventKind} and {@link String} message. */
  public static Event of(EventKind kind, String message) {
    return new Event(kind, message, ImmutableClassToInstanceMap.of());
  }

  /**
   * Constructs an event with the provided {@link EventKind}, {@link String} message, and single
   * property value.
   *
   * <p>See {@link #withProperty(Class, Object)} if more than one property value is desired.
   */
  public static <T> Event of(
      EventKind kind, String message, Class<T> propertyType, T propertyValue) {
    return new Event(kind, message, ImmutableClassToInstanceMap.of(propertyType, propertyValue));
  }

  /** Constructs an event with the provided {@link EventKind} and {@link byte[]} message. */
  public static Event of(EventKind kind, byte[] messageBytes) {
    return new Event(kind, messageBytes, ImmutableClassToInstanceMap.of());
  }

  /**
   * Constructs an event with the provided {@link EventKind}, {@link byte[]} message, and single
   * property value.
   *
   * <p>See {@link #withProperty(Class, Object)} if more than one property value is desired.
   */
  public static <T> Event of(
      EventKind kind, byte[] messageBytes, Class<T> propertyType, T propertyValue) {
    return new Event(
        kind, messageBytes, ImmutableClassToInstanceMap.of(propertyType, propertyValue));
  }

  /**
   * Constructs an event with the provided {@link EventKind} and {@link String} message, with an
   * optional {@link Location}.
   */
  public static Event of(EventKind kind, @Nullable Location location, String message) {
    return location == null ? of(kind, message) : of(kind, message, Location.class, location);
  }

  /**
   * Constructs an event with a {@code byte[]} array instead of a {@link String} for its message.
   *
   * <p>The bytes must be decodable as UTF-8 text.
   */
  public static Event of(EventKind kind, @Nullable Location location, byte[] messageBytes) {
    return location == null
        ? of(kind, messageBytes)
        : of(kind, messageBytes, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#FATAL}. */
  public static Event fatal(String message) {
    return of(EventKind.FATAL, message);
  }

  /** Constructs an event with kind {@link EventKind#ERROR}, with an optional {@link Location}. */
  public static Event error(@Nullable Location location, String message) {
    return location == null
        ? of(EventKind.ERROR, message)
        : of(EventKind.ERROR, message, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#ERROR}. */
  public static Event error(String message) {
    return of(EventKind.ERROR, message);
  }

  /** Constructs an event with kind {@link EventKind#WARNING}, with an optional {@link Location}. */
  public static Event warn(@Nullable Location location, String message) {
    return location == null
        ? of(EventKind.WARNING, message)
        : of(EventKind.WARNING, message, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#WARNING}. */
  public static Event warn(String message) {
    return of(EventKind.WARNING, message);
  }

  /** Constructs an event with kind {@link EventKind#INFO}, with an optional {@link Location}. */
  public static Event info(@Nullable Location location, String message) {
    return location == null
        ? of(EventKind.INFO, message)
        : of(EventKind.INFO, message, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#INFO}. */
  public static Event info(String message) {
    return of(EventKind.INFO, message);
  }

  /**
   * Constructs an event with kind {@link EventKind#PROGRESS}, with an optional {@link Location}.
   */
  public static Event progress(@Nullable Location location, String message) {
    return location == null
        ? of(EventKind.PROGRESS, message)
        : of(EventKind.PROGRESS, message, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#PROGRESS}. */
  public static Event progress(String message) {
    return of(EventKind.PROGRESS, message);
  }

  /** Constructs an event with kind {@link EventKind#DEBUG}, with an optional {@link Location}. */
  public static Event debug(@Nullable Location location, String message) {
    return location == null
        ? of(EventKind.DEBUG, message)
        : of(EventKind.DEBUG, message, Location.class, location);
  }

  /** Constructs an event with kind {@link EventKind#DEBUG}. */
  public static Event debug(String message) {
    return of(EventKind.DEBUG, message);
  }

  /** Replays a sequence of events on {@code handler}. */
  public static void replayEventsOn(EventHandler handler, Iterable<Event> events) {
    for (Event event : events) {
      handler.handle(event);
    }
  }

  /** Converts a list of {@link SyntaxError}s to events and replays them on {@code handler}. */
  public static void replayEventsOn(EventHandler handler, List<SyntaxError> errors) {
    for (SyntaxError error : errors) {
      handler.handle(Event.error(error.location(), error.message()));
    }
  }

  /**
   * Converts a list of {@link SyntaxError}s to events, each with a specified property, and replays
   * them on {@code handler}.
   */
  public static <T> void replayEventsOn(
      EventHandler handler,
      List<SyntaxError> errors,
      Class<T> propertyType,
      Function<SyntaxError, T> toProperty) {
    for (SyntaxError error : errors) {
      handler.handle(
          Event.error(error.location(), error.message())
              .withProperty(propertyType, toProperty.apply(error)));
    }
  }

  /**
   * Returns a {@link StarlarkThread.PrintHandler} that sends {@link EventKind#DEBUG} events to the
   * provided {@link EventHandler}.
   */
  public static StarlarkThread.PrintHandler makeDebugPrintHandler(EventHandler h) {
    return (thread, msg) -> h.handle(Event.debug(thread.getCallerLocation(), msg));
  }
}
