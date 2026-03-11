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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.CacheMissReason;
import java.util.ArrayList;

/** Exception signaling a failure to Serialize or Deserialize an Object. */
public class SerializationException extends Exception {
  private final ArrayList<String> trail = new ArrayList<>();
  private final CacheMissReason reason;

  private static CacheMissReason maybePropagateReason(Throwable cause) {
    return cause instanceof SerializationException se ? se.reason : CacheMissReason.UNKNOWN;
  }

  public SerializationException(String msg) {
    super(msg);
    this.reason = CacheMissReason.UNKNOWN;
  }

  public SerializationException(Throwable cause) {
    super(cause);
    this.reason = maybePropagateReason(cause);
  }

  public SerializationException(String msg, CacheMissReason reason) {
    super(msg);
    this.reason = reason;
  }

  public SerializationException(String msg, Throwable cause) {
    super(msg, cause);
    this.reason = maybePropagateReason(cause);
  }

  public SerializationException(String msg, Throwable cause, CacheMissReason reason) {
    super(msg, cause);
    this.reason = reason;
  }

  public CacheMissReason getReason() {
    return reason;
  }

  // No SerializationException(Throwable) overload because serialization errors should always
  // provide as much context as possible.

  /**
   * {@link SerializationException} indicating that Blaze has no serialization schema for an object
   * or type of object.
   */
  public static class NoCodecException extends SerializationException {
    NoCodecException(String message) {
      super(message);
    }

    NoCodecException(String message, Class<?> type) {
      super(message);
      addTrail(type);
    }

    // Needed for wrapping.
    NoCodecException(String message, NoCodecException e) {
      super(message, e);
    }
  }

  @Override
  public String getMessage() {
    return super.getMessage() + (trail.isEmpty() ? "" : " " + trail);
  }

  /**
   * Adds extra tracing info for debugging.
   *
   * <p>Primarily useful for {@link DynamicCodec}.
   */
  public void addTrail(Class<?> type) {
    trail.add(type.getName());
  }

  public ImmutableList<String> getTrailForTesting() {
    return ImmutableList.copyOf(trail);
  }

  /**
   * Throws a {@link SerializationException} with the given message and that wraps the given cause.
   *
   * <p>If the cause is a {@link NoCodecException}, the returned exception will also be a {@code
   * NoCodecException}.
   *
   * <p>The return type is {@link SerializationException} rather than {@code void} so that you can
   * call this function from within a {@code throw} statement. Doing so keeps the calling code more
   * readable. It also avoids spurious compiler errors, e.g. for using uninitialized variables after
   * the {@code throw}.
   */
  public static SerializationException propagate(String msg, Throwable cause) {
    if (cause instanceof NoCodecException) {
      return new NoCodecException(msg, (NoCodecException) cause);
    } else {
      return new SerializationException(msg, cause);
    }
  }
}
