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

import java.io.NotSerializableException;

/** Exception signaling a failure to Serialize or Deserialize an Object. */
public class SerializationException extends Exception {

  public SerializationException(String msg) {
    super(msg);
  }

  public SerializationException(String msg, Throwable cause) {
    super(msg, cause);
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

    NoCodecException(String message, NotSerializableException e) {
      super(message, e);
    }

    NoCodecException(String message, NotSerializableRuntimeException e) {
      super(message, e);
    }
  }
}
