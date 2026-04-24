// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

/**
 * An exception thrown by {@link BlazeService#globalInit} to signal an abrupt exit.
 *
 * <p>This exception carries the serialized {@code FailureDetail} protobuf message as a {@code
 * byte[]} to avoid exposing protobuf types in the interface jar.
 */
public class SerializedAbruptExitException extends Exception {
  private final byte[] serializedFailureDetail;

  public SerializedAbruptExitException(String message, byte[] serializedFailureDetail) {
    super(message);
    this.serializedFailureDetail = serializedFailureDetail;
  }

  public SerializedAbruptExitException(
      String message, byte[] serializedFailureDetail, Throwable cause) {
    super(message, cause);
    this.serializedFailureDetail = serializedFailureDetail;
  }

  public byte[] getSerializedFailureDetail() {
    return serializedFailureDetail;
  }
}
