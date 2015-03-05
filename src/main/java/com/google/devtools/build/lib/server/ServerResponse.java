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

package com.google.devtools.build.lib.server;

import com.google.common.base.Preconditions;

import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;

/**
 * This class models a response from the {@link RPCServer}. This is a
 * tuple of an error message and the exit status. The encoding of the response
 * is extremely simple {@link #toString()}:
 *
 * <ul><li>Iff a message is present, the wire format is
 *         <pre>message + '\n' + exit code as string + '\n'</pre>
 *     </li>
 *     <li>Otherwise it's just the exit code as string + '\n'</li>
 * </ul>
 */
final class ServerResponse {

  /**
   * Parses an input string into a {@link ServerResponse} object.
   */
  public static ServerResponse parseFrom(String input) {
    if (input.charAt(input.length() - 1) != '\n') {
      String msg = "Response must end with newline (" + input + ")";
      throw new IllegalArgumentException(msg);
    }
    int newlineAt = input.lastIndexOf('\n', input.length() - 2);

    final String exitStatusString;
    final String errorMessage;
    if (newlineAt == -1) {
      errorMessage = "";
      exitStatusString = input.substring(0, input.length() - 1);
    } else {
      errorMessage = input.substring(0, newlineAt);
      exitStatusString = input.substring(newlineAt + 1, input.length() - 1);
    }

    return new ServerResponse(errorMessage, Integer.parseInt(exitStatusString));
  }

  /**
   * Parses {@code bytes} into a {@link ServerResponse} instance, assuming
   * Latin 1 encoding.
   */
  public static ServerResponse parseFrom(byte[] bytes) {
    try {
      return parseFrom(new String(bytes, "ISO-8859-1"));
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e); // Latin 1 is everywhere.
    }
  }

  /**
   * Parses {@code bytes} into a {@link ServerResponse} instance, assuming
   * Latin 1 encoding.
   */
  public static ServerResponse parseFrom(ByteArrayOutputStream bytes) {
    return parseFrom(bytes.toByteArray());
  }

  private final String errorMessage;
  private final int exitStatus;

  /**
   * Construct a new instance given an error message and an exit status.
   */
  public ServerResponse(String errorMessage, int exitStatus) {
    Preconditions.checkNotNull(errorMessage);
    this.errorMessage = errorMessage;
    this.exitStatus = exitStatus;
  }

  /**
   * The wire representation of this response object.
   */
  @Override
  public String toString() {
    if (errorMessage.length() == 0) {
      return Integer.toString(exitStatus) + '\n';
    }
    return errorMessage + '\n' + exitStatus + '\n';
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ServerResponse)) {
      return false;
    }
    ServerResponse otherResponse = (ServerResponse) other;
    return exitStatus == otherResponse.exitStatus
        && errorMessage.equals(otherResponse.errorMessage);
  }

  @Override
  public int hashCode() {
    return exitStatus * 31 ^ errorMessage.hashCode();
  }

}
