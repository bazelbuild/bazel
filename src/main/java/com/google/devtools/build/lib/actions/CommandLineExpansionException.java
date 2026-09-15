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

package com.google.devtools.build.lib.actions;

import javax.annotation.Nullable;

/**
 * Thrown by command line classes during expansion.
 *
 * <p>This exception should be thrown deterministically, i.e. the command line should fail to expand
 * in exactly the same way every time at attempt is made. An example would be an illegal format
 * string, or a failure in Starlark evaluation of a compact command line.
 */
public final class CommandLineExpansionException extends Exception {
  public CommandLineExpansionException(String userVisibleErrorMessage) {
    this(userVisibleErrorMessage, /*cause=*/ null);
  }

  /**
   * Constructs new exception with provided user-facing error message and optional cause.
   *
   * @param userVisibleErrorMessage error string that will be displayed to the user.
   * @param cause optional exception cause used for debugging only -- {@code
   *     userVisibleErrorMessage} should carry all of the information needed for the user.
   */
  public CommandLineExpansionException(String userVisibleErrorMessage, @Nullable Throwable cause) {
    super(userVisibleErrorMessage, cause);
  }
}
