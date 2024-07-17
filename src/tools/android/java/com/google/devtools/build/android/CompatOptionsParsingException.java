// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android;

/**
 * An exception that's thrown when the {@link CompatShellQuotedParamsFilePreProcessor} fails.
 *
 * <p>This is effectively a fork of OptionsParsingException that's part of the OptionsParser lib.
 */
public class CompatOptionsParsingException extends Exception {
  private final String invalidArgument;

  public CompatOptionsParsingException(String message) {
    this(message, (String) null);
  }

  public CompatOptionsParsingException(String message, String argument) {
    super(message);
    this.invalidArgument = argument;
  }

  public CompatOptionsParsingException(String message, Throwable throwable) {
    this(message, null, throwable);
  }

  public CompatOptionsParsingException(String message, String argument, Throwable throwable) {
    super(message, throwable);
    this.invalidArgument = argument;
  }

  /**
   * Gets the name of the invalid argument or {@code null} if the exception can not determine the
   * exact invalid arguments
   */
  public String getInvalidArgument() {
    return invalidArgument;
  }
}
