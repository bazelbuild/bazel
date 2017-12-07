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

package com.google.devtools.common.options;

/**
 * An exception that's thrown when the {@link OptionsParser} fails.
 *
 * @see OptionsParser#parse(OptionPriority.PriorityCategory,String,java.util.List)
 */
public class OptionsParsingException extends Exception {
  private final String invalidArgument;

  public OptionsParsingException(String message) {
    this(message, (String) null);
  }

  public OptionsParsingException(String message, String argument) {
    super(message);
    this.invalidArgument = argument;
  }

  public OptionsParsingException(String message, Throwable throwable) {
    this(message, null, throwable);
  }

  public OptionsParsingException(String message, String argument, Throwable throwable) {
    super(message, throwable);
    this.invalidArgument = argument;
  }

  /**
   * Gets the name of the invalid argument or {@code null} if the exception
   * can not determine the exact invalid arguments
   */
  public String getInvalidArgument() {
    return invalidArgument;
  }
}
