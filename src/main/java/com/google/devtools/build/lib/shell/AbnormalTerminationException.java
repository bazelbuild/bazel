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

package com.google.devtools.build.lib.shell;

/**
 * Thrown when a command's execution terminates abnormally -- for example,
 * if it is killed, or if it terminates with a non-zero exit status.
 */
public class AbnormalTerminationException extends CommandException {

  private final CommandResult result;

  public AbnormalTerminationException(final Command command,
                                      final CommandResult result,
                                      final String message) {
    super(command, message);
    this.result = result;
  }

  public AbnormalTerminationException(final Command command,
                                      final CommandResult result,
                                      final Throwable cause) {
    super(command, cause);
    this.result = result;
  }

  public AbnormalTerminationException(final Command command,
                                      final CommandResult result,
                                      final String message,
                                      final Throwable cause) {
    super(command, message, cause);
    this.result = result;
  }

  public CommandResult getResult() {
    return result;
  }

  private static final long serialVersionUID = 2L;
}
