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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.util.ExitCode;

/**
 * This message is fired right before the Blaze command completes,
 * and can be used to modify the command's exit code.
 */
public class CommandPrecompleteEvent {
  private final ExitCode exitCode;

  /**
   * @param exitCode the exit code of the blaze command
   */
  public CommandPrecompleteEvent(ExitCode exitCode) {
    this.exitCode = exitCode;
  }

  /**
   * @return the exit code of the blaze command
   */
  public ExitCode getExitCode() {
    return exitCode;
  }
}
