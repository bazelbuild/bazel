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

/**
 * This event is fired when the Blaze command is complete
 * (clean, build, test, etc.).
 */
public class CommandCompleteEvent extends CommandEvent {

  private final int exitCode;

  /**
   * @param exitCode the exit code of the blaze command
   */
  public CommandCompleteEvent(int exitCode) {
    this.exitCode = exitCode;
  }

  /**
   * @return the exit code of the blaze command
   */
  public int getExitCode() {
    return exitCode;
  }
}
