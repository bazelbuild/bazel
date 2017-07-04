// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;
import java.util.Map;

/** Helper methods relating to implementations of {@link Spawn}. */
public final class Spawns {
  private Spawns() {}

  /**
   * Parse the timeout key in the spawn execution info, if it exists. Otherwise, return -1.
   */
  public static int getTimeoutSeconds(Spawn spawn) throws ExecException {
    String timeoutStr = spawn.getExecutionInfo().get("timeout");
    if (timeoutStr == null) {
      return -1;
    }
    try {
      return Integer.parseInt(timeoutStr);
    } catch (NumberFormatException e) {
      throw new UserExecException("could not parse timeout: ", e);
    }
  }

  /**
   * Returns whether a local {@link Spawn} runner implementation should prefetch the inputs before
   * execution, based on the spawns execution info.
   */
  public static boolean shouldPrefetchInputsForLocalExecution(Spawn spawn) {
    String disablePrefetchRequest =
        spawn.getExecutionInfo().get(ExecutionRequirements.DISABLE_LOCAL_PREFETCH);
    return (disablePrefetchRequest == null) || disablePrefetchRequest.equals("0");
  }

  /** Convert a spawn into a Bourne shell command. */
  public static String asShellCommand(Spawn spawn, Path workingDirectory) {
    return asShellCommand(spawn.getArguments(), workingDirectory, spawn.getEnvironment());
  }

  /** Convert a working dir + environment map + arg list into a Bourne shell command. */
  public static String asShellCommand(
      Collection<String> arguments, Path workingDirectory, Map<String, String> environment) {
    // We print this command out in such a way that it can safely be
    // copied+pasted as a Bourne shell command.  This is extremely valuable for
    // debugging.
    return CommandFailureUtils.describeCommand(
        CommandDescriptionForm.COMPLETE, arguments, environment, workingDirectory.getPathString());
  }
}
