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
import java.time.Duration;
import java.util.Collection;
import java.util.Map;

/** Helper methods relating to implementations of {@link Spawn}. */
public final class Spawns {
  private Spawns() {}

  /**
   * Returns {@code true} if the result of {@code spawn} may be cached.
   */
  public static boolean mayBeCached(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_CACHE);
  }

  public static boolean mayBeSandboxed(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LEGACY_NOSANDBOX)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_SANDBOX)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LOCAL);
  }

  public static boolean requiresNetwork(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.BLOCK_NETWORK);
  }

  public static boolean mayBeExecutedRemotely(Spawn spawn) {
    return !spawn.getExecutionInfo().containsKey(ExecutionRequirements.LOCAL)
        && !spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_REMOTE);
  }

  /**
   * Parse the timeout key in the spawn execution info, if it exists. Otherwise, return -1.
   */
  public static Duration getTimeout(Spawn spawn) throws ExecException {
    String timeoutStr = spawn.getExecutionInfo().get("timeout");
    if (timeoutStr == null) {
      return Duration.ZERO;
    }
    try {
      return Duration.ofSeconds(Integer.parseInt(timeoutStr));
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
