// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility functions for the process wrapper embedded tool, which should work on most platforms and
 * gives at least some isolation between running actions.
 */
public final class ProcessWrapperUtil {
  private static final String PROCESS_WRAPPER = "process-wrapper" + OsUtils.executableExtension();

  /** Returns whether using the process wrapper is supported in the {@link CommandEnvironment}. */
  public static boolean isSupported(CommandEnvironment cmdEnv) {
    // We can only use this runner, if the process-wrapper exists in the embedded tools.
    // This might not always be the case, e.g. while bootstrapping.
    return getProcessWrapper(cmdEnv) != null;
  }

  /** Returns the {@link Path} of the process wrapper binary, or null if it doesn't exist. */
  public static Path getProcessWrapper(CommandEnvironment cmdEnv) {
    PathFragment execPath = cmdEnv.getBlazeWorkspace().getBinTools().getExecPath(PROCESS_WRAPPER);
    return execPath != null ? cmdEnv.getExecRoot().getRelative(execPath) : null;
  }

  /**
   * Returns a command line to execute a specific command using the process wrapper with a timeout.
   *
   * @param processWrapper the path to the process wrapper
   * @param spawnArguments the command to execute, and its arguments
   * @param timeout the time limit to run command using the process wrapper
   * @param timeoutGraceSeconds the delay (in seconds) to kill a command that exceeds its timeout
   *
   * @return the constructed command line to execute a command using the process wrapper
   */
  public static List<String> getCommandLine(
      Path processWrapper, List<String> spawnArguments, Duration timeout, int timeoutGraceSeconds) {
    List<String> commandLineArgs = new ArrayList<>(5 + spawnArguments.size());
    commandLineArgs.add(processWrapper.getPathString());
    commandLineArgs.add("--timeout=" + timeout.getSeconds());
    commandLineArgs.add("--kill_delay=" + timeoutGraceSeconds);
    commandLineArgs.addAll(spawnArguments);
    return commandLineArgs;
  }
}
