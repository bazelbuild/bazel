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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Helper class for running the Linux sandbox. This runner is a subclass of LinuxSandboxRunner which
 * uses process-wrapper instead of linux-sandbox in the same sandbox execution environment.
 */
public class LinuxAlmostSandboxRunner extends LinuxSandboxRunner {

  LinuxAlmostSandboxRunner(
      Path execRoot,
      Path sandboxExecRoot,
      Set<Path> writablePaths,
      List<Path> inaccessiblePaths,
      boolean verboseFailures,
      boolean sandboxDebug) {
    super(
        execRoot, sandboxExecRoot, writablePaths, inaccessiblePaths, verboseFailures, sandboxDebug);
  }

  static boolean isSupported(CommandEnvironment commandEnv) {
    PathFragment embeddedTool =
        commandEnv
            .getBlazeWorkspace()
            .getBinTools()
            .getExecPath("process-wrapper" + OsUtils.executableExtension());
    if (embeddedTool == null) {
      // The embedded tool does not exist, meaning that we don't support sandboxing (e.g., while
      // bootstrapping).
      return false;
    }
    return true;
  }

  @Override
  protected void runPreparation(int timeout, boolean allowNetwork) throws IOException {
    // Create all needed directories.
    for (Path writablePath : super.getWritablePaths()) {
      if (writablePath.startsWith(super.getSandboxExecRoot())) {
        FileSystemUtils.createDirectoryAndParents(writablePath);
      }
    }
  }

  @Override
  protected List<String> runCommandLineArgs(List<String> spawnArguments, int timeout) {
    List<String> commandLineArgs = new ArrayList<>();

    commandLineArgs.add(super.getExecRoot().getRelative("_bin/process-wrapper").getPathString());
    commandLineArgs.add(Integer.toString(timeout));
    commandLineArgs.add("5"); /* kill delay: give some time to print stacktraces and whatnot. */
    commandLineArgs.add("-"); /* stdout. */
    commandLineArgs.add("-"); /* stderr. */

    commandLineArgs.addAll(spawnArguments);

    return commandLineArgs;
  }
}
