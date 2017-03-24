// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Helper class for running the Linux sandbox. This runner prepares environment inside the sandbox,
 * handles sandbox output, performs cleanup and changes invocation if necessary.
 */
final class LinuxSandboxRunner extends SandboxRunner {
  private static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();

  private final Path execRoot;
  private final Path sandboxExecRoot;
  private final Set<Path> writableDirs;
  private final Set<Path> tmpfsPaths;
  // a <target, source> mapping of paths to bind mount
  private final Map<Path, Path> bindMounts;
  private final boolean sandboxDebug;

  LinuxSandboxRunner(
      Path execRoot,
      Path sandboxExecRoot,
      Set<Path> writableDirs,
      Set<Path> tmpfsPaths,
      Map<Path, Path> bindMounts,
      boolean verboseFailures,
      boolean sandboxDebug) {
    super(verboseFailures);
    this.execRoot = execRoot;
    this.sandboxExecRoot = sandboxExecRoot;
    this.writableDirs = writableDirs;
    this.tmpfsPaths = tmpfsPaths;
    this.bindMounts = bindMounts;
    this.sandboxDebug = sandboxDebug;
  }

  static boolean isSupported(CommandEnvironment commandEnv) {
    PathFragment embeddedTool =
        commandEnv.getBlazeWorkspace().getBinTools().getExecPath(LINUX_SANDBOX);
    if (embeddedTool == null) {
      // The embedded tool does not exist, meaning that we don't support sandboxing (e.g., while
      // bootstrapping).
      return false;
    }

    Path execRoot = commandEnv.getExecRoot();

    List<String> args = new ArrayList<>();
    args.add(execRoot.getRelative(embeddedTool).getPathString());
    args.add("--");
    args.add("/bin/true");

    ImmutableMap<String, String> env = ImmutableMap.of();
    File cwd = execRoot.getPathFile();

    Command cmd = new Command(args.toArray(new String[0]), env, cwd);
    try {
      cmd.execute(
          /* stdin */ new byte[] {},
          Command.NO_OBSERVER,
          ByteStreams.nullOutputStream(),
          ByteStreams.nullOutputStream(),
          /* killSubprocessOnInterrupt */ true);
    } catch (CommandException e) {
      return false;
    }

    return true;
  }

  @Override
  protected Command getCommand(
      List<String> spawnArguments,
      Map<String, String> env,
      int timeout,
      boolean allowNetwork,
      boolean useFakeHostname,
      boolean useFakeUsername)
      throws IOException {
    List<String> commandLineArgs = new ArrayList<>();
    commandLineArgs.add(execRoot.getRelative("_bin/linux-sandbox").getPathString());

    if (sandboxDebug) {
      commandLineArgs.add("-D");
    }

    // Kill the process after a timeout.
    if (timeout != -1) {
      commandLineArgs.add("-T");
      commandLineArgs.add(Integer.toString(timeout));
    }

    // Create all needed directories.
    for (Path writablePath : writableDirs) {
      commandLineArgs.add("-w");
      commandLineArgs.add(writablePath.getPathString());
    }

    for (Path tmpfsPath : tmpfsPaths) {
      commandLineArgs.add("-e");
      commandLineArgs.add(tmpfsPath.getPathString());
    }

    for (ImmutableMap.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      commandLineArgs.add("-M");
      commandLineArgs.add(bindMount.getValue().getPathString());

      // The file is mounted in a custom location inside the sandbox.
      if (!bindMount.getKey().equals(bindMount.getValue())) {
        commandLineArgs.add("-m");
        commandLineArgs.add(bindMount.getKey().getPathString());
      }
    }

    if (!allowNetwork) {
      // Block network access out of the namespace.
      commandLineArgs.add("-N");
    }

    if (useFakeHostname) {
      // Use a fake hostname ("localhost") inside the sandbox.
      commandLineArgs.add("-H");
    }

    if (useFakeUsername) {
      // Use a fake username ("nobody") inside the sandbox.
      commandLineArgs.add("-U");
    }

    commandLineArgs.add("--");
    commandLineArgs.addAll(spawnArguments);
    return new Command(commandLineArgs.toArray(new String[0]), env, sandboxExecRoot.getPathFile());
  }

}
