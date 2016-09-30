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
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Helper class for running the Linux sandbox. This runner prepares environment inside the sandbox,
 * handles sandbox output, performs cleanup and changes invocation if necessary.
 */
final class LinuxSandboxRunner extends SandboxRunner {
  protected static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();

  private final Path execRoot;
  private final Path sandboxExecRoot;
  private final Path sandboxTempDir;
  private final Path argumentsFilePath;
  private final Set<Path> writableDirs;
  private final Set<Path> inaccessiblePaths;
  private final Set<Path> tmpfsPaths;
  private final Set<Path> bindMounts;
  private final boolean sandboxDebug;

  LinuxSandboxRunner(
      Path execRoot,
      Path sandboxPath,
      Path sandboxExecRoot,
      Path sandboxTempDir,
      Set<Path> writableDirs,
      Set<Path> inaccessiblePaths,
      Set<Path> tmpfsPaths,
      Set<Path> bindMounts,
      boolean verboseFailures,
      boolean sandboxDebug) {
    super(sandboxExecRoot, verboseFailures);
    this.execRoot = execRoot;
    this.sandboxExecRoot = sandboxExecRoot;
    this.sandboxTempDir = sandboxTempDir;
    this.argumentsFilePath = sandboxPath.getRelative("linux-sandbox.params");
    this.writableDirs = writableDirs;
    this.inaccessiblePaths = inaccessiblePaths;
    this.tmpfsPaths = tmpfsPaths;
    this.bindMounts = bindMounts;
    this.sandboxDebug = sandboxDebug;
  }

  static boolean isSupported(CommandEnvironment commandEnv) {
    Path execRoot = commandEnv.getExecRoot();

    PathFragment embeddedTool =
        commandEnv.getBlazeWorkspace().getBinTools().getExecPath(LINUX_SANDBOX);
    if (embeddedTool == null) {
      // The embedded tool does not exist, meaning that we don't support sandboxing (e.g., while
      // bootstrapping).
      return false;
    }

    List<String> args = new ArrayList<>();
    args.add(execRoot.getRelative(embeddedTool).getPathString());
    args.add("-C");

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
      List<String> spawnArguments, Map<String, String> env, int timeout, boolean allowNetwork)
      throws IOException {
    writeConfig(timeout, allowNetwork);

    List<String> commandLineArgs = new ArrayList<>(3 + spawnArguments.size());
    commandLineArgs.add(execRoot.getRelative("_bin/linux-sandbox").getPathString());
    commandLineArgs.add("@" + argumentsFilePath.getPathString());
    commandLineArgs.add("--");
    commandLineArgs.addAll(spawnArguments);
    return new Command(commandLineArgs.toArray(new String[0]), env, sandboxExecRoot.getPathFile());
  }

  private void writeConfig(int timeout, boolean allowNetwork) throws IOException {
    List<String> fileArgs = new ArrayList<>();

    if (sandboxDebug) {
      fileArgs.add("-D");
    }

    // Temporary directory of the sandbox.
    fileArgs.add("-S");
    fileArgs.add(sandboxTempDir.toString());

    // Working directory of the spawn.
    fileArgs.add("-W");
    fileArgs.add(sandboxExecRoot.toString());

    // Kill the process after a timeout.
    if (timeout != -1) {
      fileArgs.add("-T");
      fileArgs.add(Integer.toString(timeout));
    }

    // Create all needed directories.
    for (Path writablePath : writableDirs) {
      fileArgs.add("-w");
      fileArgs.add(writablePath.getPathString());
    }

    for (Path inaccessiblePath : inaccessiblePaths) {
      fileArgs.add("-i");
      fileArgs.add(inaccessiblePath.getPathString());
    }

    for (Path tmpfsPath : tmpfsPaths) {
      fileArgs.add("-e");
      fileArgs.add(tmpfsPath.getPathString());
    }

    for (Path bindMount : bindMounts) {
      fileArgs.add("-b");
      fileArgs.add(bindMount.getPathString());
    }

    if (!allowNetwork) {
      // Block network access out of the namespace.
      fileArgs.add("-N");
    }

    FileSystemUtils.writeLinesAs(argumentsFilePath, StandardCharsets.ISO_8859_1, fileArgs);
  }
}
