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
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.KillableObserver;
import com.google.devtools.build.lib.shell.TimeoutKillableObserver;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Helper class for running the namespace sandbox. This runner prepares environment inside the
 * sandbox, handles sandbox output, performs cleanup and changes invocation if necessary.
 */
final class DarwinSandboxRunner extends SandboxRunner {
  private static final String SANDBOX_EXEC = "/usr/bin/sandbox-exec";

  private final Path sandboxExecRoot;
  private final Path argumentsFilePath;
  private final Set<Path> writableDirs;
  private final Set<Path> inaccessiblePaths;
  private final Path runUnderPath;

  DarwinSandboxRunner(
      Path sandboxPath,
      Path sandboxExecRoot,
      Set<Path> writableDirs,
      Set<Path> inaccessiblePaths,
      Path runUnderPath,
      boolean verboseFailures) {
    super(sandboxExecRoot, verboseFailures);
    this.sandboxExecRoot = sandboxExecRoot;
    this.argumentsFilePath = sandboxPath.getRelative("sandbox.sb");
    this.writableDirs = writableDirs;
    this.inaccessiblePaths = inaccessiblePaths;
    this.runUnderPath = runUnderPath;
  }

  static boolean isSupported() {
    // Check osx version, only >=10.11 is supported.
    // And we should check if sandbox still work when it gets 11.x
    String osxVersion = OS.getVersion();
    String[] parts = osxVersion.split("\\.");
    if (parts.length != 3) {
      // Currently the format is 10.11.x
      return false;
    }
    try {
      int v0 = Integer.parseInt(parts[0]);
      int v1 = Integer.parseInt(parts[1]);
      if (v0 != 10 || v1 < 11) {
        return false;
      }
    } catch (NumberFormatException e) {
      return false;
    }

    List<String> args = new ArrayList<>();
    args.add(SANDBOX_EXEC);
    args.add("-p");
    args.add("(version 1) (allow default)");
    args.add("/usr/bin/true");

    ImmutableMap<String, String> env = ImmutableMap.of();
    File cwd = new File("/usr/bin");

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
      List<String> arguments, Map<String, String> environment, int timeout, boolean allowNetwork)
      throws IOException {
    writeConfig(allowNetwork);

    List<String> commandLineArgs = new ArrayList<>();
    commandLineArgs.add(SANDBOX_EXEC);
    commandLineArgs.add("-f");
    commandLineArgs.add(argumentsFilePath.getPathString());
    commandLineArgs.addAll(arguments);
    return new Command(
        commandLineArgs.toArray(new String[0]), environment, sandboxExecRoot.getPathFile());
  }

  private void writeConfig(boolean allowNetwork) throws IOException {
    try (PrintWriter out = new PrintWriter(argumentsFilePath.getOutputStream())) {
      // Note: In Apple's sandbox configuration language, the *last* matching rule wins.
      out.println("(version 1)");
      out.println("(debug deny)");
      out.println("(allow default)");

      if (!allowNetwork) {
        out.println("(deny network*)");
      }

      out.println("(allow network* (local ip \"localhost:*\"))");
      out.println("(allow network* (remote ip \"localhost:*\"))");

      for (Path inaccessiblePath : inaccessiblePaths) {
        out.println("(deny file-read* (subpath \"" + inaccessiblePath + "\"))");
      }
      if (runUnderPath != null) {
        out.println("(allow file-read* (subpath \"" + runUnderPath + "\"))");
      }

      // Almost everything else is read-only.
      out.println("(deny file-write* (subpath \"/\"))");

      allowWriteSubpath(out, sandboxExecRoot);
      for (Path path : writableDirs) {
        allowWriteSubpath(out, path);
      }
    }
  }

  private void allowWriteSubpath(PrintWriter out, Path path) throws IOException {
    out.println("(allow file-write* (subpath \"" + path.getPathString() + "\"))");
    Path resolvedPath = path.resolveSymbolicLinks();
    if (!resolvedPath.equals(path)) {
      out.println("(allow file-write* (subpath \"" + resolvedPath.getPathString() + "\"))");
    }
  }

  @Override
  protected KillableObserver getCommandObserver(int timeout) {
    return (timeout >= 0) ? new TimeoutKillableObserver(timeout * 1000) : Command.NO_OBSERVER;
  }

  @Override
  protected int getSignalOnTimeout() {
    return 15; /* SIGTERM */
  }
}
