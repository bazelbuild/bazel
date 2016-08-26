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
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Helper class for running the Linux sandbox. This runner prepares environment inside the sandbox,
 * handles sandbox output, performs cleanup and changes invocation if necessary.
 */
public class LinuxSandboxRunner {
  private static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();
  private final Path execRoot;
  private final Path sandboxExecRoot;
  private final Path argumentsFilePath;
  private final Set<Path> writablePaths;
  private final List<Path> inaccessiblePaths;
  private final boolean verboseFailures;
  private final boolean sandboxDebug;

  LinuxSandboxRunner(
      Path execRoot,
      Path sandboxExecRoot,
      Set<Path> writablePaths,
      List<Path> inaccessiblePaths,
      boolean verboseFailures,
      boolean sandboxDebug) {
    this.execRoot = execRoot;
    this.sandboxExecRoot = sandboxExecRoot;
    this.argumentsFilePath =
        sandboxExecRoot.getParentDirectory().getRelative(sandboxExecRoot.getBaseName() + ".params");
    this.writablePaths = writablePaths;
    this.inaccessiblePaths = inaccessiblePaths;
    this.verboseFailures = verboseFailures;
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

  /**
   * Runs given
   *
   * @param spawnArguments - arguments of spawn to run inside the sandbox
   * @param env - environment to run sandbox in
   * @param outErr - error output to capture sandbox's and command's stderr
   * @param outputs - files to extract from the sandbox, paths are relative to the exec root @throws
   *     ExecException
   */
  public void run(
      List<String> spawnArguments,
      Map<String, String> env,
      FileOutErr outErr,
      Map<PathFragment, Path> inputs,
      Collection<PathFragment> outputs,
      int timeout,
      boolean allowNetwork)
      throws IOException, ExecException {
    createFileSystem(inputs, outputs);

    List<String> fileArgs = new ArrayList<>();
    List<String> commandLineArgs = new ArrayList<>();

    commandLineArgs.add(execRoot.getRelative("_bin/linux-sandbox").getPathString());

    if (sandboxDebug) {
      fileArgs.add("-D");
    }

    // Working directory of the spawn.
    fileArgs.add("-W");
    fileArgs.add(sandboxExecRoot.toString());

    // Kill the process after a timeout.
    if (timeout != -1) {
      fileArgs.add("-T");
      fileArgs.add(Integer.toString(timeout));
    }

    // Create all needed directories.
    for (Path writablePath : writablePaths) {
      fileArgs.add("-w");
      fileArgs.add(writablePath.getPathString());
      if (writablePath.startsWith(sandboxExecRoot)) {
        FileSystemUtils.createDirectoryAndParents(writablePath);
      }
    }

    for (Path inaccessiblePath : inaccessiblePaths) {
      fileArgs.add("-i");
      fileArgs.add(inaccessiblePath.getPathString());
    }

    if (!allowNetwork) {
      // Block network access out of the namespace.
      fileArgs.add("-N");
    }

    FileSystemUtils.writeLinesAs(argumentsFilePath, StandardCharsets.ISO_8859_1, fileArgs);
    commandLineArgs.add("@" + argumentsFilePath.getPathString());

    commandLineArgs.add("--");
    commandLineArgs.addAll(spawnArguments);

    Command cmd =
        new Command(commandLineArgs.toArray(new String[0]), env, sandboxExecRoot.getPathFile());

    try {
      cmd.execute(
          /* stdin */ new byte[] {},
          Command.NO_OBSERVER,
          outErr.getOutputStream(),
          outErr.getErrorStream(),
          /* killSubprocessOnInterrupt */ true);
    } catch (CommandException e) {
      boolean timedOut = false;
      if (e instanceof AbnormalTerminationException) {
        TerminationStatus status =
            ((AbnormalTerminationException) e).getResult().getTerminationStatus();
        timedOut = !status.exited() && (status.getTerminatingSignal() == 14 /* SIGALRM */);
      }
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, commandLineArgs, env, sandboxExecRoot.getPathString());
      throw new UserExecException(message, e, timedOut);
    } finally {
      copyOutputs(outputs);
    }
  }

  private void createFileSystem(Map<PathFragment, Path> inputs, Collection<PathFragment> outputs)
      throws IOException {
    Set<Path> createdDirs = new HashSet<>();
    FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, sandboxExecRoot);
    createParentDirectoriesForInputs(createdDirs, inputs.keySet());
    createSymlinksForInputs(inputs);
    createDirectoriesForOutputs(createdDirs, outputs);
  }

  /**
   * No input can be a child of another input, because otherwise we might try to create a symlink
   * below another symlink we created earlier - which means we'd actually end up writing somewhere
   * in the workspace.
   *
   * <p>If all inputs were regular files, this situation could naturally not happen - but
   * unfortunately, we might get the occasional action that has directories in its inputs.
   *
   * <p>Creating all parent directories first ensures that we can safely create symlinks to
   * directories, too, because we'll get an IOException with EEXIST if inputs happen to be nested
   * once we start creating the symlinks for all inputs.
   */
  private void createParentDirectoriesForInputs(Set<Path> createdDirs, Set<PathFragment> inputs)
      throws IOException {
    for (PathFragment inputPath : inputs) {
      Path dir = sandboxExecRoot.getRelative(inputPath).getParentDirectory();
      Preconditions.checkArgument(dir.startsWith(sandboxExecRoot));
      FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, dir);
    }
  }

  private void createSymlinksForInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = sandboxExecRoot.getRelative(entry.getKey());
      key.createSymbolicLink(entry.getValue());
    }
  }

  /** Prepare the output directories in the sandbox. */
  private void createDirectoriesForOutputs(Set<Path> createdDirs, Collection<PathFragment> outputs)
      throws IOException {
    for (PathFragment output : outputs) {
      FileSystemUtils.createDirectoryAndParentsWithCache(
          createdDirs, sandboxExecRoot.getRelative(output.getParentDirectory()));
      FileSystemUtils.createDirectoryAndParentsWithCache(
          createdDirs, execRoot.getRelative(output.getParentDirectory()));
    }
  }

  private void copyOutputs(Collection<PathFragment> outputs) throws IOException {
    for (PathFragment output : outputs) {
      Path source = sandboxExecRoot.getRelative(output);
      if (source.isFile() || source.isSymbolicLink()) {
        Path target = execRoot.getRelative(output);
        Files.move(source.getPathFile(), target.getPathFile());
      }
    }
  }

  public void cleanup() throws IOException {
    if (sandboxExecRoot.exists()) {
      FileSystemUtils.deleteTree(sandboxExecRoot);
    }
    if (argumentsFilePath.exists()) {
      argumentsFilePath.delete();
    }
  }
}
