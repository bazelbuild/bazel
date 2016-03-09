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
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Helper class for running the namespace sandbox. This runner prepares environment inside the
 * sandbox, handles sandbox output, performs cleanup and changes invocation if necessary.
 */
public class NamespaceSandboxRunner {
  private static final String NAMESPACE_SANDBOX =
      "namespace-sandbox" + OsUtils.executableExtension();
  private static final String SANDBOX_TIP =
      "\n\nSandboxed execution failed, which may be legitimate (e.g. a compiler error), "
          + "or due to missing dependencies. To enter the sandbox environment for easier debugging,"
          + " run the following command in parentheses. On command failure, "
          + "a bash shell running inside the sandbox will then automatically be spawned:\n\n";
  private final Path execRoot;
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final Path argumentsFilePath;
  private final ImmutableMap<Path, Path> mounts;
  private final ImmutableSet<Path> createDirs;
  private final boolean verboseFailures;
  private final boolean sandboxDebug;

  public NamespaceSandboxRunner(
      Path execRoot,
      Path sandboxPath,
      ImmutableMap<Path, Path> mounts,
      ImmutableSet<Path> createDirs,
      boolean verboseFailures,
      boolean sandboxDebug) {
    this.execRoot = execRoot;
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxPath.getRelative(execRoot.asFragment().relativeTo("/"));
    this.argumentsFilePath =
        sandboxPath.getParentDirectory().getRelative(sandboxPath.getBaseName() + ".params");
    this.mounts = mounts;
    this.createDirs = createDirs;
    this.verboseFailures = verboseFailures;
    this.sandboxDebug = sandboxDebug;
  }

  static boolean isSupported(BlazeRuntime runtime) {
    Path execRoot = runtime.getExecRoot();
    BinTools binTools = runtime.getBinTools();

    PathFragment embeddedTool = binTools.getExecPath(NAMESPACE_SANDBOX);
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
   * @param cwd - current working directory
   * @param outErr - error output to capture sandbox's and command's stderr
   * @param outputs - files to extract from the sandbox, paths are relative to the exec root
   * @throws ExecException
   */
  public void run(
      List<String> spawnArguments,
      ImmutableMap<String, String> env,
      File cwd,
      FileOutErr outErr,
      Collection<PathFragment> outputs,
      int timeout,
      boolean blockNetwork)
      throws IOException, ExecException {
    createFileSystem(outputs);

    List<String> fileArgs = new ArrayList<>();
    List<String> commandLineArgs = new ArrayList<>();

    commandLineArgs.add(execRoot.getRelative("_bin/namespace-sandbox").getPathString());

    if (sandboxDebug) {
      fileArgs.add("-D");
    }

    // Sandbox directory.
    fileArgs.add("-S");
    fileArgs.add(sandboxPath.getPathString());

    // Working directory of the spawn.
    fileArgs.add("-W");
    fileArgs.add(cwd.toString());

    // Kill the process after a timeout.
    if (timeout != -1) {
      fileArgs.add("-T");
      fileArgs.add(Integer.toString(timeout));
    }

    // Create all needed directories.
    for (Path createDir : createDirs) {
      fileArgs.add("-d");
      fileArgs.add(createDir.getPathString());
    }

    if (blockNetwork) {
      // Block network access out of the namespace.
      fileArgs.add("-n");
    }

    // Mount all the inputs.
    for (ImmutableMap.Entry<Path, Path> mount : mounts.entrySet()) {
      fileArgs.add("-M");
      fileArgs.add(mount.getValue().getPathString());

      // The file is mounted in a custom location inside the sandbox.
      if (!mount.getValue().equals(mount.getKey())) {
        fileArgs.add("-m");
        fileArgs.add(mount.getKey().getPathString());
      }
    }

    FileSystemUtils.writeLinesAs(argumentsFilePath, StandardCharsets.ISO_8859_1, fileArgs);
    commandLineArgs.add("@" + argumentsFilePath.getPathString());

    commandLineArgs.add("--");
    commandLineArgs.addAll(spawnArguments);

    Command cmd = new Command(commandLineArgs.toArray(new String[0]), env, cwd);

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
              verboseFailures, commandLineArgs, env, cwd.getPath());
      String finalMsg = (sandboxDebug && verboseFailures) ? SANDBOX_TIP + message : message;
      throw new UserExecException(finalMsg, e, timedOut);
    } finally {
      copyOutputs(outputs);
    }
  }

  private void createFileSystem(Collection<PathFragment> outputs) throws IOException {
    FileSystemUtils.createDirectoryAndParents(sandboxPath);

    // Prepare the output directories in the sandbox.
    for (PathFragment output : outputs) {
      FileSystemUtils.createDirectoryAndParents(
          sandboxExecRoot.getRelative(output.getParentDirectory()));
    }
  }

  private void copyOutputs(Collection<PathFragment> outputs) throws IOException {
    for (PathFragment output : outputs) {
      Path source = sandboxExecRoot.getRelative(output);
      Path target = execRoot.getRelative(output);
      FileSystemUtils.createDirectoryAndParents(target.getParentDirectory());
      if (source.isFile() || source.isSymbolicLink()) {
        Files.move(source.getPathFile(), target.getPathFile());
      }
    }
  }

  public void cleanup() throws IOException {
    if (sandboxPath.exists()) {
      FileSystemUtils.deleteTree(sandboxPath);
    }
    if (!sandboxDebug && argumentsFilePath.exists()) {
      argumentsFilePath.delete();
    }
  }
}
