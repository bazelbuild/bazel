// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableMultimap;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.io.IOException;
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
  private final Path execRoot;
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final ImmutableMultimap<Path, Path> mounts;
  private final boolean debug;

  public NamespaceSandboxRunner(
      Path execRoot, Path sandboxPath, ImmutableMultimap<Path, Path> mounts, boolean debug) {
    this.execRoot = execRoot;
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxPath.getRelative(execRoot.asFragment().relativeTo("/"));
    this.mounts = mounts;
    this.debug = debug;
  }

  static boolean isSupported(BlazeRuntime runtime) {
    Path execRoot = runtime.getExecRoot();
    BinTools binTools = runtime.getBinTools();

    List<String> args = new ArrayList<>();
    args.add(execRoot.getRelative(binTools.getExecPath(NAMESPACE_SANDBOX)).getPathString());
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
   * @throws CommandException
   */
  public void run(
      List<String> spawnArguments,
      ImmutableMap<String, String> env,
      File cwd,
      FileOutErr outErr,
      Collection<? extends ActionInput> outputs,
      int timeout)
      throws IOException, CommandException {
    createFileSystem(outputs);

    List<String> args = new ArrayList<>();

    args.add(execRoot.getRelative("_bin/namespace-sandbox").getPathString());

    if (debug) {
      args.add("-D");
    }

    // Sandbox directory.
    args.add("-S");
    args.add(sandboxPath.getPathString());

    // Working directory of the spawn.
    args.add("-W");
    args.add(cwd.toString());

    // Kill the process after a timeout.
    if (timeout != -1) {
      args.add("-T");
      args.add(Integer.toString(timeout));
    }

    // Mount all the inputs.
    for (ImmutableMap.Entry<Path, Path> mount : mounts.entries()) {
      args.add("-M");
      args.add(mount.getKey().getPathString());
      args.add("-m");
      args.add(mount.getValue().getPathString());
    }

    args.add("--");
    args.addAll(spawnArguments);

    Command cmd = new Command(args.toArray(new String[0]), env, cwd);

    cmd.execute(
        /* stdin */ new byte[] {},
        Command.NO_OBSERVER,
        outErr.getOutputStream(),
        outErr.getErrorStream(),
        /* killSubprocessOnInterrupt */ true);

    copyOutputs(outputs);
  }

  private void createFileSystem(Collection<? extends ActionInput> outputs) throws IOException {
    FileSystemUtils.createDirectoryAndParents(sandboxPath);

    // Prepare the output directories in the sandbox.
    for (ActionInput output : outputs) {
      PathFragment parentDirectory =
          new PathFragment(output.getExecPathString()).getParentDirectory();
      FileSystemUtils.createDirectoryAndParents(sandboxExecRoot.getRelative(parentDirectory));
    }
  }

  private void copyOutputs(Collection<? extends ActionInput> outputs) throws IOException {
    for (ActionInput output : outputs) {
      Path source = sandboxExecRoot.getRelative(output.getExecPathString());
      Path target = execRoot.getRelative(output.getExecPathString());
      FileSystemUtils.createDirectoryAndParents(target.getParentDirectory());
      if (source.isFile()) {
        Files.move(new File(source.getPathString()), new File(target.getPathString()));
      }
    }
  }

  public void cleanup() throws IOException {
    if (sandboxPath.exists()) {
      FilesystemUtils.rmTree(sandboxPath.getPathString());
    }
  }
}
