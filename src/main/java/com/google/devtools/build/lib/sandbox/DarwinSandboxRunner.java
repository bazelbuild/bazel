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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.shell.TimeoutKillableObserver;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Helper class for running the namespace sandbox. This runner prepares environment inside the
 * sandbox, handles sandbox output, performs cleanup and changes invocation if necessary.
 */
public class DarwinSandboxRunner {

  private final Path execRoot;
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final Path argumentsFilePath;
  private final ImmutableSet<PathFragment> createDirs;
  private final boolean verboseFailures;
  private final boolean sandboxDebug;

  private final Path sandboxConfigPath;
  private final ImmutableMap<PathFragment, Path> linkPaths;

  public DarwinSandboxRunner(
      Path execRoot,
      Path sandboxPath,
      Path sandboxExecRoot,
      Path sandboxConfigPath,
      ImmutableMap<PathFragment, Path> linkPaths,
      ImmutableSet<PathFragment> createDirs,
      boolean verboseFailures,
      boolean sandboxDebug) {
    this.execRoot = execRoot;
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxExecRoot;
    this.argumentsFilePath =
        sandboxPath.getParentDirectory().getRelative(sandboxPath.getBaseName() + ".params");
    this.createDirs = createDirs;
    this.verboseFailures = verboseFailures;
    this.sandboxDebug = sandboxDebug;
    this.sandboxConfigPath = sandboxConfigPath;
    this.linkPaths = linkPaths;
  }

  static boolean isSupported() {
    List<String> args = new ArrayList<>();
    args.add("sandbox-exec");
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

  /**
   * Runs given command inside the sandbox.
   *
   * @param spawnArguments - arguments of spawn to run inside the sandbox
   * @param env - environment to run sandbox in
   * @param outErr - error output to capture sandbox's and command's stderr
   * @param outputs - files to extract from the sandbox, paths are relative to the exec root
   * @throws ExecException
   */
  public void run(
      List<String> spawnArguments,
      ImmutableMap<String, String> env,
      FileOutErr outErr,
      Collection<PathFragment> outputs,
      int timeout)
      throws IOException, ExecException {
    createFileSystem(outputs);

    List<String> commandLineArgs = sandboxPreperationAndGetArgs(spawnArguments, outErr);

    Command cmd =
        new Command(commandLineArgs.toArray(new String[0]), env, sandboxExecRoot.getPathFile());

    try {
      cmd.execute(
          /* stdin */ new byte[] {},
          (timeout >= 0) ? new TimeoutKillableObserver(timeout * 1000) : Command.NO_OBSERVER,
          outErr.getOutputStream(),
          outErr.getErrorStream(),
          /* killSubprocessOnInterrupt */ true);
    } catch (CommandException e) {
      boolean timedOut = false;
      if (e instanceof AbnormalTerminationException) {
        TerminationStatus status =
            ((AbnormalTerminationException) e).getResult().getTerminationStatus();
        timedOut = !status.exited() && (status.getTerminatingSignal() == 15 /* SIGTERM */);
      }
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, commandLineArgs, env, sandboxExecRoot.getPathString());
      throw new UserExecException(message, e, timedOut);
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
        com.google.common.io.Files.move(source.getPathFile(), target.getPathFile());
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

  private List<String> sandboxPreperationAndGetArgs(List<String> spawnArguments, FileOutErr outErr)
      throws IOException {
    FileSystem fs = sandboxPath.getFileSystem();
    PrintWriter errWriter = new PrintWriter(outErr.getErrorStream());
    List<String> commandLineArgs = new ArrayList<>();

    if (sandboxDebug) {
      errWriter.printf("sandbox root is %s\n", sandboxPath.toString());
      errWriter.printf("working dir is %s\n", sandboxExecRoot.toString());
    }

    // Create all needed directories.
    for (PathFragment createDir : createDirs) {
      Path dir;
      if (createDir.isAbsolute()) {
        dir = fs.getPath(createDir);
      } else {
        dir = sandboxPath.getRelative(createDir);
      }
      if (sandboxDebug) {
        errWriter.printf("createdir: %s\n", dir);
      }
      FileSystemUtils.createDirectoryAndParents(dir);
    }

    // Link all the inputs.
    linkInputs(linkPaths, errWriter);

    errWriter.flush();

    commandLineArgs.add("/usr/bin/sandbox-exec");
    commandLineArgs.add("-f");
    commandLineArgs.add(sandboxConfigPath.getPathString());
    commandLineArgs.addAll(spawnArguments);

    return commandLineArgs;
  }

  /**
   * Make all specified inputs available in the sandbox.
   *
   * We want the sandboxed process to have access only to these input files and not anything else
   * from the workspace. Furthermore, the process should not be able to modify these input files.
   * We achieve this by hardlinking all input files into a temporary "inputs" directory, then
   * symlinking them into their correct place inside the sandbox.
   *
   * The hardlinks / symlinks combination (as opposed to simply directly hardlinking to the final
   * destination) is necessary, because we build a solib symlink tree for shared libraries where the
   * original file and the created symlink have two different file names (libblaze_util.so vs.
   * src_Stest_Scpp_Sblaze_Uutil_Utest.so) and our cc_wrapper.sh needs to be able to figure out both
   * names (by following solib symlinks back) to modify the paths to the shared libraries in
   * cc_binaries.
   */
  private void linkInputs(ImmutableMap<PathFragment, Path> inputs, PrintWriter errWriter)
      throws IOException {
    // create directory for input files
    Path inputsDir = sandboxPath.getRelative("inputs");
    if (!inputsDir.exists()) {
      inputsDir.createDirectory();
    }

    for (ImmutableMap.Entry<PathFragment, Path> entry : inputs.entrySet()) {
      // hardlink, resolve symlink here instead in finalizeLinks
      Path hardlinkOldPath = entry.getValue().resolveSymbolicLinks();
      Path hardlinkNewPath =
          hardlinkOldPath.startsWith(execRoot)
              ? inputsDir.getRelative(hardlinkOldPath.relativeTo(execRoot))
              : inputsDir.getRelative(entry.getKey());
      if (sandboxDebug) {
        errWriter.printf("hardlink: %s -> %s\n", hardlinkNewPath, hardlinkOldPath);
      }
      try {
        createHardLink(hardlinkNewPath, hardlinkOldPath);
      } catch (IOException e) {
        // Creating a hardlink might fail when the input file and the sandbox directory are not on
        // the same filesystem / device. Then we use symlink instead.
        hardlinkNewPath.createSymbolicLink(hardlinkOldPath);
      }

      // symlink
      Path symlinkNewPath = sandboxExecRoot.getRelative(entry.getKey());
      if (sandboxDebug) {
        errWriter.printf("symlink: %s -> %s\n", hardlinkNewPath, symlinkNewPath);
      }
      FileSystemUtils.createDirectoryAndParents(symlinkNewPath.getParentDirectory());
      symlinkNewPath.createSymbolicLink(hardlinkNewPath);
    }
  }

  // TODO(yueg): import unix.FilesystemUtils and use FilesystemUtils.createHardLink() instead
  private void createHardLink(Path target, Path source) throws IOException {
    java.nio.file.Path targetNio = java.nio.file.Paths.get(target.toString());
    java.nio.file.Path sourceNio = java.nio.file.Paths.get(source.toString());

    if (!source.exists() || target.exists()) {
      return;
    }
    // Regular file
    if (source.isFile()) {
      Path parentDir = target.getParentDirectory();
      if (!parentDir.exists()) {
        FileSystemUtils.createDirectoryAndParents(parentDir);
      }
      Files.createLink(targetNio, sourceNio);
      // Directory
    } else if (source.isDirectory()) {
      Collection<Path> subpaths = source.getDirectoryEntries();
      for (Path sourceSubpath : subpaths) {
        Path targetSubpath = target.getRelative(sourceSubpath.relativeTo(source));
        createHardLink(targetSubpath, sourceSubpath);
      }
    }
  }
}
