// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;

/** Spawn runner that uses linux sandboxing APIs to execute a local subprocess. */
final class LinuxSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {
  private static final String LINUX_SANDBOX = "linux-sandbox";

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    if (OS.getCurrent() != OS.LINUX) {
      return false;
    }
    Path embeddedTool = getLinuxSandbox(cmdEnv);
    if (embeddedTool == null) {
      // The embedded tool does not exist, meaning that we don't support sandboxing (e.g., while
      // bootstrapping).
      return false;
    }

    Path execRoot = cmdEnv.getExecRoot();

    List<String> args = new ArrayList<>();
    args.add(embeddedTool.getPathString());
    args.add("--");
    args.add("/bin/true");

    ImmutableMap<String, String> env = ImmutableMap.of();
    File cwd = execRoot.getPathFile();

    Command cmd = new Command(args.toArray(new String[0]), env, cwd);
    try {
      cmd.execute(ByteStreams.nullOutputStream(), ByteStreams.nullOutputStream());
    } catch (CommandException e) {
      return false;
    }

    return true;
  }

  private static Path getLinuxSandbox(CommandEnvironment cmdEnv) {
    PathFragment execPath = cmdEnv.getBlazeWorkspace().getBinTools().getExecPath(LINUX_SANDBOX);
    return execPath != null ? cmdEnv.getExecRoot().getRelative(execPath) : null;
  }

  private final FileSystem fileSystem;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean allowNetwork;
  private final Path linuxSandbox;
  private final Path inaccessibleHelperFile;
  private final Path inaccessibleHelperDir;
  private final LocalEnvProvider localEnvProvider;
  private final int timeoutGraceSeconds;
  private final String productName;

  LinuxSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir,
      int timeoutGraceSeconds) {
    super(cmdEnv, sandboxBase);
    this.fileSystem = cmdEnv.getRuntime().getFileSystem();
    this.blazeDirs = cmdEnv.getDirectories();
    this.execRoot = cmdEnv.getExecRoot();
    this.productName = productName;
    this.allowNetwork = SandboxHelpers.shouldAllowNetwork(cmdEnv.getOptions());
    this.linuxSandbox = getLinuxSandbox(cmdEnv);
    this.inaccessibleHelperFile = inaccessibleHelperFile;
    this.inaccessibleHelperDir = inaccessibleHelperDir;
    this.timeoutGraceSeconds = timeoutGraceSeconds;
    this.localEnvProvider = LocalEnvProvider.ADD_TEMP_POSIX;
  }

  @Override
  protected SpawnResult actuallyExec(Spawn spawn, SpawnExecutionPolicy policy)
      throws IOException, ExecException, InterruptedException {
    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = getSandboxRoot();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    // Each sandboxed action runs in its own execroot, so we don't need to make the temp directory's
    // name unique (like we have to with standalone execution strategy).
    Path tmpDir = sandboxExecRoot.getRelative("tmp");

    Set<Path> writableDirs = getWritableDirs(sandboxExecRoot, spawn.getEnvironment(), tmpDir);
    ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);
    Duration timeout = policy.getTimeout();
    List<String> arguments =
        computeCommandLine(
            spawn,
            timeout,
            linuxSandbox,
            writableDirs,
            getTmpfsPaths(),
            getReadOnlyBindMounts(blazeDirs, sandboxExecRoot),
            allowNetwork || Spawns.requiresNetwork(spawn),
            spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_FAKEROOT));
    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, tmpDir, productName);

    SandboxedSpawn sandbox =
        new SymlinkedSandboxedSpawn(
            sandboxPath,
            sandboxExecRoot,
            arguments,
            environment,
            SandboxHelpers.getInputFiles(spawn, policy, execRoot),
            outputs,
            writableDirs);
    return runSpawn(spawn, sandbox, policy, execRoot, tmpDir, timeout);
  }

  private List<String> computeCommandLine(
      Spawn spawn,
      Duration timeout,
      Path linuxSandbox,
      Set<Path> writableDirs,
      Set<Path> tmpfsPaths,
      Map<Path, Path> bindMounts,
      boolean allowNetwork,
      boolean requiresFakeRoot) {
    List<String> commandLineArgs = new ArrayList<>();
    commandLineArgs.add(linuxSandbox.getPathString());

    if (getSandboxOptions().sandboxDebug) {
      commandLineArgs.add("-D");
    }

    // Kill the process after a timeout.
    if (!timeout.isZero()) {
      commandLineArgs.add("-T");
      commandLineArgs.add(Long.toString(timeout.getSeconds()));
    }

    if (timeoutGraceSeconds != -1) {
      commandLineArgs.add("-t");
      commandLineArgs.add(Integer.toString(timeoutGraceSeconds));
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

    if (getSandboxOptions().sandboxFakeHostname) {
      // Use a fake hostname ("localhost") inside the sandbox.
      commandLineArgs.add("-H");
    }

    if (requiresFakeRoot) {
      // Use fake root.
      commandLineArgs.add("-R");
    } else if (getSandboxOptions().sandboxFakeUsername) {
      // Use a fake username ("nobody") inside the sandbox.
      commandLineArgs.add("-U");
    }

    commandLineArgs.add("--");
    commandLineArgs.addAll(spawn.getArguments());
    return commandLineArgs;
  }

  @Override
  protected String getName() {
    return "linux-sandbox";
  }

  @Override
  protected ImmutableSet<Path> getWritableDirs(
      Path sandboxExecRoot, Map<String, String> env, Path tmpDir) throws IOException {
    ImmutableSet.Builder<Path> writableDirs = ImmutableSet.builder();
    writableDirs.addAll(super.getWritableDirs(sandboxExecRoot, env, tmpDir));

    FileSystem fs = sandboxExecRoot.getFileSystem();
    writableDirs.add(fs.getPath("/dev/shm").resolveSymbolicLinks());
    writableDirs.add(fs.getPath("/tmp"));

    return writableDirs.build();
  }

  private ImmutableSet<Path> getTmpfsPaths() {
    ImmutableSet.Builder<Path> tmpfsPaths = ImmutableSet.builder();
    for (String tmpfsPath : getSandboxOptions().sandboxTmpfsPath) {
      tmpfsPaths.add(fileSystem.getPath(tmpfsPath));
    }
    return tmpfsPaths.build();
  }

  private SortedMap<Path, Path> getReadOnlyBindMounts(
      BlazeDirectories blazeDirs, Path sandboxExecRoot) throws UserExecException {
    Path tmpPath = fileSystem.getPath("/tmp");
    final SortedMap<Path, Path> bindMounts = Maps.newTreeMap();
    if (blazeDirs.getWorkspace().startsWith(tmpPath)) {
      bindMounts.put(blazeDirs.getWorkspace(), blazeDirs.getWorkspace());
    }
    if (blazeDirs.getOutputBase().startsWith(tmpPath)) {
      bindMounts.put(blazeDirs.getOutputBase(), blazeDirs.getOutputBase());
    }
    for (ImmutableMap.Entry<String, String> additionalMountPath :
        getSandboxOptions().sandboxAdditionalMounts) {
      try {
        final Path mountTarget = fileSystem.getPath(additionalMountPath.getValue());
        // If source path is relative, treat it as a relative path inside the execution root
        final Path mountSource = sandboxExecRoot.getRelative(additionalMountPath.getKey());
        // If a target has more than one source path, the latter one will take effect.
        bindMounts.put(mountTarget, mountSource);
      } catch (IllegalArgumentException e) {
        throw new UserExecException(
            String.format("Error occurred when analyzing bind mount pairs. %s", e.getMessage()));
      }
    }
    for (Path inaccessiblePath : getInaccessiblePaths()) {
      if (inaccessiblePath.isDirectory(Symlinks.NOFOLLOW)) {
        bindMounts.put(inaccessiblePath, inaccessibleHelperDir);
      } else {
        bindMounts.put(inaccessiblePath, inaccessibleHelperFile);
      }
    }
    validateBindMounts(bindMounts);
    return bindMounts;
  }

  /**
   * This method does the following things: - If mount source does not exist on the host system,
   * throw an error message - If mount target exists, check whether the source and target are of the
   * same type - If mount target does not exist on the host system, throw an error message
   *
   * @param bindMounts the bind mounts map with target as key and source as value
   * @throws UserExecException
   */
  private void validateBindMounts(SortedMap<Path, Path> bindMounts) throws UserExecException {
    for (SortedMap.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      final Path source = bindMount.getValue();
      final Path target = bindMount.getKey();
      // Mount source should exist in the file system
      if (!source.exists()) {
        throw new UserExecException(String.format("Mount source '%s' does not exist.", source));
      }
      // If target exists, but is not of the same type as the source, then we cannot mount it.
      if (target.exists()) {
        boolean areBothDirectories = source.isDirectory() && target.isDirectory();
        boolean isSourceFile = source.isFile() || source.isSymbolicLink();
        boolean isTargetFile = target.isFile() || target.isSymbolicLink();
        boolean areBothFiles = isSourceFile && isTargetFile;
        if (!(areBothDirectories || areBothFiles)) {
          // Source and target are not of the same type; we cannot mount it.
          throw new UserExecException(
              String.format(
                  "Mount target '%s' is not of the same type as mount source '%s'.",
                  target, source));
        }
      } else {
        // Mount target should exist in the file system
        throw new UserExecException(
            String.format(
                "Mount target '%s' does not exist. Bazel only supports bind mounting on top of "
                    + "existing files/directories. Please create an empty file or directory at "
                    + "the mount target path according to the type of mount source.",
                target));
      }
    }
  }
}
