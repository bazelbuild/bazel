// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.WindowsLocalEnvProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
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
import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** Spawn runner that uses BuildXL Sandbox APIs to execute a local subprocess. */
final class WindowsSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  private final FileSystem fileSystem;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final PathFragment windowsSandbox;
  private final Path sandboxBase;
  private final LocalEnvProvider localEnvProvider;
  private final Duration timeoutKillDelay;
  private final @Nullable SandboxfsProcess sandboxfsProcess;
  private final boolean sandboxfsMapSymlinkTargets;
  private final TreeDeleter treeDeleter;

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   * @param sandboxfsProcess instance of the sandboxfs process to use; may be null for none, in
   *     which case the runner uses a symlinked sandbox
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   */
  WindowsSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Duration timeoutKillDelay,
      PathFragment windowsSandboxPath,
      boolean sandboxfsMapSymlinkTargets,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.fileSystem = cmdEnv.getRuntime().getFileSystem();
    this.blazeDirs = cmdEnv.getDirectories();
    this.execRoot = cmdEnv.getExecRoot();
    this.windowsSandbox = windowsSandboxPath;
    this.sandboxBase = sandboxBase;
    this.timeoutKillDelay = timeoutKillDelay;
    this.sandboxfsProcess = null;
    this.sandboxfsMapSymlinkTargets = sandboxfsMapSymlinkTargets;
    this.localEnvProvider = new WindowsLocalEnvProvider(cmdEnv.getClientEnv());
    this.treeDeleter = treeDeleter;
  }

  @Override
  protected SpawnResult actuallyExec(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ExecException, InterruptedException {
    // Each invocation of "exec" gets its own sandbox base.
    // Note that the value returned by context.getId() is only unique inside one given SpawnRunner,
    // so we have to prefix our name to turn it into a globally unique value.
    Path sandboxPath =
        sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));
    sandboxPath.getParentDirectory().createDirectory();
    sandboxPath.createDirectory();

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());
    sandboxExecRoot.getParentDirectory().createDirectory();
    sandboxExecRoot.createDirectory();

    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    ImmutableSet<Path> writableDirs = getWritableDirs(sandboxExecRoot, environment);
    SandboxOutputs outputs = SandboxHelpers.getOutputs(spawn);
    Duration timeout = context.getTimeout();

    WindowsSandboxUtil.CommandLineBuilder commandLineBuilder =
        WindowsSandboxUtil.commandLineBuilder(windowsSandbox, spawn.getArguments())
            .setWritableFilesAndDirectories(writableDirs)
            .setUseDebugMode(getSandboxOptions().sandboxDebug)
            .setKillDelay(timeoutKillDelay);

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }

    Path statisticsPath = null;

    SandboxedSpawn sandbox =
          new WindowsSandboxedSpawn(
              sandboxPath,
              sandboxExecRoot,
              commandLineBuilder.build(),
              environment,
              SandboxHelpers.processInputFiles(
                  spawn,
                  context,
                  execRoot,
                  getSandboxOptions().symlinkedSandboxExpandsTreeArtifactsInRunfilesTree),
              outputs,
              writableDirs,
              treeDeleter);

    return runSpawn(spawn, sandbox, context, execRoot, timeout, statisticsPath);
  }

  @Override
  public String getName() {
    return "windows-sandbox";
  }

  @Override
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env)
      throws IOException {
    ImmutableSet.Builder<Path> writableDirs = ImmutableSet.builder();
    writableDirs.addAll(super.getWritableDirs(sandboxExecRoot, env));

    FileSystem fs = sandboxExecRoot.getFileSystem();
    // writableDirs.add(fs.getPath("/dev/shm").resolveSymbolicLinks());
    // writableDirs.add(fs.getPath("/tmp"));

    return writableDirs.build();
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
