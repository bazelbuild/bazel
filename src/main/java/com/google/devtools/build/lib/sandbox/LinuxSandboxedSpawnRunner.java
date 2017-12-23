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
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.LinuxSandboxUtil;
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
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.SortedMap;

/** Spawn runner that uses linux sandboxing APIs to execute a local subprocess. */
final class LinuxSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    if (OS.getCurrent() != OS.LINUX) {
      return false;
    }
    if (!LinuxSandboxUtil.isSupported(cmdEnv)) {
      return false;
    }

    List<String> linuxSandboxArgv =
        LinuxSandboxUtil.commandLineBuilder(
                LinuxSandboxUtil.getLinuxSandbox(cmdEnv).getPathString(),
                ImmutableList.of("/bin/true"))
            .build();
    ImmutableMap<String, String> env = ImmutableMap.of();
    Path execRoot = cmdEnv.getExecRoot();
    File cwd = execRoot.getPathFile();

    Command cmd = new Command(linuxSandboxArgv.toArray(new String[0]), env, cwd);
    try {
      cmd.execute(ByteStreams.nullOutputStream(), ByteStreams.nullOutputStream());
    } catch (CommandException e) {
      return false;
    }

    return true;
  }

  private final FileSystem fileSystem;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean allowNetwork;
  private final Path linuxSandbox;
  private final Path inaccessibleHelperFile;
  private final Path inaccessibleHelperDir;
  private final LocalEnvProvider localEnvProvider;
  private final Optional<Duration> timeoutKillDelay;
  private final String productName;

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool. If a spawn exceeds
   * its timeout, then it will be killed instantly.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param productName the product name to use
   * @param inaccessibleHelperFile path to a file that is (already) inaccessible
   * @param inaccessibleHelperDir path to a directory that is (already) inaccessible
   */
  LinuxSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir) {
    this(
        cmdEnv,
        sandboxBase,
        productName,
        inaccessibleHelperFile,
        inaccessibleHelperDir,
        Optional.empty());
  }

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool. If a spawn exceeds
   * its timeout, then it will be killed after the specified delay.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param productName the product name to use
   * @param inaccessibleHelperFile path to a file that is (already) inaccessible
   * @param inaccessibleHelperDir path to a directory that is (already) inaccessible
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   */
  LinuxSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir,
      Duration timeoutKillDelay) {
    this(
        cmdEnv,
        sandboxBase,
        productName,
        inaccessibleHelperFile,
        inaccessibleHelperDir,
        Optional.of(timeoutKillDelay));
  }

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param productName the product name to use
   * @param inaccessibleHelperFile path to a file that is (already) inaccessible
   * @param inaccessibleHelperDir path to a directory that is (already) inaccessible
   * @param timeoutKillDelay an optional, additional grace period before killing timing out
   *     commands. If not present, then no grace period is used and commands are killed instantly.
   */
  LinuxSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir,
      Optional<Duration> timeoutKillDelay) {
    super(cmdEnv, sandboxBase);
    this.fileSystem = cmdEnv.getRuntime().getFileSystem();
    this.blazeDirs = cmdEnv.getDirectories();
    this.execRoot = cmdEnv.getExecRoot();
    this.productName = productName;
    this.allowNetwork = SandboxHelpers.shouldAllowNetwork(cmdEnv.getOptions());
    this.linuxSandbox = LinuxSandboxUtil.getLinuxSandbox(cmdEnv);
    this.inaccessibleHelperFile = inaccessibleHelperFile;
    this.inaccessibleHelperDir = inaccessibleHelperDir;
    this.timeoutKillDelay = timeoutKillDelay;
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

    LinuxSandboxUtil.CommandLineBuilder commandLineBuilder =
        LinuxSandboxUtil.commandLineBuilder(linuxSandbox.getPathString(), spawn.getArguments())
            .setWritableFilesAndDirectories(writableDirs)
            .setTmpfsDirectories(getTmpfsPaths())
            .setBindMounts(getReadOnlyBindMounts(blazeDirs, sandboxExecRoot))
            .setUseFakeHostname(getSandboxOptions().sandboxFakeHostname)
            .setCreateNetworkNamespace(!(allowNetwork || Spawns.requiresNetwork(spawn)))
            .setUseDebugMode(getSandboxOptions().sandboxDebug);

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }
    if (timeoutKillDelay.isPresent()) {
      commandLineBuilder.setKillDelay(timeoutKillDelay.get());
    }
    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_FAKEROOT)) {
      commandLineBuilder.setUseFakeRoot(true);
    } else if (getSandboxOptions().sandboxFakeUsername) {
      commandLineBuilder.setUseFakeUsername(true);
    }

    Optional<String> statisticsPath = Optional.empty();
    if (getSandboxOptions().collectLocalSandboxExecutionStatistics) {
      statisticsPath = Optional.of(sandboxPath.getRelative("stats.out").getPathString());
      commandLineBuilder.setStatisticsPath(statisticsPath.get());
    }

    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, tmpDir, productName);

    SandboxedSpawn sandbox =
        new SymlinkedSandboxedSpawn(
            sandboxPath,
            sandboxExecRoot,
            commandLineBuilder.build(),
            environment,
            SandboxHelpers.getInputFiles(spawn, policy, execRoot),
            outputs,
            writableDirs);

    return runSpawn(spawn, sandbox, policy, execRoot, tmpDir, timeout, statisticsPath);
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
