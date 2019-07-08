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
  private final TreeDeleter treeDeleter;

  /**
   * Creates a sandboxed spawn runner that uses the {@code windows-sandbox} tool.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   */
  WindowsSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Duration timeoutKillDelay,
      PathFragment windowsSandboxPath,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.fileSystem = cmdEnv.getRuntime().getFileSystem();
    this.blazeDirs = cmdEnv.getDirectories();
    this.execRoot = cmdEnv.getExecRoot();
    this.windowsSandbox = windowsSandboxPath;
    this.sandboxBase = sandboxBase;
    this.timeoutKillDelay = timeoutKillDelay;
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
    sandboxPath.createDirectoryAndParents();

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());
    sandboxExecRoot.createDirectoryAndParents();

    Path tmpDir = createActionTemp(execRoot);
    Path commandTmpDir = tmpDir.getRelative("work");
    commandTmpDir.createDirectory();
    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, commandTmpDir.getPathString());

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
          new CopyingSandboxedSpawn(
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

  private Path createActionTemp(Path execRoot) throws IOException {
    return execRoot.getRelative(
        java.nio.file.Files.createTempDirectory(
                java.nio.file.Paths.get(execRoot.getPathString()), "windows-sandbox.")
            .getFileName()
            .toString());
  }

  @Override
  public String getName() {
    return "windows-sandbox";
  }
}
