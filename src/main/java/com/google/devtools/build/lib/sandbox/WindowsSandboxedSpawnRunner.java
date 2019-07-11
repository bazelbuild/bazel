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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.WindowsLocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;

/** Spawn runner that uses BuildXL Sandbox APIs to execute a local subprocess. */
final class WindowsSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  private final Path execRoot;
  private final PathFragment windowsSandbox;
  private final LocalEnvProvider localEnvProvider;
  private final Duration timeoutKillDelay;

  /**
   * Creates a sandboxed spawn runner that uses the {@code windows-sandbox} tool.
   *
   * @param cmdEnv the command environment to use
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   * @param windowsSandboxPath path to windows-sandbox binary
   */
  WindowsSandboxedSpawnRunner(
      CommandEnvironment cmdEnv, Duration timeoutKillDelay, PathFragment windowsSandboxPath) {
    super(cmdEnv);
    this.execRoot = cmdEnv.getExecRoot();
    this.windowsSandbox = windowsSandboxPath;
    this.timeoutKillDelay = timeoutKillDelay;
    this.localEnvProvider = new WindowsLocalEnvProvider(cmdEnv.getClientEnv());
  }

  @Override
  protected SpawnResult actuallyExec(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ExecException, InterruptedException {
    Path tmpDir = createActionTemp(execRoot);
    Path commandTmpDir = tmpDir.getRelative("work");
    commandTmpDir.createDirectory();
    ImmutableMap<String, String> environment =
        ImmutableMap.copyOf(
            localEnvProvider.rewriteLocalEnv(
                spawn.getEnvironment(), binTools, commandTmpDir.getPathString()));

    Map<PathFragment, Path> readablePaths =
        SandboxHelpers.processInputFiles(
            spawn,
            context,
            execRoot,
            getSandboxOptions().symlinkedSandboxExpandsTreeArtifactsInRunfilesTree);

    ImmutableSet.Builder<Path> writablePaths = ImmutableSet.builder();
    writablePaths.addAll(getWritableDirs(execRoot, environment));
    for (ActionInput output : spawn.getOutputFiles()) {
      writablePaths.add(execRoot.getRelative(output.getExecPath()));
    }

    Duration timeout = context.getTimeout();

    WindowsSandboxUtil.CommandLineBuilder commandLineBuilder =
        WindowsSandboxUtil.commandLineBuilder(windowsSandbox, spawn.getArguments())
            .setWritableFilesAndDirectories(writablePaths.build())
            .setReadableFilesAndDirectories(readablePaths)
            .setInaccessiblePaths(getInaccessiblePaths())
            .setUseDebugMode(getSandboxOptions().sandboxDebug)
            .setKillDelay(timeoutKillDelay);

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }

    Path statisticsPath = null;

    SandboxedSpawn sandbox =
        new WindowsSandboxedSpawn(execRoot, environment, commandLineBuilder.build());

    return runSpawn(spawn, sandbox, context, execRoot, timeout, statisticsPath);
  }

  private static Path createActionTemp(Path execRoot) throws IOException {
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
