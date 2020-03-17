// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ProcessWrapperUtil;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import javax.annotation.Nullable;

/** Strategy that uses sandboxing to execute a process. */
final class ProcessWrapperSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    return OS.isPosixCompatible() && ProcessWrapperUtil.isSupported(cmdEnv);
  }

  private final Path processWrapper;
  private final Path execRoot;
  private final Path sandboxBase;
  private final LocalEnvProvider localEnvProvider;
  private final Duration timeoutKillDelay;
  @Nullable private final SandboxfsProcess sandboxfsProcess;
  private final boolean sandboxfsMapSymlinkTargets;
  private final TreeDeleter treeDeleter;

  /**
   * Creates a sandboxed spawn runner that uses the {@code process-wrapper} tool.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param timeoutKillDelay additional grace period before killing timing out commands
   * @param sandboxfsProcess instance of the sandboxfs process to use; may be null for none, in
   *     which case the runner uses a symlinked sandbox
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   */
  ProcessWrapperSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Duration timeoutKillDelay,
      @Nullable SandboxfsProcess sandboxfsProcess,
      boolean sandboxfsMapSymlinkTargets,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.processWrapper = ProcessWrapperUtil.getProcessWrapper(cmdEnv);
    this.execRoot = cmdEnv.getExecRoot();
    this.localEnvProvider = LocalEnvProvider.forCurrentOs(cmdEnv.getClientEnv());
    this.sandboxBase = sandboxBase;
    this.timeoutKillDelay = timeoutKillDelay;
    this.sandboxfsProcess = sandboxfsProcess;
    this.sandboxfsMapSymlinkTargets = sandboxfsMapSymlinkTargets;
    this.treeDeleter = treeDeleter;
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ExecException {
    // Each invocation of "exec" gets its own sandbox base.
    // Note that the value returned by context.getId() is only unique inside one given SpawnRunner,
    // so we have to prefix our name to turn it into a globally unique value.
    Path sandboxPath =
        sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));
    sandboxPath.getParentDirectory().createDirectory();
    sandboxPath.createDirectory();

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    String workspaceName = execRoot.getBaseName();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(workspaceName);
    sandboxExecRoot.getParentDirectory().createDirectory();
    sandboxExecRoot.createDirectory();

    ImmutableMap<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    Duration timeout = context.getTimeout();
    ProcessWrapperUtil.CommandLineBuilder commandLineBuilder =
        ProcessWrapperUtil.commandLineBuilder(processWrapper.getPathString(), spawn.getArguments())
            .setTimeout(timeout);

    commandLineBuilder.setKillDelay(timeoutKillDelay);

    Path statisticsPath = null;
    if (getSandboxOptions().collectLocalSandboxExecutionStatistics) {
      statisticsPath = sandboxPath.getRelative("stats.out");
      commandLineBuilder.setStatisticsPath(statisticsPath);
    }

    SandboxInputs inputs =
        SandboxHelpers.processInputFiles(
            context.getInputMapping(
                getSandboxOptions().symlinkedSandboxExpandsTreeArtifactsInRunfilesTree),
            spawn,
            context.getArtifactExpander(),
            execRoot);
    SandboxOutputs outputs = SandboxHelpers.getOutputs(spawn);

    if (sandboxfsProcess != null) {
      return new SandboxfsSandboxedSpawn(
          sandboxfsProcess,
          sandboxPath,
          workspaceName,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          ImmutableSet.of(),
          sandboxfsMapSymlinkTargets,
          treeDeleter,
          statisticsPath);
    } else {
      return new SymlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          getWritableDirs(sandboxExecRoot, environment),
          treeDeleter,
          statisticsPath);
    }
  }

  @Override
  public String getName() {
    return "processwrapper-sandbox";
  }
}
