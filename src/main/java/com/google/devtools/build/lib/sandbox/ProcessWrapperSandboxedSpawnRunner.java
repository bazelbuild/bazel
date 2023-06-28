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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.time.Duration;
import javax.annotation.Nullable;

/** Strategy that uses sandboxing to execute a process. */
final class ProcessWrapperSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    return OS.isPosixCompatible() && ProcessWrapper.fromCommandEnvironment(cmdEnv) != null;
  }

  private final SandboxHelpers helpers;
  private final ProcessWrapper processWrapper;
  private final Path execRoot;
  private final ImmutableList<Root> packageRoots;
  private final Path sandboxBase;
  private final LocalEnvProvider localEnvProvider;
  @Nullable private final SandboxfsProcess sandboxfsProcess;
  private final boolean sandboxfsMapSymlinkTargets;
  private final TreeDeleter treeDeleter;

  /**
   * Creates a sandboxed spawn runner that uses the {@code process-wrapper} tool.
   *
   * @param helpers common tools and state across all spawns during sandboxed execution
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param sandboxfsProcess instance of the sandboxfs process to use; may be null for none, in
   *     which case the runner uses a symlinked sandbox
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   */
  ProcessWrapperSandboxedSpawnRunner(
      SandboxHelpers helpers,
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      @Nullable SandboxfsProcess sandboxfsProcess,
      boolean sandboxfsMapSymlinkTargets,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.helpers = helpers;
    this.processWrapper = ProcessWrapper.fromCommandEnvironment(cmdEnv);
    this.execRoot = cmdEnv.getExecRoot();
    this.packageRoots = cmdEnv.getPackageLocator().getPathEntries();
    this.localEnvProvider = LocalEnvProvider.forCurrentOs(cmdEnv.getClientEnv());
    this.sandboxBase = sandboxBase;
    this.sandboxfsProcess = sandboxfsProcess;
    this.sandboxfsMapSymlinkTargets = sandboxfsMapSymlinkTargets;
    this.treeDeleter = treeDeleter;
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException, InterruptedException {
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
    ProcessWrapper.CommandLineBuilder commandLineBuilder =
        processWrapper
            .commandLineBuilder(spawn.getArguments())
            .addExecutionInfo(spawn.getExecutionInfo())
            .setTimeout(timeout);

    Path statisticsPath = sandboxPath.getRelative("stats.out");
    commandLineBuilder.setStatisticsPath(statisticsPath);

    SandboxInputs inputs =
        helpers.processInputFiles(
            context.getInputMapping(PathFragment.EMPTY_FRAGMENT, /* willAccessRepeatedly= */ true),
            execRoot,
            execRoot,
            packageRoots,
            null);
    SandboxOutputs outputs = helpers.getOutputs(spawn);

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
          spawn.getMnemonic(),
          /* sandboxDebugPath= */ null,
          statisticsPath);
    } else {
      return new SymlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          getWritableDirs(sandboxExecRoot, sandboxExecRoot, environment),
          treeDeleter,
          /* sandboxDebugPath= */ null,
          statisticsPath,
          spawn.getMnemonic());
    }
  }

  @Override
  public String getName() {
    return "processwrapper-sandbox";
  }
}
