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
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/** Strategy that uses sandboxing to execute a process. */
@ExecutionStrategy(
  name = {"sandboxed", "processwrapper-sandbox"},
  contextType = SpawnActionContext.class
)
public class ProcessWrapperSandboxedStrategy extends SandboxStrategy {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    return ProcessWrapperRunner.isSupported(cmdEnv);
  }

  private final SandboxOptions sandboxOptions;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final String productName;
  private final SpawnInputExpander spawnInputExpander;

  ProcessWrapperSandboxedStrategy(
      CommandEnvironment cmdEnv,
      BuildRequest buildRequest,
      Path sandboxBase,
      boolean verboseFailures,
      String productName) {
    super(
        cmdEnv,
        buildRequest,
        sandboxBase,
        verboseFailures,
        buildRequest.getOptions(SandboxOptions.class));
    this.sandboxOptions = buildRequest.getOptions(SandboxOptions.class);
    this.execRoot = cmdEnv.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.productName = productName;
    this.spawnInputExpander = new SpawnInputExpander(false);
  }

  @Override
  protected void actuallyExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException, IOException {
    Executor executor = actionExecutionContext.getExecutor();
    executor
        .getEventBus()
        .post(
            ActionStatusMessage.runningStrategy(
                spawn.getResourceOwner(), "processwrapper-sandbox"));
    SandboxHelpers.reportSubcommand(executor, spawn);

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = getSandboxRoot();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    ImmutableMap<String, String> spawnEnvironment =
        StandaloneSpawnStrategy.locallyDeterminedEnv(execRoot, productName, spawn.getEnvironment());

    Set<Path> writableDirs = getWritableDirs(sandboxExecRoot, spawn.getEnvironment());
    SymlinkedExecRoot symlinkedExecRoot = new SymlinkedExecRoot(sandboxExecRoot);
    ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);
    symlinkedExecRoot.createFileSystem(
        SandboxHelpers.getInputFiles(
            spawnInputExpander, this.execRoot, spawn, actionExecutionContext),
        outputs,
        writableDirs);

    SandboxRunner runner = new ProcessWrapperRunner(sandboxExecRoot, verboseFailures);
    try {
      runSpawn(
          spawn,
          actionExecutionContext,
          spawnEnvironment,
          symlinkedExecRoot,
          outputs,
          runner,
          writeOutputFiles);
    } finally {
      if (!sandboxOptions.sandboxDebug) {
        try {
          FileSystemUtils.deleteTree(sandboxPath);
        } catch (IOException e) {
          // This usually means that the Spawn itself exited, but still has children running that
          // we couldn't wait for, which now block deletion of the sandbox directory. On Linux this
          // should never happen, as we use PID namespaces and where they are not available the
          // subreaper feature to make sure all children have been reliably killed before returning,
          // but on other OS this might not always work. The SandboxModule will try to delete them
          // again when the build is all done, at which point it hopefully works, so let's just go
          // on here.
        }
      }
    }
  }
}
