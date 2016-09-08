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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/** Strategy that uses sandboxing to execute a process. */
@ExecutionStrategy(
  name = {"sandboxed"},
  contextType = SpawnActionContext.class
)
public class LinuxSandboxedStrategy extends SandboxStrategy {
  private static Boolean sandboxingSupported = null;

  public static boolean isSupported(CommandEnvironment env) {
    if (sandboxingSupported == null) {
      sandboxingSupported =
          ProcessWrapperRunner.isSupported(env) || LinuxSandboxRunner.isSupported(env);
    }
    return sandboxingSupported.booleanValue();
  }

  private final BuildRequest buildRequest;
  private final SandboxOptions sandboxOptions;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final ExecutorService backgroundWorkers;
  private final boolean verboseFailures;
  private final String productName;
  private final boolean fullySupported;

  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();

  LinuxSandboxedStrategy(
      BuildRequest buildRequest,
      BlazeDirectories blazeDirs,
      ExecutorService backgroundWorkers,
      boolean verboseFailures,
      String productName,
      boolean fullySupported) {
    super(blazeDirs, verboseFailures, buildRequest.getOptions(SandboxOptions.class));
    this.buildRequest = buildRequest;
    this.sandboxOptions = buildRequest.getOptions(SandboxOptions.class);
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.backgroundWorkers = Preconditions.checkNotNull(backgroundWorkers);
    this.verboseFailures = verboseFailures;
    this.productName = productName;
    this.fullySupported = fullySupported;
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException {
    Executor executor = actionExecutionContext.getExecutor();

    // Certain actions can't run remotely or in a sandbox - pass them on to the standalone strategy.
    if (!spawn.isRemotable()) {
      SandboxHelpers.fallbackToNonSandboxedExecution(spawn, actionExecutionContext, executor);
      return;
    }

    SandboxHelpers.reportSubcommand(executor, spawn);
    SandboxHelpers.postActionStatusMessage(executor, spawn);

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = SandboxHelpers.getSandboxRoot(blazeDirs, productName, uuid, execCounter);
    Path sandboxExecRoot = sandboxPath.getRelative("execroot");

    Set<Path> writableDirs = getWritableDirs(sandboxExecRoot, spawn.getEnvironment());

    try {
      // Build the execRoot for the sandbox.
      SymlinkedExecRoot symlinkedExecRoot = new SymlinkedExecRoot(sandboxExecRoot);
      ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);
      symlinkedExecRoot.createFileSystem(
          getMounts(spawn, actionExecutionContext), outputs, writableDirs);

      final SandboxRunner runner;
      if (fullySupported) {
        runner =
            new LinuxSandboxRunner(
                execRoot,
                sandboxPath,
                sandboxExecRoot,
                getWritableDirs(sandboxExecRoot, spawn.getEnvironment()),
                getInaccessiblePaths(),
                verboseFailures,
                sandboxOptions.sandboxDebug);
      } else {
        runner = new ProcessWrapperRunner(execRoot, sandboxPath, sandboxExecRoot, verboseFailures);
      }
      try {
        runner.run(
            spawn.getArguments(),
            spawn.getEnvironment(),
            actionExecutionContext.getFileOutErr(),
            Spawns.getTimeoutSeconds(spawn),
            SandboxHelpers.shouldAllowNetwork(buildRequest, spawn));
      } finally {
        symlinkedExecRoot.copyOutputs(execRoot, outputs);
        if (!sandboxOptions.sandboxDebug) {
          SandboxHelpers.lazyCleanup(backgroundWorkers, runner);
        }
      }
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    }
  }

}
