// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/** Abstract common ancestor for sandbox strategies implementing the common parts. */
abstract class SandboxStrategy implements SandboxedSpawnActionContext {

  private final BuildRequest buildRequest;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final SandboxOptions sandboxOptions;
  private final SpawnHelpers spawnHelpers;

  public SandboxStrategy(
      BuildRequest buildRequest,
      BlazeDirectories blazeDirs,
      boolean verboseFailures,
      SandboxOptions sandboxOptions) {
    this.buildRequest = buildRequest;
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.sandboxOptions = sandboxOptions;
    this.spawnHelpers = new SpawnHelpers(blazeDirs.getExecRoot());
  }

  protected void runSpawn(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      Map<String, String> spawnEnvironment,
      SandboxExecRoot sandboxExecRoot,
      Set<PathFragment> outputs,
      SandboxRunner runner,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    EventHandler eventHandler = actionExecutionContext.getExecutor().getEventHandler();
    ExecException execException = null;
    OutErr outErr = actionExecutionContext.getFileOutErr();
    try {
      runner.run(
          spawn.getArguments(),
          spawnEnvironment,
          outErr,
          Spawns.getTimeoutSeconds(spawn),
          SandboxHelpers.shouldAllowNetwork(buildRequest, spawn),
          sandboxOptions.sandboxDebug,
          sandboxOptions.sandboxFakeHostname);
    } catch (ExecException e) {
      execException = e;
    }

    if (writeOutputFiles != null && !writeOutputFiles.compareAndSet(null, SandboxStrategy.class)) {
      throw new InterruptedException();
    }

    try {
      // We copy the outputs even when the command failed, otherwise StandaloneTestStrategy
      // won't be able to get the test logs of a failed test. (We should probably do this in
      // some better way.)
      sandboxExecRoot.copyOutputs(execRoot, outputs);
    } catch (IOException e) {
      if (execException == null) {
        throw new UserExecException("Could not move output artifacts from sandboxed execution", e);
      } else {
        // Catch the IOException and turn it into an error message, otherwise this might hide an
        // exception thrown during runner.run earlier.
        eventHandler.handle(
            Event.error(
                "I/O exception while extracting output artifacts from sandboxed execution: " + e));
      }
    }

    if (execException != null) {
      outErr.printErr(
          "Use --strategy="
          + spawn.getMnemonic()
          + "=standalone to disable sandboxing for the failing actions.\n");
      throw execException;
    }
  }

  /** Gets the list of directories that the spawn will assume to be writable. */
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env) {
    Builder<Path> writableDirs = ImmutableSet.builder();
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    if (env.containsKey("TEST_TMPDIR")) {
      writableDirs.add(sandboxExecRoot.getRelative(env.get("TEST_TMPDIR")));
    }
    return writableDirs.build();
  }

  protected ImmutableSet<Path> getInaccessiblePaths() {
    ImmutableSet.Builder<Path> inaccessiblePaths = ImmutableSet.builder();
    for (String path : sandboxOptions.sandboxBlockPath) {
      inaccessiblePaths.add(blazeDirs.getFileSystem().getPath(path));
    }
    return inaccessiblePaths.build();
  }

  @Override
  public String toString() {
    return "sandboxed";
  }

  @Override
  public boolean shouldPropagateExecException() {
    return verboseFailures && sandboxOptions.sandboxDebug;
  }

  public Map<PathFragment, Path> getMounts(Spawn spawn, ActionExecutionContext executionContext)
      throws ExecException {
    try {
      return spawnHelpers.getMounts(spawn, executionContext);
    } catch (IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }
  }

}
