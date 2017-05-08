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
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicReference;

/** Abstract common ancestor for sandbox strategies implementing the common parts. */
abstract class SandboxStrategy implements SandboxedSpawnActionContext {

  private final BuildRequest buildRequest;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final SandboxOptions sandboxOptions;
  private final SpawnInputExpander spawnInputExpander;
  private final Path sandboxBase;

  public SandboxStrategy(
      BuildRequest buildRequest,
      BlazeDirectories blazeDirs,
      Path sandboxBase,
      boolean verboseFailures,
      SandboxOptions sandboxOptions) {
    this.buildRequest = buildRequest;
    this.execRoot = blazeDirs.getExecRoot();
    this.sandboxBase = sandboxBase;
    this.verboseFailures = verboseFailures;
    this.sandboxOptions = sandboxOptions;
    this.spawnInputExpander = new SpawnInputExpander(/*strict=*/false);
  }

  /** Executes the given {@code spawn}. */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    exec(spawn, actionExecutionContext, null);
  }

  @Override
  public void exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    // Certain actions can't run remotely or in a sandbox - pass them on to the standalone strategy.
    if (!spawn.isRemotable() || spawn.hasNoSandbox()) {
      SandboxHelpers.fallbackToNonSandboxedExecution(spawn, actionExecutionContext, executor);
      return;
    }

    EventBus eventBus = actionExecutionContext.getExecutor().getEventBus();
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    eventBus.post(ActionStatusMessage.schedulingStrategy(owner));
    try (ResourceHandle ignored =
        ResourceManager.instance().acquireResources(owner, spawn.getLocalResources())) {
      actuallyExec(spawn, actionExecutionContext, writeOutputFiles);
    } catch (IOException e) {
      throw new UserExecException("I/O exception during sandboxed execution", e);
    }
  }

  protected abstract void actuallyExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException, IOException;

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
          sandboxOptions.sandboxFakeHostname,
          sandboxOptions.sandboxFakeUsername);
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

  /**
   * Returns a temporary directory that should be used as the sandbox directory for a single action.
   */
  protected Path getSandboxRoot() throws IOException {
    return sandboxBase.getRelative(
        java.nio.file.Files.createTempDirectory(
                java.nio.file.Paths.get(sandboxBase.getPathString()), "")
            .getFileName()
            .toString());
  }

  /**
   * Gets the list of directories that the spawn will assume to be writable.
   *
   * @throws IOException because we might resolve symlinks, which throws {@link IOException}.
   */
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env)
      throws IOException {
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    if (env.containsKey("TEST_TMPDIR")) {
      return ImmutableSet.of(sandboxExecRoot.getRelative(env.get("TEST_TMPDIR")));
    }
    return ImmutableSet.of();
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
      Map<PathFragment, ActionInput> inputMap = spawnInputExpander
          .getInputMapping(
              spawn,
              executionContext.getArtifactExpander(),
              executionContext.getActionInputFileCache(),
              executionContext.getExecutor().getContext(FilesetActionContext.class));
      // SpawnInputExpander#getInputMapping uses ArtifactExpander#expandArtifacts to expand
      // middlemen and tree artifacts, which expands empty tree artifacts to no entry. However,
      // actions that accept TreeArtifacts as inputs generally expect that the empty directory is
      // created. So we add those explicitly here.
      // TODO(ulfjack): Move this code to SpawnInputExpander.
      for (ActionInput input : spawn.getInputFiles()) {
        if (input instanceof Artifact && ((Artifact) input).isTreeArtifact()) {
          List<Artifact> containedArtifacts = new ArrayList<>();
          executionContext.getArtifactExpander().expand((Artifact) input, containedArtifacts);
          // Attempting to mount a non-empty directory results in ERR_DIRECTORY_NOT_EMPTY, so we
          // only mount empty TreeArtifacts as directories.
          if (containedArtifacts.isEmpty()) {
            inputMap.put(input.getExecPath(), input);
          }
        }
      }

      Map<PathFragment, Path> mounts = new TreeMap<>();
      for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
        Path inputPath = e.getValue() == SpawnInputExpander.EMPTY_FILE
            ? null
            : execRoot.getRelative(e.getValue().getExecPath());
        mounts.put(e.getKey(), inputPath);
      }
      return mounts;
    } catch (IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }
  }

}
