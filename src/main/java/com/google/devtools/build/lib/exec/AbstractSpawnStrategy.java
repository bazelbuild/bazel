// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Abstract common ancestor for spawn strategies implementing the common parts. */
public abstract class AbstractSpawnStrategy implements SandboxedSpawnActionContext {
  private final boolean verboseFailures;
  private final SpawnInputExpander spawnInputExpander;
  private final SpawnRunner spawnRunner;
  private final AtomicInteger execCount = new AtomicInteger();

  public AbstractSpawnStrategy(
      boolean verboseFailures,
      SpawnRunner spawnRunner) {
    this.verboseFailures = verboseFailures;
    this.spawnInputExpander = new SpawnInputExpander(false);
    this.spawnRunner = spawnRunner;
  }

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
    if (actionExecutionContext.reportsSubcommands()) {
      actionExecutionContext.reportSubcommand(spawn);
    }
    final Duration timeout = Spawns.getTimeout(spawn);
    SpawnExecutionPolicy policy =
        new SpawnExecutionPolicyImpl(
            spawn, actionExecutionContext, writeOutputFiles, timeout);
    SpawnResult result;
    try {
      result = spawnRunner.exec(spawn, policy);
    } catch (IOException e) {
      throw new EnvironmentalExecException("Unexpected IO error.", e);
    }

    if ((result.status() != Status.SUCCESS) || (result.exitCode() != 0)) {
      String cwd = actionExecutionContext.getExecRoot().getPathString();
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, spawn.getArguments(), spawn.getEnvironment(), cwd);
      throw new SpawnExecException(
          message, result, /*forciblyRunRemotely=*/false, /*catastrophe=*/false);
    }
  }

  private final class SpawnExecutionPolicyImpl implements SpawnExecutionPolicy {
    private final Spawn spawn;
    private final ActionExecutionContext actionExecutionContext;
    private final AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles;
    private final Duration timeout;

    private final int id = execCount.incrementAndGet();
    // Memoize the input mapping so that prefetchInputs can reuse it instead of recomputing it.
    // TODO(ulfjack): Guard against client modification of this map.
    private SortedMap<PathFragment, ActionInput> lazyInputMapping;

    public SpawnExecutionPolicyImpl(
        Spawn spawn,
        ActionExecutionContext actionExecutionContext,
        AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles,
        Duration timeout) {
      this.spawn = spawn;
      this.actionExecutionContext = actionExecutionContext;
      this.writeOutputFiles = writeOutputFiles;
      this.timeout = timeout;
    }

    @Override
    public int getId() {
      return id;
    }

    @Override
    public void prefetchInputs() throws IOException {
      if (Spawns.shouldPrefetchInputsForLocalExecution(spawn)) {
        // TODO(philwo): Benchmark whether using an ExecutionService to do multiple operations in
        // parallel speeds up prefetching of inputs.
        // TODO(philwo): Do we have to expand middleman artifacts here?
        actionExecutionContext.getActionInputPrefetcher().prefetchFiles(
            Iterables.filter(getInputMapping().values(), Predicates.notNull()));
      }
    }

    @Override
    public ActionInputFileCache getActionInputFileCache() {
      return actionExecutionContext.getActionInputFileCache();
    }

    @Override
    public ArtifactExpander getArtifactExpander() {
      return actionExecutionContext.getArtifactExpander();
    }

    @Override
    public void lockOutputFiles() throws InterruptedException {
      Class<? extends SpawnActionContext> token = AbstractSpawnStrategy.this.getClass();
      if (writeOutputFiles != null
          && writeOutputFiles.get() != token
          && !writeOutputFiles.compareAndSet(null, token)) {
        throw new InterruptedException();
      }
    }

    @Override
    public Duration getTimeout() {
      return timeout;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return actionExecutionContext.getFileOutErr();
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
      if (lazyInputMapping == null) {
        lazyInputMapping = spawnInputExpander.getInputMapping(
            spawn,
            actionExecutionContext.getArtifactExpander(),
            actionExecutionContext.getActionInputFileCache(),
            actionExecutionContext.getContext(FilesetActionContext.class));
      }
      return lazyInputMapping;
    }

    @Override
    public void report(ProgressStatus state, String name) {
      // TODO(ulfjack): We should report more details to the UI.
      EventBus eventBus = actionExecutionContext.getEventBus();
      switch (state) {
        case EXECUTING:
          eventBus.post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), name));
          break;
        case SCHEDULING:
          eventBus.post(ActionStatusMessage.schedulingStrategy(spawn.getResourceOwner()));
          break;
        default:
          break;
      }
    }
  }
}
