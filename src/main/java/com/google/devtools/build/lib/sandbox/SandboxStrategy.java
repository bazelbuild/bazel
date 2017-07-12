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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.SortedMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** Abstract common ancestor for sandbox strategies implementing the common parts. */
abstract class SandboxStrategy implements SandboxedSpawnActionContext {
  private final boolean verboseFailures;
  private final SandboxOptions sandboxOptions;
  private final SpawnInputExpander spawnInputExpander;
  private final AbstractSandboxSpawnRunner spawnRunner;

  public SandboxStrategy(
      boolean verboseFailures,
      SandboxOptions sandboxOptions,
      AbstractSandboxSpawnRunner spawnRunner) {
    this.verboseFailures = verboseFailures;
    this.sandboxOptions = sandboxOptions;
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
    // Certain actions can't run remotely or in a sandbox - pass them on to the standalone strategy.
    if (!spawn.isRemotable() || spawn.hasNoSandbox()) {
      SandboxHelpers.fallbackToNonSandboxedExecution(spawn, actionExecutionContext);
      return;
    }

    if (actionExecutionContext.reportsSubcommands()) {
      actionExecutionContext.reportSubcommand(spawn);
    }
    final int timeoutSeconds = Spawns.getTimeoutSeconds(spawn);
    final EventBus eventBus = actionExecutionContext.getEventBus();
    SpawnExecutionPolicy policy = new SpawnExecutionPolicy() {
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
        Class<? extends SpawnActionContext> token = SandboxStrategy.this.getClass();
        if (writeOutputFiles != null
            && writeOutputFiles.get() != token
            && !writeOutputFiles.compareAndSet(null, token)) {
          throw new InterruptedException();
        }
      }

      @Override
      public long getTimeoutMillis() {
        return TimeUnit.SECONDS.toMillis(timeoutSeconds);
      }

      @Override
      public FileOutErr getFileOutErr() {
        return actionExecutionContext.getFileOutErr();
      }

      @Override
      public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
        return spawnInputExpander.getInputMapping(
            spawn,
            actionExecutionContext.getArtifactExpander(),
            actionExecutionContext.getActionInputFileCache(),
            actionExecutionContext.getContext(FilesetActionContext.class));
      }

      @Override
      public void report(ProgressStatus state) {
        // TODO(ulfjack): We should report more details to the UI.
        switch (state) {
          case EXECUTING:
            String strategyName = spawnRunner.getName();
            eventBus.post(
                ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), strategyName));
            break;
          case SCHEDULING:
            eventBus.post(ActionStatusMessage.schedulingStrategy(spawn.getResourceOwner()));
            break;
          default:
            break;
        }
      }
    };
    SpawnResult result = spawnRunner.exec(spawn, policy);
    if (result.status() != Status.SUCCESS || result.exitCode() != 0) {
      String message =
          CommandFailureUtils.describeCommandFailure(
              true, spawn.getArguments(), spawn.getEnvironment(), null);
      throw new SpawnExecException(
          message, result, /*forciblyRunRemotely=*/false, /*catastrophe=*/false);
    }
  }

  @Override
  public String toString() {
    return "sandboxed";
  }

  @Override
  public boolean shouldPropagateExecException() {
    return verboseFailures && sandboxOptions.sandboxDebug;
  }
}
