// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Throwables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ActionInputPrefetcher;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.SortedMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Strategy that uses a distributed cache for sharing action input and output files. Optionally this
 * strategy also support offloading the work to a remote worker.
 */
@ExecutionStrategy(
  name = {"remote"},
  contextType = SpawnActionContext.class
)
final class RemoteSpawnStrategy implements SpawnActionContext {
  private final SpawnInputExpander spawnInputExpander = new SpawnInputExpander(/*strict=*/false);
  private final SpawnRunner spawnRunner;
  private final boolean verboseFailures;
  private final ActionInputPrefetcher inputPrefetcher;
  private final AtomicInteger execCount = new AtomicInteger();

  RemoteSpawnStrategy(SpawnRunner spawnRunner, boolean verboseFailures) {
    this.spawnRunner = spawnRunner;
    this.verboseFailures = verboseFailures;
    this.inputPrefetcher = ActionInputPrefetcher.NONE;
  }

  @Override
  public String toString() {
    return "remote";
  }

  @Override
  public void exec(final Spawn spawn, final ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    if (!spawn.isRemotable()) {
      StandaloneSpawnStrategy standaloneStrategy =
          Preconditions.checkNotNull(
              actionExecutionContext.getContext(StandaloneSpawnStrategy.class));
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    if (actionExecutionContext.reportsSubcommands()) {
      actionExecutionContext.reportSubcommand(spawn);
    }
    final int timeoutSeconds = Spawns.getTimeoutSeconds(spawn);
    SpawnExecutionPolicy policy = new SpawnExecutionPolicy() {
      private final int id = execCount.incrementAndGet();

      @Override
      public int getId() {
        return id;
      }

      @Override
      public void prefetchInputs(Iterable<ActionInput> inputs) throws IOException {
        inputPrefetcher.prefetchFiles(inputs);
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
        // This is only needed for the dynamic spawn strategy, which we still need to actually
        // implement.
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
      public void report(ProgressStatus state, String name) {
        EventBus eventBus = actionExecutionContext.getEventBus();
        switch (state) {
          case EXECUTING:
            eventBus.post(
                ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), name));
            break;
          case SCHEDULING:
            eventBus.post(ActionStatusMessage.schedulingStrategy(spawn.getResourceOwner()));
            break;
          default:
            break;
        }
      }
    };

    SpawnResult result;
    try {
      result = spawnRunner.exec(spawn, policy);
    } catch (IOException e) {
      if (verboseFailures) {
        actionExecutionContext
            .getEventHandler()
            .handle(
                Event.warn(
                    spawn.getMnemonic()
                        + " remote work failed:\n"
                        + Throwables.getStackTraceAsString(e)));
      }
      throw new EnvironmentalExecException("Unexpected IO error.", e);
    }

    if ((result.status() != Status.SUCCESS) || (result.exitCode() != 0)) {
      // TODO(ulfjack): Return SpawnResult from here and let the upper layers worry about error
      // handling and reporting.
      String cwd = actionExecutionContext.getExecRoot().getPathString();
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, spawn.getArguments(), spawn.getEnvironment(), cwd);
      throw new SpawnExecException(message, result, /*catastrophe=*/ false);
    }
  }
}
