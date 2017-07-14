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
package com.google.devtools.build.lib.standalone;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.exec.ActionInputPrefetcher;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.SortedMap;

/**
 * Strategy that uses subprocessing to execute a process.
 */
@ExecutionStrategy(name = { "standalone", "local" }, contextType = SpawnActionContext.class)
public class StandaloneSpawnStrategy implements SpawnActionContext {
  private final boolean verboseFailures;
  private final LocalSpawnRunner localSpawnRunner;

  public StandaloneSpawnStrategy(
      Path execRoot, ActionInputPrefetcher actionInputPrefetcher,
      LocalExecutionOptions localExecutionOptions, boolean verboseFailures, String productName,
      ResourceManager resourceManager) {
    this.verboseFailures = verboseFailures;
    LocalEnvProvider localEnvProvider = OS.getCurrent() == OS.DARWIN
        ? new XCodeLocalEnvProvider()
        : LocalEnvProvider.UNMODIFIED;
    this.localSpawnRunner = new LocalSpawnRunner(
        execRoot,
        actionInputPrefetcher,
        localExecutionOptions,
        resourceManager,
        productName,
        localEnvProvider);
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(final Spawn spawn, final ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
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
        // Do nothing for now.
      }

      @Override
      public long getTimeoutMillis() {
        return timeoutSeconds * 1000L;
      }

      @Override
      public FileOutErr getFileOutErr() {
        return actionExecutionContext.getFileOutErr();
      }

      @Override
      public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
        return new SpawnInputExpander(/*strict*/false)
            .getInputMapping(
                spawn,
                actionExecutionContext.getArtifactExpander(),
                actionExecutionContext.getActionInputFileCache(),
                actionExecutionContext.getContext(FilesetActionContext.class));
      }

      @Override
      public void report(ProgressStatus state) {
        switch (state) {
          case EXECUTING:
            String strategyName = "local";
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

    if (actionExecutionContext.reportsSubcommands()) {
      actionExecutionContext.reportSubcommand(spawn);
    }

    try {
      SpawnResult result = localSpawnRunner.exec(spawn, policy);
      if (result.status() != Status.SUCCESS || result.exitCode() != 0) {
        String message =
            CommandFailureUtils.describeCommandFailure(
                verboseFailures, spawn.getArguments(), spawn.getEnvironment(), null);
        throw new SpawnExecException(
            message, result, /*forciblyRunRemotely=*/false, /*catastrophe=*/false);
      }
    } catch (IOException e) {
      throw new UserExecException("I/O exception during local execution", e);
    }
  }

  @Override
  public String toString() {
    return "standalone";
  }
}
