// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ChangedFilesMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.OptionsClassProvider;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A spawn action context that launches Spawns the first time they are used in a persistent mode and
 * then shards work over all the processes.
 */
@ExecutionStrategy(name = { "worker" }, contextType = SpawnActionContext.class)
final class WorkerSpawnStrategy implements SpawnActionContext {
  private final WorkerPool workers;
  private final IncrementalHeuristic incrementalHeuristic;

  public WorkerSpawnStrategy(
      OptionsClassProvider optionsProvider, WorkerPool workers, EventBus eventBus) {
    Preconditions.checkNotNull(optionsProvider);
    WorkerOptions options = optionsProvider.getOptions(WorkerOptions.class);
    workers.setMaxTotalPerKey(options.workerMaxInstances);
    workers.setMaxIdlePerKey(options.workerMaxInstances);
    workers.setMinIdlePerKey(options.workerMaxInstances);
    this.workers = workers;
    this.incrementalHeuristic = new IncrementalHeuristic(options.workerMaxChangedFiles);
    eventBus.register(incrementalHeuristic);
  }

  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    if (!incrementalHeuristic.shouldUseWorkers()) {
      SpawnActionContext context = actionExecutionContext.getExecutor().getSpawnActionContext("");
      if (context != this) {
        context.exec(spawn, actionExecutionContext);
        return;
      }
    }

    String paramFile = Iterables.getLast(spawn.getArguments());

    // We assume that the spawn to be executed always gets a single argument, which is a flagfile
    // prefixed with @ and that it will start in persistent mode when we don't pass it one.
    // Thus, we can extract the last element from its args (which will be the flagfile) to start the
    // persistent mode and then pass it the flagfile via a WorkRequest to make it actually do the
    // work.
    Preconditions.checkArgument(Iterables.getLast(spawn.getArguments()).startsWith("@"));
    ImmutableList<String> args = ImmutableList.<String>builder()
        .addAll(spawn.getArguments().subList(0, spawn.getArguments().size() - 1))
        .add("--persistent_worker")
        .build();
    ImmutableMap<String, String> env = spawn.getEnvironment();
    Path workDir = actionExecutionContext.getExecutor().getExecRoot();
    WorkerKey key = new WorkerKey(args, env, workDir);

    try {
      Worker worker = workers.borrowObject(key);
      try {
        WorkRequest.newBuilder()
            .addArguments(paramFile)
            .build()
            .writeDelimitedTo(worker.getOutputStream());
        worker.getOutputStream().flush();

        WorkResponse response = WorkResponse.parseDelimitedFrom(worker.getInputStream());

        if (response == null) {
          throw new UserExecException("Worker did not return a correct WorkResponse");
        }

        String trimmedOutput = response.getOutput().trim();
        if (!trimmedOutput.isEmpty()) {
          System.err.println(trimmedOutput);
        }

        if (response.getExitCode() != 0) {
          throw new UserExecException(String.format("Failed with exit code: %d.",
              response.getExitCode()));
        }
      } finally {
        if (worker != null) {
          workers.returnObject(key, worker);
        }
      }
    } catch (Exception e) {
      throw new UserExecException(e.getMessage(), e);
    }
  }

  @Override
  public String strategyLocality(String mnemonic, boolean remotable) {
    return "worker";
  }

  @Override
  public boolean isRemotable(String mnemonic, boolean remotable) {
    return false;
  }

  /**
   * For installation with remote execution, non-incremental builds may be slowed down by the
   * persistent worker system. To avoid this we only use workers for builds where few files
   * changed.
   */
  @ThreadSafety.ThreadSafe
  private static class IncrementalHeuristic {
    private final AtomicBoolean fewFilesChanged = new AtomicBoolean(false);
    private int limit = 0;

    public boolean shouldUseWorkers() {
      return limit == 0 || fewFilesChanged.get();
    }

    IncrementalHeuristic(int limit) {
      this.limit = limit;
    }

    @Subscribe
    public void changedFiles(ChangedFilesMessage msg) {
      fewFilesChanged.set(msg.getChangedFiles().size() <= limit);
    }
  }
}
