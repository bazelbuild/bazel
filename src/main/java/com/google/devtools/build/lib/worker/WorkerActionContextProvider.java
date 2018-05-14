// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XcodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.WindowsLocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;

/**
 * Factory for the Worker-based execution strategy.
 */
final class WorkerActionContextProvider extends ActionContextProvider {
  private final ImmutableList<ActionContext> strategies;

  public WorkerActionContextProvider(CommandEnvironment env, WorkerPool workers) {
    ImmutableMultimap<String, String> extraFlags =
        ImmutableMultimap.copyOf(env.getOptions().getOptions(WorkerOptions.class).workerExtraFlags);

    WorkerSpawnRunner spawnRunner =
        new WorkerSpawnRunner(
            env.getExecRoot(),
            workers,
            extraFlags,
            env.getReporter(),
            createFallbackRunner(env));

    WorkerSpawnStrategy workerSpawnStrategy =
        new WorkerSpawnStrategy(env.getExecRoot(), spawnRunner);
    this.strategies = ImmutableList.of(workerSpawnStrategy);
  }

  private static SpawnRunner createFallbackRunner(CommandEnvironment env) {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    LocalEnvProvider localEnvProvider =
        OS.getCurrent() == OS.DARWIN
            ? new XcodeLocalEnvProvider(env.getClientEnv())
            : (OS.getCurrent() == OS.WINDOWS
                ? new WindowsLocalEnvProvider(env.getClientEnv())
                : new PosixLocalEnvProvider(env.getClientEnv()));
    return new LocalSpawnRunner(
        env.getExecRoot(),
        localExecutionOptions,
        ResourceManager.instance(),
        localEnvProvider);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    return strategies;
  }
}
