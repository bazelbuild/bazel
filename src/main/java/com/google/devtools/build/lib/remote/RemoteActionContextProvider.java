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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import javax.annotation.Nullable;

/**
 * Provide a remote execution context.
 */
final class RemoteActionContextProvider extends ActionContextProvider {
  private final CommandEnvironment env;
  private final RemoteActionCache cache;
  private final GrpcRemoteExecutor executor;

  private RemoteSpawnRunner spawnRunner;
  private RemoteSpawnStrategy spawnStrategy;

  RemoteActionContextProvider(CommandEnvironment env, @Nullable RemoteActionCache cache,
      @Nullable GrpcRemoteExecutor executor) {
    this.env = env;
    this.executor = executor;
    this.cache = cache;
  }

  @Override
  public void init(
      ActionInputFileCache actionInputFileCache, ActionInputPrefetcher actionInputPrefetcher) {
    ExecutionOptions executionOptions =
        checkNotNull(env.getOptions().getOptions(ExecutionOptions.class));
    RemoteOptions remoteOptions = checkNotNull(env.getOptions().getOptions(RemoteOptions.class));

    spawnRunner = new RemoteSpawnRunner(
        env.getExecRoot(),
        remoteOptions,
        createFallbackRunner(),
        cache,
        executor);
    spawnStrategy =
        new RemoteSpawnStrategy(
            spawnRunner,
            executionOptions.verboseFailures);
  }

  private SpawnRunner createFallbackRunner() {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    LocalEnvProvider localEnvProvider = OS.getCurrent() == OS.DARWIN
        ? new XCodeLocalEnvProvider()
        : LocalEnvProvider.UNMODIFIED;
    return
        new LocalSpawnRunner(
            env.getExecRoot(),
            localExecutionOptions,
            ResourceManager.instance(),
            env.getRuntime().getProductName(),
            localEnvProvider);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    return ImmutableList.of(checkNotNull(spawnStrategy));
  }

  @Override
  public void executionPhaseEnding() {
    if (spawnRunner != null) {
      spawnRunner.close();
    }
    spawnRunner = null;
    spawnStrategy = null;
  }
}
