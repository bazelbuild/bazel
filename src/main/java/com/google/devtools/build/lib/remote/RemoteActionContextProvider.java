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
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XcodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.WindowsLocalEnvProvider;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * Provide a remote execution context.
 */
final class RemoteActionContextProvider extends ActionContextProvider {
  private final CommandEnvironment env;
  private final AbstractRemoteActionCache cache;
  private final GrpcRemoteExecutor executor;
  private final DigestUtil digestUtil;
  private final Path logDir;

  RemoteActionContextProvider(
      CommandEnvironment env,
      @Nullable AbstractRemoteActionCache cache,
      @Nullable GrpcRemoteExecutor executor,
      DigestUtil digestUtil,
      Path logDir) {
    this.env = env;
    this.executor = executor;
    this.cache = cache;
    this.digestUtil = digestUtil;
    this.logDir = logDir;
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    ExecutionOptions executionOptions =
        checkNotNull(env.getOptions().getOptions(ExecutionOptions.class));
    RemoteOptions remoteOptions = checkNotNull(env.getOptions().getOptions(RemoteOptions.class));
    String buildRequestId = env.getBuildRequestId().toString();
    String commandId = env.getCommandId().toString();

    if (remoteOptions.experimentalRemoteSpawnCache || remoteOptions.diskCache != null) {
      RemoteSpawnCache spawnCache =
          new RemoteSpawnCache(
              env.getExecRoot(),
              remoteOptions,
              cache,
              buildRequestId,
              commandId,
              env.getReporter(),
              digestUtil);
      return ImmutableList.of(spawnCache);
    } else {
      RemoteSpawnRunner spawnRunner =
          new RemoteSpawnRunner(
              env.getExecRoot(),
              remoteOptions,
              createFallbackRunner(env),
              executionOptions.verboseFailures,
              env.getReporter(),
              buildRequestId,
              commandId,
              cache,
              executor,
              digestUtil,
              logDir);
      return ImmutableList.of(new RemoteSpawnStrategy(env.getExecRoot(), spawnRunner));
    }
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
    return
        new LocalSpawnRunner(
            env.getExecRoot(),
            localExecutionOptions,
            ResourceManager.instance(),
            localEnvProvider);
  }

  @Override
  public void executionPhaseEnding() {
    if (cache != null) {
      cache.close();
    }
  }
}
