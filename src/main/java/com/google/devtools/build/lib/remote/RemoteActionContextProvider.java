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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * Provide a remote execution context.
 */
final class RemoteActionContextProvider extends ActionContextProvider {
  private final CommandEnvironment env;
  private final AbstractRemoteActionCache cache;
  @Nullable private final GrpcRemoteExecutor executor;
  private final RemoteRetrier retrier;
  private final DigestUtil digestUtil;
  @Nullable private final Path logDir;
  private final AtomicReference<SpawnRunner> fallbackRunner = new AtomicReference<>();

  private RemoteActionContextProvider(
      CommandEnvironment env,
      AbstractRemoteActionCache cache,
      @Nullable GrpcRemoteExecutor executor,
      RemoteRetrier retrier,
      DigestUtil digestUtil,
      @Nullable Path logDir) {
    this.env = Preconditions.checkNotNull(env, "env");
    this.cache = Preconditions.checkNotNull(cache, "cache");
    this.executor = executor;
    this.retrier = retrier;
    this.digestUtil = digestUtil;
    this.logDir = logDir;
  }

  public static RemoteActionContextProvider createForRemoteCaching(
      CommandEnvironment env,
      AbstractRemoteActionCache cache,
      RemoteRetrier retrier,
      DigestUtil digestUtil) {
    return new RemoteActionContextProvider(
        env, cache, /*executor=*/ null, retrier, digestUtil, /*logDir=*/ null);
  }

  public static RemoteActionContextProvider createForRemoteExecution(
      CommandEnvironment env,
      GrpcRemoteCache cache,
      GrpcRemoteExecutor executor,
      RemoteRetrier retrier,
      DigestUtil digestUtil,
      Path logDir) {
    return new RemoteActionContextProvider(env, cache, executor, retrier, digestUtil, logDir);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    ExecutionOptions executionOptions =
        checkNotNull(env.getOptions().getOptions(ExecutionOptions.class));
    RemoteOptions remoteOptions = checkNotNull(env.getOptions().getOptions(RemoteOptions.class));
    String buildRequestId = env.getBuildRequestId();
    String commandId = env.getCommandId().toString();

    if (executor == null) {
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
              env.getOptions().getOptions(ExecutionOptions.class),
              fallbackRunner,
              executionOptions.verboseFailures,
              env.getReporter(),
              buildRequestId,
              commandId,
              (GrpcRemoteCache) cache,
              executor,
              retrier,
              digestUtil,
              logDir);
      return ImmutableList.of(new RemoteSpawnStrategy(env.getExecRoot(), spawnRunner));
    }
  }

  @Override
  public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {
    SortedSet<String> validStrategies = new TreeSet<>();
    fallbackRunner.set(null);

    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    String strategyName = remoteOptions.remoteLocalFallbackStrategy;

    for (ActionContext context : usedContexts) {
      if (context instanceof RemoteSpawnStrategy && cache == null) {
        throw new ExecutorInitException(
            "--remote_cache or --remote_executor should be initialized when using "
                + "--spawn_strategy=remote",
            ExitCode.COMMAND_LINE_ERROR);
      }
      if (context instanceof AbstractSpawnStrategy) {
        ExecutionStrategy annotation = context.getClass().getAnnotation(ExecutionStrategy.class);
        if (annotation != null) {
          Collections.addAll(validStrategies, annotation.name());
          if (!strategyName.equals("remote")
              && Arrays.asList(annotation.name()).contains(strategyName)) {
            AbstractSpawnStrategy spawnStrategy = (AbstractSpawnStrategy) context;
            SpawnRunner spawnRunner = Preconditions.checkNotNull(spawnStrategy.getSpawnRunner());
            fallbackRunner.set(spawnRunner);
          }
        }
      }
    }

    if (fallbackRunner.get() == null) {
      validStrategies.remove("remote");
      throw new ExecutorInitException(
          String.format(
              "'%s' is an invalid value for --remote_local_fallback_strategy. Valid values are: %s",
              strategyName, validStrategies),
          ExitCode.COMMAND_LINE_ERROR);
    }
  }

  @Override
  public void executionPhaseEnding() {
    if (cache != null) {
      cache.close();
    }
    if (executor != null) {
      executor.close();
    }
  }
}
