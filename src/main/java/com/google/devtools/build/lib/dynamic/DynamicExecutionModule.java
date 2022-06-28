// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.dynamic;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.ExecutionOptions;
import com.google.devtools.build.lib.server.FailureDetails.ExecutionOptions.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.common.options.OptionsBase;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** {@link BlazeModule} providing support for dynamic spawn execution and scheduling. */
public class DynamicExecutionModule extends BlazeModule {

  private ExecutorService executorService;

  /**
   * If true, this is the first build since this server started (excluding failed builds). This
   * allows turning off dynamic execution for an initial build, avoiding dynamic execution on most
   * clean builds.
   */
  private static boolean firstBuild = true;

  public DynamicExecutionModule() {}

  @VisibleForTesting
  DynamicExecutionModule(ExecutorService executorService) {
    this.executorService = executorService;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(DynamicExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    executorService =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("dynamic-execution-thread-%d").build());
    env.getEventBus().register(this);
  }

  @VisibleForTesting
  ImmutableMap<String, List<String>> getLocalStrategies(DynamicExecutionOptions options)
      throws AbruptExitException {
    // Options that set "allowMultiple" to true ignore the default value, so we replicate that
    // functionality here.
    // ImmutableMap.Builder fails on duplicates, so we use a regular map first to remove dups.
    Map<String, List<String>> localAndWorkerStrategies = new HashMap<>();
    localAndWorkerStrategies.put("", ImmutableList.of("worker", "sandboxed"));

    if (!options.dynamicLocalStrategy.isEmpty()) {
      for (Map.Entry<String, List<String>> entry : options.dynamicLocalStrategy) {
        localAndWorkerStrategies.put(entry.getKey(), entry.getValue());
        throwIfContainsDynamic(entry.getValue(), "--dynamic_local_strategy");
      }
    }
    return ImmutableMap.copyOf(localAndWorkerStrategies);
  }

  private ImmutableMap<String, List<String>> getRemoteStrategies(DynamicExecutionOptions options)
      throws AbruptExitException {
    Map<String, List<String>> strategies = new HashMap<>(); // Needed to dedup
    for (Map.Entry<String, List<String>> e : options.dynamicRemoteStrategy) {
      throwIfContainsDynamic(e.getValue(), "--dynamic_remote_strategy");
      strategies.put(e.getKey(), e.getValue());
    }
    return options.dynamicRemoteStrategy.isEmpty()
        ? ImmutableMap.of("", ImmutableList.of("remote"))
        : ImmutableMap.copyOf(strategies);
  }

  @Override
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env)
      throws AbruptExitException {
    DynamicExecutionOptions options = env.getOptions().getOptions(DynamicExecutionOptions.class);
    com.google.devtools.build.lib.exec.ExecutionOptions execOptions =
        env.getOptions().getOptions(com.google.devtools.build.lib.exec.ExecutionOptions.class);
    registerSpawnStrategies(
        registryBuilder,
        options,
        (int) execOptions.localCpuResources,
        env.getOptions().getOptions(BuildRequestOptions.class).jobs);
  }

  // CommandEnvironment is difficult to access in tests, so use this method for testing.
  @VisibleForTesting
  final void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder,
      DynamicExecutionOptions options,
      int numCpus,
      int jobs)
      throws AbruptExitException {
    if (!options.internalSpawnScheduler) {
      return;
    }

    SpawnStrategy strategy =
        new DynamicSpawnStrategy(
            executorService,
            options,
            this::getExecutionPolicy,
            this::getPostProcessingSpawnForLocalExecution,
            firstBuild,
            numCpus,
            jobs,
            this::canIgnoreFailure);
    registryBuilder.registerStrategy(strategy, "dynamic", "dynamic_worker");
    registryBuilder.addDynamicLocalStrategies(getLocalStrategies(options));
    registryBuilder.addDynamicRemoteStrategies(getRemoteStrategies(options));
  }

  private void throwIfContainsDynamic(List<String> strategies, String flagName)
      throws AbruptExitException {
    ImmutableSet<String> identifiers = ImmutableSet.of("dynamic", "dynamic_worker");
    if (!Sets.intersection(identifiers, ImmutableSet.copyOf(strategies)).isEmpty()) {
      String message =
          String.format(
              "Cannot use strategy %s in flag %s as it would create a cycle during" + " execution",
              identifiers, flagName);
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(message)
                  .setExecutionOptions(
                      ExecutionOptions.newBuilder().setCode(Code.INVALID_CYCLIC_DYNAMIC_STRATEGY))
                  .build()));
    }
  }

  /**
   * Use the {@link Spawn} metadata to determine if it can be executed locally, remotely, or both.
   *
   * @param spawn the {@link Spawn} action
   * @return the {@link ExecutionPolicy} containing local/remote execution policies
   */
  protected ExecutionPolicy getExecutionPolicy(Spawn spawn) {
    if (!Spawns.mayBeExecutedRemotely(spawn)) {
      return ExecutionPolicy.LOCAL_EXECUTION_ONLY;
    }
    if (!Spawns.mayBeExecutedLocally(spawn)) {
      return ExecutionPolicy.REMOTE_EXECUTION_ONLY;
    }

    return ExecutionPolicy.ANYWHERE;
  }

  /**
   * Returns a post processing {@link Spawn} if one needs to be executed after given {@link Spawn}
   * when running locally.
   *
   * <p>The intention of this is to allow post-processing of the original {@linkplain Spawn spawn}
   * when executing it locally. In particular, such spawn should never create outputs which are not
   * included in the generating action of the original one.
   */
  protected Optional<Spawn> getPostProcessingSpawnForLocalExecution(Spawn spawn) {
    return Optional.empty();
  }

  /**
   * If true, the failure passed in can be ignored in one branch to allow the other branch to finish
   * it instead. This can e.g. allow ignoring remote execution timeouts or local-only permission
   * failures.
   *
   * @param spawn The spawn being executed.
   * @param exitCode The exit code from executing the spawn
   * @param errorMessage Error messages returned from executing the spawn
   * @param outErr The location of the stdout and stderr from the spawn.
   * @param isLocal True if this is the locally-executed branch.
   * @return True if this failure is one that we want to allow the other branch to succeed at, even
   *     though this branch failed already.
   */
  protected boolean canIgnoreFailure(
      Spawn spawn,
      ActionExecutionContext context,
      int exitCode,
      String errorMessage,
      FileOutErr outErr,
      boolean isLocal) {
    return false;
  }

  @FunctionalInterface
  interface IgnoreFailureCheck {
    boolean canIgnoreFailure(
        Spawn spawn,
        ActionExecutionContext context,
        int exitCode,
        String errorMessage,
        FileOutErr outErr,
        boolean isLocal);
  }

  @Subscribe
  public void buildCompleteEvent(BuildCompleteEvent event) {
    BuildResult result = event.getResult();
    if (result.getSuccess()) {
      firstBuild = false;
    }
  }

  @Override
  public void afterCommand() {
    ExecutorUtil.interruptibleShutdown(executorService);
    executorService = null;
  }
}
