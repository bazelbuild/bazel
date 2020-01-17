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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * {@link BlazeModule} providing support for dynamic spawn execution and scheduling.
 */
public class DynamicExecutionModule extends BlazeModule {

  /** Strings that can be used to select this strategy in flags. */
  private static final String[] COMMANDLINE_IDENTIFIERS = {"dynamic", "dynamic_worker"};

  private ExecutorService executorService;

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

  private List<Map.Entry<String, List<String>>> getLocalStrategies(
      DynamicExecutionOptions options) {
    // Options that set "allowMultiple" to true ignore the default value, so we replicate that
    // functionality here. Additionally, since we are still supporting --dynamic_worker_strategy,
    // but will deprecate it soon, we add its functionality to --dynamic_local_strategy. This allows
    // users to set --dynamic_local_strategy and not --dynamic_worker_strategy to stop defaulting to
    // worker strategy.
    // TODO(steinman): Deprecate --dynamic_worker_strategy and clean this up.
    if (options.dynamicLocalStrategy == null || options.dynamicLocalStrategy.isEmpty()) {
      String workerStrategy =
          options.dynamicWorkerStrategy.isEmpty() ? "worker" : options.dynamicWorkerStrategy;
      return ImmutableList.of(
          Maps.immutableEntry("", ImmutableList.of(workerStrategy, "sandboxed")));
    }

    ImmutableList.Builder<Map.Entry<String, List<String>>> localAndWorkerStrategies =
        ImmutableList.builder();
    for (Map.Entry<String, List<String>> entry : options.dynamicLocalStrategy) {
      if ("".equals(entry.getKey())) {
        List<String> newValue = Lists.newArrayList(options.dynamicWorkerStrategy);
        newValue.addAll(entry.getValue());
        localAndWorkerStrategies.add(Maps.immutableEntry("", newValue));
      } else {
        localAndWorkerStrategies.add(entry);
      }
    }
    return localAndWorkerStrategies.build();
  }

  private List<Map.Entry<String, List<String>>> getRemoteStrategies(
      DynamicExecutionOptions options) {
    return (options.dynamicRemoteStrategy == null || options.dynamicRemoteStrategy.isEmpty())
        ? ImmutableList.of(Maps.immutableEntry("", ImmutableList.of("remote")))
        : options.dynamicRemoteStrategy;
  }

  @VisibleForTesting
  void initStrategies(ExecutorBuilder builder, DynamicExecutionOptions options)
      throws ExecutorInitException {
    if (!options.internalSpawnScheduler) {
      return;
    }

    if (options.legacySpawnScheduler) {
      builder.addActionContext(
          SpawnStrategy.class,
          new LegacyDynamicSpawnStrategy(executorService, options, this::getExecutionPolicy),
          COMMANDLINE_IDENTIFIERS);
    } else {
      builder.addActionContext(
          SpawnStrategy.class,
          new DynamicSpawnStrategy(executorService, options, this::getExecutionPolicy),
          COMMANDLINE_IDENTIFIERS);
    }
    builder.addStrategyByContext(SpawnStrategy.class, "dynamic");

    for (Map.Entry<String, List<String>> mnemonicToStrategies : getLocalStrategies(options)) {
      throwIfContainsDynamic(mnemonicToStrategies.getValue(), "--dynamic_local_strategy");
      builder.addDynamicLocalStrategiesByMnemonic(
          mnemonicToStrategies.getKey(), mnemonicToStrategies.getValue());
    }
    for (Map.Entry<String, List<String>> mnemonicToStrategies : getRemoteStrategies(options)) {
      throwIfContainsDynamic(mnemonicToStrategies.getValue(), "--dynamic_remote_strategy");
      builder.addDynamicRemoteStrategiesByMnemonic(
          mnemonicToStrategies.getKey(), mnemonicToStrategies.getValue());
    }
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder)
      throws ExecutorInitException {
    initStrategies(builder, env.getOptions().getOptions(DynamicExecutionOptions.class));
  }

  private void throwIfContainsDynamic(List<String> strategies, String flagName)
      throws ExecutorInitException {
    ImmutableSet<String> identifiers = ImmutableSet.copyOf(COMMANDLINE_IDENTIFIERS);
    if (!Sets.intersection(identifiers, ImmutableSet.copyOf(strategies)).isEmpty()) {
      throw new ExecutorInitException(
          "Cannot use strategy "
              + identifiers
              + " in flag "
              + flagName
              + " as it would create a cycle during execution");
    }
  }

  /**
   * Use the {@link Spawn} metadata to determine if it can be executed locally, remotely, or both.
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

  @Override
  public void afterCommand() {
    ExecutorUtil.interruptibleShutdown(executorService);
    executorService = null;
  }
}
