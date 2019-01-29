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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsBase;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * {@link BlazeModule} providing support for dynamic spawn execution and scheduling.
 */
public class DynamicExecutionModule extends BlazeModule {
  private ExecutorService executorService;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(DynamicExecutionOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    executorService =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("dynamic-execution-thread-%d").build());
    env.getEventBus().register(this);
  }

  /**
   * Adds a strategy that backs the dynamic scheduler to the executor builder.
   *
   * @param builder the executor builder to modify
   * @param name the name of the strategy
   * @param flagName name of the flag the strategy came from; used for error reporting
   *     purposes only
   * @throws ExecutorInitException if the provided strategy would cause a scheduling cycle
   */
  private static void addBackingStrategy(ExecutorBuilder builder, String name, String flagName)
      throws ExecutorInitException {
    ExecutionStrategy strategy = DynamicSpawnStrategy.class.getAnnotation(ExecutionStrategy.class);
    checkNotNull(strategy, "DynamicSpawnStrategy lacks expected ExecutionStrategy annotation");

    if (Arrays.asList(strategy.name()).contains(name)) {
      throw new ExecutorInitException("Cannot use strategy " + name + " in flag " + flagName
          + " as it would create a cycle during execution");
    }

    builder.addStrategyByContext(SpawnActionContext.class, name);
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder)
      throws ExecutorInitException {
    DynamicExecutionOptions options = env.getOptions().getOptions(DynamicExecutionOptions.class);
    if (options.internalSpawnScheduler) {
      builder.addActionContext(
          new DynamicSpawnStrategy(executorService, options, this::getExecutionPolicy));
      builder.addStrategyByContext(SpawnActionContext.class, "dynamic");
      addBackingStrategy(builder, options.dynamicLocalStrategy, "--dynamic_local_strategy");
      addBackingStrategy(builder, options.dynamicRemoteStrategy, "--dynamic_remote_strategy");
      addBackingStrategy(builder, options.dynamicWorkerStrategy, "--dynamic_worker_strategy");
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

    return ExecutionPolicy.ANYWHERE;
  }

  @Override
  public void afterCommand() {
    ExecutorUtil.interruptibleShutdown(executorService);
    executorService = null;
  }
}
