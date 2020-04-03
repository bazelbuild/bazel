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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.CleanCommand.CleanStartingEvent;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxOptions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.worker.WorkerOptions.MultiResourceConverter;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/** A module that adds the WorkerActionContextProvider to the available action context providers. */
public class WorkerModule extends BlazeModule {
  private CommandEnvironment env;

  private WorkerFactory workerFactory;
  private WorkerPool workerPool;
  private WorkerOptions options;
  private ImmutableMap<String, Integer> workerPoolConfig;
  
  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(WorkerOptions.class)
        : ImmutableList.of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Subscribe
  public void cleanStarting(CleanStartingEvent event) {
    if (workerPool != null) {
      this.options = event.getOptionsProvider().getOptions(WorkerOptions.class);
      workerFactory.setReporter(env.getReporter());
      workerFactory.setOptions(options);
      shutdownPool("Clean command is running, shutting down worker pool...");
    }
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    options = event.getRequest().getOptions(WorkerOptions.class);

    if (workerFactory == null) {
      Path workerDir =
          env.getOutputBase().getRelative(env.getRuntime().getProductName() + "-workers");
      try {
        if (!workerDir.createDirectory()) {
          // Clean out old log files.
          for (Path logFile : workerDir.getDirectoryEntries()) {
            if (logFile.getBaseName().endsWith(".log")) {
              try {
                logFile.delete();
              } catch (IOException e) {
                env.getReporter()
                    .handle(Event.error("Could not delete old worker log: " + logFile));
              }
            }
          }
        }
      } catch (IOException e) {
        env.getReporter()
            .handle(Event.error("Could not create base directory for workers: " + workerDir));
      }

      workerFactory = new WorkerFactory(options, workerDir);
    }

    workerFactory.setReporter(env.getReporter());
    workerFactory.setOptions(options);

    // Use a LinkedHashMap instead of an ImmutableMap.Builder to allow duplicates; the last value
    // passed wins.
    LinkedHashMap<String, Integer> newConfigBuilder = new LinkedHashMap<>();
    for (Map.Entry<String, Integer> entry : options.workerMaxInstances) {
      newConfigBuilder.put(entry.getKey(), entry.getValue());
    }

    if (!newConfigBuilder.containsKey("")) {
      // Empty string gives the number of workers for any type of worker not explicitly specified.
      // If no value is given, use the default, 4.
      // TODO(steinman): Calculate a reasonable default value instead of arbitrarily defaulting to
      // 4.
      newConfigBuilder.put("", MultiResourceConverter.DEFAULT_VALUE);
    }

    ImmutableMap<String, Integer> newConfig = ImmutableMap.copyOf(newConfigBuilder);

    // If the config changed compared to the last run, we have to create a new pool.
    if (workerPoolConfig != null && !workerPoolConfig.equals(newConfig)) {
      shutdownPool(
          "Worker configuration has changed, restarting worker pool...",
          /* alwaysLog= */ true);
    }

    if (workerPool == null) {
      workerPoolConfig = newConfig;
      workerPool = new WorkerPool(workerFactory, workerPoolConfig, options.highPriorityWorkers);
    }
  }

  @Override
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env) {
    Preconditions.checkNotNull(workerPool);
    SandboxOptions sandboxOptions = env.getOptions().getOptions(SandboxOptions.class);
    ImmutableMultimap<String, String> extraFlags =
        ImmutableMultimap.copyOf(env.getOptions().getOptions(WorkerOptions.class).workerExtraFlags);
    LocalEnvProvider localEnvProvider = LocalEnvProvider.forCurrentOs(env.getClientEnv());
    WorkerSpawnRunner spawnRunner =
        new WorkerSpawnRunner(
            new SandboxHelpers(sandboxOptions.delayVirtualInputMaterialization),
            env.getExecRoot(),
            workerPool,
            extraFlags,
            env.getReporter(),
            createFallbackRunner(env, localEnvProvider),
            localEnvProvider,
            sandboxOptions.symlinkedSandboxExpandsTreeArtifactsInRunfilesTree,
            env.getBlazeWorkspace().getBinTools(),
            env.getLocalResourceManager(),
            // TODO(buchgr): Replace singleton by a command-scoped RunfilesTreeUpdater
            RunfilesTreeUpdater.INSTANCE);
    registryBuilder.registerStrategy(
        new WorkerSpawnStrategy(env.getExecRoot(), spawnRunner), "worker");
  }

  private static SpawnRunner createFallbackRunner(
      CommandEnvironment env, LocalEnvProvider localEnvProvider) {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    return new LocalSpawnRunner(
        env.getExecRoot(),
        localExecutionOptions,
        env.getLocalResourceManager(),
        localEnvProvider,
        env.getBlazeWorkspace().getBinTools(),
        // TODO(buchgr): Replace singleton by a command-scoped RunfilesTreeUpdater
        RunfilesTreeUpdater.INSTANCE);
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    if (options != null && options.workerQuitAfterBuild) {
      shutdownPool("Build completed, shutting down worker pool...");
    }
  }

  // Kill workers on Ctrl-C to quickly end the interrupted build.
  // TODO(philwo) - make sure that this actually *kills* the workers and not just politely waits
  // for them to finish.
  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    shutdownPool("Build interrupted, shutting down worker pool...");
  }

  /** Shuts down the worker pool and sets {#code workerPool} to null. */
  private void shutdownPool(String reason) {
    shutdownPool(reason, /* alwaysLog= */ false);
  }

  /** Shuts down the worker pool and sets {#code workerPool} to null. */
  private void shutdownPool(String reason, boolean alwaysLog) {
    Preconditions.checkArgument(!reason.isEmpty());

    if (workerPool != null) {
      if ((options != null && options.workerVerbose) || alwaysLog) {
        env.getReporter().handle(Event.info(reason));
      }
      workerPool.close();
      workerPool = null;
    }
  }

  @Override
  public void afterCommand() {
    this.env = null;
    this.options = null;

    if (this.workerFactory != null) {
      this.workerFactory.setReporter(null);
    }
  }
}
