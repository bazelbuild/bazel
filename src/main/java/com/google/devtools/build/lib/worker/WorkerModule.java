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
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;

import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

import java.io.IOException;

/**
 * A module that adds the WorkerActionContextProvider to the available action context providers.
 */
public class WorkerModule extends BlazeModule {
  private WorkerPool workers;

  private CommandEnvironment env;
  private BuildRequest buildRequest;
  private boolean verbose;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(WorkerOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);

    if (workers == null) {
      Path logDir = env.getOutputBase().getRelative("worker-logs");
      try {
        logDir.createDirectory();
      } catch (IOException e) {
        env
            .getReporter()
            .handle(Event.error("Could not create directory for worker logs: " + logDir));
      }

      GenericKeyedObjectPoolConfig config = new GenericKeyedObjectPoolConfig();

      // It's better to re-use a worker as often as possible and keep it hot, in order to profit
      // from JIT optimizations as much as possible.
      config.setLifo(true);

      // Check for & deal with idle workers every 5 seconds.
      config.setTimeBetweenEvictionRunsMillis(5 * 1000);

      // Always test the liveliness of worker processes.
      config.setTestOnBorrow(true);
      config.setTestOnCreate(true);
      config.setTestOnReturn(true);
      config.setTestWhileIdle(true);

      // Don't limit the total number of worker processes, as otherwise the pool might be full of
      // e.g. Java workers and could never accommodate another request for a different kind of
      // worker.
      config.setMaxTotal(-1);

      workers = new WorkerPool(new WorkerFactory(), config);
      workers.setReporter(env.getReporter());
      workers.setLogDirectory(logDir);
    }
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    Preconditions.checkNotNull(workers);

    this.buildRequest = event.getRequest();

    WorkerOptions options = buildRequest.getOptions(WorkerOptions.class);
    workers.setMaxTotalPerKey(options.workerMaxInstances);
    workers.setMaxIdlePerKey(options.workerMaxInstances);
    workers.setMinIdlePerKey(options.workerMaxInstances);
    workers.setVerbose(options.workerVerbose);
    this.verbose = options.workerVerbose;
  }

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    Preconditions.checkNotNull(env);
    Preconditions.checkNotNull(buildRequest);
    Preconditions.checkNotNull(workers);

    return ImmutableList.<ActionContextProvider>of(
        new WorkerActionContextProvider(env, buildRequest, workers));
  }

  @Override
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    return ImmutableList.<ActionContextConsumer>of(new WorkerActionContextConsumer());
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    if (workers != null && buildRequest.getOptions(WorkerOptions.class).workerQuitAfterBuild) {
      if (verbose) {
        env
            .getReporter()
            .handle(Event.info("Build completed, shutting down worker pool..."));
      }
      workers.close();
      workers = null;
    }
  }

  // Kill workers on Ctrl-C to quickly end the interrupted build.
  // TODO(philwo) - make sure that this actually *kills* the workers and not just politely waits
  // for them to finish.
  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    if (workers != null) {
      if (verbose) {
        env
            .getReporter()
            .handle(Event.info("Build interrupted, shutting down worker pool..."));
      }
      workers.close();
      workers = null;
    }
  }

  @Override
  public void afterCommand() {
    this.env = null;
    this.buildRequest = null;
    this.verbose = false;
  }
}
