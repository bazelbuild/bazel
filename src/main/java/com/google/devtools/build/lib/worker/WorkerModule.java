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

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.common.options.OptionsBase;

import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

/**
 * A module that adds the WorkerActionContextProvider to the available action context providers.
 */
public class WorkerModule extends BlazeModule {
  private final WorkerPool workers;

  private BuildRequest buildRequest;
  private BlazeRuntime blazeRuntime;

  public WorkerModule() {
    GenericKeyedObjectPoolConfig config = new GenericKeyedObjectPoolConfig();
    config.setTimeBetweenEvictionRunsMillis(10 * 1000);
    workers = new WorkerPool(new WorkerFactory(), config);
  }

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.<ActionContextProvider>of(
        new WorkerActionContextProvider(buildRequest, workers, blazeRuntime.getEventBus()));
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return command.builds()
        ? ImmutableList.<Class<? extends OptionsBase>>of(WorkerOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command) {
    this.blazeRuntime = blazeRuntime;
    blazeRuntime.getEventBus().register(this);
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
  }

  @Override
  public void afterCommand() {
    buildRequest = null;
  }
}
