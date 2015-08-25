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
package com.google.devtools.build.lib.sandbox;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * This module provides the Sandbox spawn strategy.
 */
public class SandboxModule extends BlazeModule {
  private final ExecutorService backgroundWorkers = Executors.newCachedThreadPool();
  private BuildRequest buildRequest;
  private BlazeRuntime runtime;

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.<ActionContextProvider>of(
        new SandboxActionContextProvider(runtime, buildRequest, backgroundWorkers));
  }

  @Override
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    return ImmutableList.<ActionContextConsumer>of(new SandboxActionContextConsumer());
  }

  @Override
  public void beforeCommand(BlazeRuntime runtime, Command command) {
    if (this.runtime == null) {
      this.runtime = runtime;
    } else {
      // The BlazeRuntime is guaranteed to never change.
      Preconditions.checkArgument(runtime == this.runtime);
    }
    runtime.getEventBus().register(this);
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
  }

  /**
   * Shut down the background worker pool in the canonical way.
   *
   * <p>See https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ExecutorService.html
   */
  @Override
  public void blazeShutdown() {
    // Disable new tasks from being submitted
    backgroundWorkers.shutdown();

    try {
      // Wait a while for existing tasks to terminate
      if (!backgroundWorkers.awaitTermination(5, TimeUnit.SECONDS)) {
        backgroundWorkers.shutdownNow(); // Cancel currently executing tasks

        // Wait a while for tasks to respond to being cancelled and force-kill them if necessary
        // after the timeout.
        backgroundWorkers.awaitTermination(5, TimeUnit.SECONDS);
      }
    } catch (InterruptedException ie) {
      // (Re-)Cancel if current thread also interrupted
      backgroundWorkers.shutdownNow();

      // Preserve interrupt status
      Thread.currentThread().interrupt();
    }
  }
}
