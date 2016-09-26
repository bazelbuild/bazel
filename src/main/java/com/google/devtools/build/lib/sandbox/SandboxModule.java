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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This module provides the Sandbox spawn strategy.
 */
public final class SandboxModule extends BlazeModule {
  // Per-command state
  private CommandEnvironment env;
  private BuildRequest buildRequest;
  private ExecutorService backgroundWorkers;
  private SandboxOptions sandboxOptions;

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    Preconditions.checkNotNull(env);
    Preconditions.checkNotNull(buildRequest);
    Preconditions.checkNotNull(backgroundWorkers);
    sandboxOptions = buildRequest.getOptions(SandboxOptions.class);
    try {
      return ImmutableList.<ActionContextProvider>of(
          SandboxActionContextProvider.create(env, buildRequest, backgroundWorkers));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    Preconditions.checkNotNull(env);
    return ImmutableList.<ActionContextConsumer>of(new SandboxActionContextConsumer(env));
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(SandboxOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    backgroundWorkers =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("sandbox-background-worker-%d").build());
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    // We want to make sure that all sandbox directories are deleted after a command finishes or at
    // least the user gets notified if some of them can't be deleted. However we can't rely on the
    // background workers for that, because a) they can't log, and b) if a directory is undeletable,
    // the Runnable might never finish. So we cancel them and delete the remaining directories here,
    // where we have more control.
    backgroundWorkers.shutdownNow();
    if (sandboxOptions != null && !sandboxOptions.sandboxDebug) {
      Path sandboxRoot =
          env.getDirectories()
              .getOutputBase()
              .getRelative(env.getRuntime().getProductName() + "-sandbox");
      if (sandboxRoot.exists()) {
        try {
          for (Path child : sandboxRoot.getDirectoryEntries()) {
            try {
              FileSystemUtils.deleteTree(child);
            } catch (IOException e) {
              env.getReporter()
                  .handle(
                      Event.warn(
                          String.format(
                              "Could not delete sandbox directory: %s (%s)",
                              child.getPathString(), e)));
            }
          }
          sandboxRoot.delete();
        } catch (IOException e) {
          env.getReporter()
              .handle(
                  Event.warn(
                      String.format(
                          "Could not delete %s directory: %s", sandboxRoot.getBaseName(), e)));
        }
      }
    }

    env = null;
    buildRequest = null;
    backgroundWorkers = null;
    sandboxOptions = null;
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
  }
}
