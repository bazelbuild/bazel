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
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/**
 * This module provides the Sandbox spawn strategy.
 */
public final class SandboxModule extends BlazeModule {
  // Per-command state
  private CommandEnvironment env;
  private BuildRequest buildRequest;

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    Preconditions.checkNotNull(env);
    Preconditions.checkNotNull(buildRequest);
    try {
      return ImmutableList.<ActionContextProvider>of(
          SandboxActionContextProvider.create(env, buildRequest));
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
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    env = null;
    buildRequest = null;
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
  }
}
