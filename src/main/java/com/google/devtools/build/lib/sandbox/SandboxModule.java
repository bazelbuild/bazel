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
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsBase;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This module provides the Sandbox spawn strategy.
 */
public class SandboxModule extends BlazeModule {
  public static final String SANDBOX_NOT_SUPPORTED_MESSAGE =
      "Sandboxed execution is not supported on your system and thus hermeticity of actions cannot "
          + "be guaranteed. See http://bazel.io/docs/bazel-user-manual.html#sandboxing for more "
          + "information. You can turn off this warning via --ignore_unsupported_sandboxing";

  // Per-server state
  private ExecutorService backgroundWorkers;
  private Boolean sandboxingSupported = null;

  // Per-command state
  private CommandEnvironment env;
  private BuildRequest buildRequest;

  private synchronized boolean isSandboxingSupported(CommandEnvironment env) {
    if (sandboxingSupported == null) {
      sandboxingSupported = NamespaceSandboxRunner.isSupported(env);
    }
    return sandboxingSupported.booleanValue();
  }

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    Preconditions.checkNotNull(buildRequest);
    Preconditions.checkNotNull(env);
    if (isSandboxingSupported(env)) {
      return ImmutableList.<ActionContextProvider>of(
          new SandboxActionContextProvider(env, buildRequest, backgroundWorkers));
    }

    // For now, sandboxing is only supported on Linux and there's not much point in showing a scary
    // warning to the user if they can't do anything about it.
    if (!buildRequest.getOptions(SandboxOptions.class).ignoreUnsupportedSandboxing
        && OS.getCurrent() == OS.LINUX) {
      env.getReporter().handle(Event.warn(SANDBOX_NOT_SUPPORTED_MESSAGE));
    }

    return ImmutableList.of();
  }

  @Override
  public Iterable<ActionContextConsumer> getActionContextConsumers() {
    Preconditions.checkNotNull(env);
    if (isSandboxingSupported(env)) {
      return ImmutableList.<ActionContextConsumer>of(new SandboxActionContextConsumer());
    }
    return ImmutableList.of();
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(SandboxOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    backgroundWorkers = Executors.newCachedThreadPool();
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    env = null;
    buildRequest = null;

    // "bazel clean" will also try to delete the sandbox directories, leading to a race condition
    // if it is run right after a "bazel build". We wait for and shutdown the background worker pool
    // before continuing to avoid this.
    ExecutorUtil.interruptibleShutdown(backgroundWorkers);
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
  }
}
