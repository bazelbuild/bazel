// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsBase;

import java.net.URI;
import java.net.URISyntaxException;

/**
 * RemoteModule provides distributed cache and remote execution for Bazel.
 */
public final class RemoteModule extends BlazeModule {
  private CommandEnvironment env;
  private BuildRequest buildRequest;
  private RemoteActionCache actionCache;
  private RemoteWorkExecutor workExecutor;

  public RemoteModule() {}

  @Override
  public Iterable<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.<ActionContextProvider>of(
        new RemoteActionContextProvider(env, buildRequest, actionCache, workExecutor));
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    this.env = null;
    this.buildRequest = null;
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    buildRequest = event.getRequest();
    RemoteOptions options = buildRequest.getOptions(RemoteOptions.class);

    // Don't provide the remote spawn unless at least action cache is initialized.
    if (actionCache == null
        && (options.hazelcastNode != null || options.hazelcastClientConfig != null)) {
      MemcacheActionCache cache =
          new MemcacheActionCache(
              this.env.getDirectories().getExecRoot(),
              options,
              HazelcastCacheFactory.create(options));
      actionCache = cache;
      if (workExecutor == null && options.remoteWorker != null) {
        try {
          URI uri = new URI("dummy://" + options.remoteWorker);
          if (uri.getHost() == null || uri.getPort() == -1) {
            throw new URISyntaxException("Invalid host or port.", "");
          }
          workExecutor =
              MemcacheWorkExecutor.createRemoteWorkExecutor(cache, uri.getHost(), uri.getPort());
        } catch (URISyntaxException e) {
          env.getReporter()
              .handle(Event.warn("Invalid argument for the address of remote worker."));
        }
      }
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(RemoteOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }
}
