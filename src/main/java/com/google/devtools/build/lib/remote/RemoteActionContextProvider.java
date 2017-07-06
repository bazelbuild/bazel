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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ActionInputPrefetcher;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;

/**
 * Provide a remote execution context.
 */
final class RemoteActionContextProvider extends ActionContextProvider {
  private final CommandEnvironment env;
  private RemoteSpawnStrategy spawnStrategy;

  RemoteActionContextProvider(CommandEnvironment env) {
    this.env = env;
  }

  @Override
  public void init(
      ActionInputFileCache actionInputFileCache, ActionInputPrefetcher actionInputPrefetcher) {
    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    SpawnActionContext fallbackStrategy =
        new StandaloneSpawnStrategy(
            env.getExecRoot(),
            actionInputPrefetcher,
            localExecutionOptions,
            executionOptions.verboseFailures,
            env.getRuntime().getProductName(),
            ResourceManager.instance());

    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    ChannelOptions channelOptions = ChannelOptions.create(authAndTlsOptions);

    // Initialize remote cache and execution handlers. We use separate handlers for every
    // action to enable server-side parallelism (need a different gRPC channel per action).
    RemoteActionCache remoteCache;
    if (SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)) {
      remoteCache = new SimpleBlobStoreActionCache(SimpleBlobStoreFactory.create(remoteOptions));
    } else if (GrpcRemoteCache.isRemoteCacheOptions(remoteOptions)) {
      remoteCache =
          new GrpcRemoteCache(
              GrpcUtils.createChannel(remoteOptions.remoteCache, channelOptions),
              channelOptions,
              remoteOptions);
    } else {
      remoteCache = null;
    }

    // Otherwise remoteCache remains null and remote caching/execution are disabled.
    GrpcRemoteExecutor remoteExecutor;
    if (remoteCache != null && GrpcRemoteExecutor.isRemoteExecutionOptions(remoteOptions)) {
      remoteExecutor =
          new GrpcRemoteExecutor(
              GrpcUtils.createChannel(remoteOptions.remoteExecutor, channelOptions),
              channelOptions,
              remoteOptions);
    } else {
      remoteExecutor = null;
    }
    spawnStrategy =
        new RemoteSpawnStrategy(
            env.getExecRoot(),
            remoteOptions,
            remoteCache,
            remoteExecutor,
            executionOptions.verboseFailures,
            fallbackStrategy);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    return ImmutableList.of(Preconditions.checkNotNull(spawnStrategy));
  }

  @Override
  public void executionPhaseEnding() {
    if (spawnStrategy != null) {
      spawnStrategy.close();
      spawnStrategy = null;
    }
  }
}
