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
  private ActionInputPrefetcher actionInputPrefetcher;

  RemoteActionContextProvider(CommandEnvironment env) {
    this.env = env;
  }

  @Override
  public void init(
      ActionInputFileCache actionInputFileCache, ActionInputPrefetcher actionInputPrefetcher) {
    this.actionInputPrefetcher = Preconditions.checkNotNull(actionInputPrefetcher);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
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
    return ImmutableList.of(
        new RemoteSpawnStrategy(
            env.getExecRoot(),
            env.getOptions().getOptions(RemoteOptions.class),
            env.getOptions().getOptions(AuthAndTLSOptions.class),
            executionOptions.verboseFailures,
            fallbackStrategy));
  }
}
