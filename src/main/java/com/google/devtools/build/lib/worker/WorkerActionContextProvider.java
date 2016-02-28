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
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;

/**
 * Factory for the Worker-based execution strategy.
 */
final class WorkerActionContextProvider extends ActionContextProvider {
  private final ImmutableList<ActionContext> strategies;

  public WorkerActionContextProvider(
      BlazeRuntime runtime, BuildRequest buildRequest, WorkerPool workers, EventBus eventBus) {
    boolean verboseFailures = buildRequest.getOptions(ExecutionOptions.class).verboseFailures;
    int maxRetries = buildRequest.getOptions(WorkerOptions.class).workerMaxRetries;

    this.strategies =
        ImmutableList.<ActionContext>of(
            new WorkerSpawnStrategy(
                runtime.getDirectories(),
                buildRequest,
                workers,
                verboseFailures,
                maxRetries));
  }

  @Override
  public Iterable<ActionContext> getActionContexts() {
    return strategies;
  }
}
