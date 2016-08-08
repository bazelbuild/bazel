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
import com.google.common.collect.ImmutableList.Builder;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.util.concurrent.ExecutorService;

/**
 * Provides the sandboxed spawn strategy.
 */
public class SandboxActionContextProvider extends ActionContextProvider {

  @SuppressWarnings("unchecked")
  private final ImmutableList<ActionContext> strategies;

  private SandboxActionContextProvider(ImmutableList<ActionContext> strategies) {
    this.strategies = strategies;
  }

  public static SandboxActionContextProvider create(
      CommandEnvironment env, BuildRequest buildRequest, ExecutorService backgroundWorkers)
      throws IOException {
    boolean verboseFailures = buildRequest.getOptions(ExecutionOptions.class).verboseFailures;
    boolean unblockNetwork =
        buildRequest
            .getOptions(BuildConfiguration.Options.class)
            .testArguments
            .contains("--wrapper_script_flag=--debug");
    Builder<ActionContext> strategies = ImmutableList.builder();

    if (OS.getCurrent() == OS.LINUX) {
      strategies.add(
          new LinuxSandboxedStrategy(
              buildRequest.getOptions(SandboxOptions.class),
              env.getClientEnv(),
              env.getDirectories(),
              backgroundWorkers,
              verboseFailures,
              unblockNetwork,
              env.getRuntime().getProductName()));
    } else if (OS.getCurrent() == OS.DARWIN) {
      strategies.add(
          DarwinSandboxedStrategy.create(
              buildRequest.getOptions(SandboxOptions.class),
              env.getClientEnv(),
              env.getDirectories(),
              backgroundWorkers,
              verboseFailures,
              unblockNetwork,
              env.getRuntime().getProductName()));
    }

    return new SandboxActionContextProvider(strategies.build());
  }

  @Override
  public Iterable<ActionContext> getActionContexts() {
    return strategies;
  }

}
