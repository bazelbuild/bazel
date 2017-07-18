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
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/**
 * Provides the sandboxed spawn strategy.
 */
final class SandboxActionContextProvider extends ActionContextProvider {
  private final ImmutableList<ActionContext> contexts;

  private SandboxActionContextProvider(ImmutableList<ActionContext> contexts) {
    this.contexts = contexts;
  }

  public static SandboxActionContextProvider create(
      CommandEnvironment cmdEnv, BuildRequest buildRequest, Path sandboxBase) throws IOException {
    ImmutableList.Builder<ActionContext> contexts = ImmutableList.builder();

    int timeoutGraceSeconds =
        buildRequest.getOptions(LocalExecutionOptions.class).localSigkillGraceSeconds;
    boolean verboseFailures = buildRequest.getOptions(ExecutionOptions.class).verboseFailures;
    String productName = cmdEnv.getRuntime().getProductName();

    // This works on most platforms, but isn't the best choice, so we put it first and let later
    // platform-specific sandboxing strategies become the default.
    if (ProcessWrapperSandboxedSpawnRunner.isSupported(cmdEnv)) {
      contexts.add(
          new ProcessWrapperSandboxedStrategy(
              cmdEnv,
              buildRequest,
              sandboxBase,
              verboseFailures,
              productName,
              timeoutGraceSeconds));
    }

    // This is the preferred sandboxing strategy on Linux.
    if (LinuxSandboxedSpawnRunner.isSupported(cmdEnv)) {
      contexts.add(
          LinuxSandboxedStrategy.create(
              cmdEnv, buildRequest, sandboxBase, verboseFailures, timeoutGraceSeconds));
    }

    // This is the preferred sandboxing strategy on macOS.
    if (DarwinSandboxedSpawnRunner.isSupported(cmdEnv)) {
      contexts.add(
          new DarwinSandboxedStrategy(
              cmdEnv,
              buildRequest,
              sandboxBase,
              verboseFailures,
              productName,
              timeoutGraceSeconds));
    }

    return new SandboxActionContextProvider(contexts.build());
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    return contexts;
  }
}
