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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;

/**
 * Provides the sandboxed spawn strategy.
 */
final class SandboxActionContextProvider extends ActionContextProvider {
  private final ImmutableList<ActionContext> contexts;

  private SandboxActionContextProvider(ImmutableList<ActionContext> contexts) {
    this.contexts = contexts;
  }

  public static SandboxActionContextProvider create(CommandEnvironment cmdEnv, Path sandboxBase)
      throws IOException {
    ImmutableList.Builder<ActionContext> contexts = ImmutableList.builder();

    OptionsProvider options = cmdEnv.getOptions();
    int timeoutGraceSeconds =
        options.getOptions(LocalExecutionOptions.class).localSigkillGraceSeconds;
    String productName = cmdEnv.getRuntime().getProductName();

    // This works on most platforms, but isn't the best choice, so we put it first and let later
    // platform-specific sandboxing strategies become the default.
    if (ProcessWrapperSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new ProcessWrapperSandboxedSpawnRunner(
                  cmdEnv, sandboxBase, productName, timeoutGraceSeconds));
      contexts.add(new ProcessWrapperSandboxedStrategy(spawnRunner));
    }

    // This is the preferred sandboxing strategy on Linux.
    if (LinuxSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              LinuxSandboxedStrategy.create(cmdEnv, sandboxBase, productName, timeoutGraceSeconds));
      contexts.add(new LinuxSandboxedStrategy(spawnRunner));
    }

    // This is the preferred sandboxing strategy on macOS.
    if (DarwinSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new DarwinSandboxedSpawnRunner(
                  cmdEnv, sandboxBase, productName, timeoutGraceSeconds));
      contexts.add(new DarwinSandboxedStrategy(spawnRunner));
    }

    return new SandboxActionContextProvider(contexts.build());
  }

  private static SpawnRunner withFallback(CommandEnvironment env, SpawnRunner sandboxSpawnRunner) {
    return new SandboxFallbackSpawnRunner(sandboxSpawnRunner,  createFallbackRunner(env));
  }

  private static SpawnRunner createFallbackRunner(CommandEnvironment env) {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    LocalEnvProvider localEnvProvider =
        OS.getCurrent() == OS.DARWIN
            ? new XCodeLocalEnvProvider()
            : LocalEnvProvider.ADD_TEMP_POSIX;
    return
        new LocalSpawnRunner(
            env.getExecRoot(),
            localExecutionOptions,
            ResourceManager.instance(),
            env.getRuntime().getProductName(),
            localEnvProvider);
  }

  @Override
  public Iterable<? extends ActionContext> getActionContexts() {
    return contexts;
  }

  private static final class SandboxFallbackSpawnRunner implements SpawnRunner {
    private final SpawnRunner sandboxSpawnRunner;
    private final SpawnRunner fallbackSpawnRunner;

    SandboxFallbackSpawnRunner(SpawnRunner sandboxSpawnRunner, SpawnRunner fallbackSpawnRunner) {
      this.sandboxSpawnRunner = sandboxSpawnRunner;
      this.fallbackSpawnRunner = fallbackSpawnRunner;
    }

    @Override
    public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
        throws InterruptedException, IOException, ExecException {
      if (!Spawns.mayBeSandboxed(spawn)) {
        return fallbackSpawnRunner.exec(spawn, policy);
      } else {
        return sandboxSpawnRunner.exec(spawn, policy);
      }
    }
  }
}
