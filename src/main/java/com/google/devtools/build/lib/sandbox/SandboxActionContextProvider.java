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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XcodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Provides the sandboxed spawn strategy.
 */
final class SandboxActionContextProvider extends ActionContextProvider {
  private final ImmutableList<ActionContext> contexts;

  private SandboxActionContextProvider(ImmutableList<ActionContext> contexts) {
    this.contexts = contexts;
  }

  public static SandboxActionContextProvider create(CommandEnvironment cmdEnv, Path sandboxBase,
      @Nullable SandboxfsProcess process)
      throws IOException {
    ImmutableList.Builder<ActionContext> contexts = ImmutableList.builder();

    OptionsProvider options = cmdEnv.getOptions();
    Duration timeoutKillDelay =
        Duration.ofSeconds(
            options.getOptions(LocalExecutionOptions.class).localSigkillGraceSeconds);

    // This works on most platforms, but isn't the best choice, so we put it first and let later
    // platform-specific sandboxing strategies become the default.
    if (ProcessWrapperSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new ProcessWrapperSandboxedSpawnRunner(
                  cmdEnv, sandboxBase, cmdEnv.getRuntime().getProductName(), timeoutKillDelay));
      contexts.add(new ProcessWrapperSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    SandboxOptions sandboxOptions = options.getOptions(SandboxOptions.class);

    if (sandboxOptions.enableDockerSandbox) {
      // This strategy uses Docker to execute spawns. It should work on all platforms that support
      // Docker.
      getPathToDockerClient(cmdEnv)
          .ifPresent(
              dockerClient -> {
                if (DockerSandboxedSpawnRunner.isSupported(cmdEnv, dockerClient)) {
                  String defaultImage = sandboxOptions.dockerImage;
                  boolean useCustomizedImages = sandboxOptions.dockerUseCustomizedImages;
                  SpawnRunner spawnRunner =
                      withFallback(
                          cmdEnv,
                          new DockerSandboxedSpawnRunner(
                              cmdEnv,
                              dockerClient,
                              sandboxBase,
                              defaultImage,
                              timeoutKillDelay,
                              useCustomizedImages));
                  contexts.add(new DockerSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
                }
              });
    } else if (sandboxOptions.dockerVerbose) {
      cmdEnv.getReporter().handle(Event.info(
          "Docker sandboxing disabled. Use the '--experimental_enable_docker_sandbox' command "
          + "line option to enable it"));
    }

    // This is the preferred sandboxing strategy on Linux.
    if (LinuxSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              LinuxSandboxedStrategy.create(cmdEnv, sandboxBase, timeoutKillDelay, process));
      contexts.add(new LinuxSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    // This is the preferred sandboxing strategy on macOS.
    if (DarwinSandboxedSpawnRunner.isSupported(cmdEnv)) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new DarwinSandboxedSpawnRunner(cmdEnv, sandboxBase, timeoutKillDelay, process));
      contexts.add(new DarwinSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    return new SandboxActionContextProvider(contexts.build());
  }

  private static Optional<Path> getPathToDockerClient(CommandEnvironment cmdEnv) {
    String path = cmdEnv.getClientEnv().getOrDefault("PATH", "");

    Splitter pathSplitter =
        Splitter.on(OS.getCurrent() == OS.WINDOWS ? ';' : ':').trimResults().omitEmptyStrings();

    FileSystem fs = cmdEnv.getRuntime().getFileSystem();

    for (String pathElement : pathSplitter.split(path)) {
      // Sometimes the PATH contains the non-absolute entry "." - this resolves it against the
      // current working directory.
      pathElement = new File(pathElement).getAbsolutePath();
      try {
        for (Path dentry : fs.getPath(pathElement).getDirectoryEntries()) {
          if (dentry.getBaseName().replace(".exe", "").equals("docker")) {
            return Optional.of(dentry);
          }
        }
      } catch (IOException e) {
        continue;
      }
    }

    return Optional.empty();
  }

  private static SpawnRunner withFallback(CommandEnvironment env, SpawnRunner sandboxSpawnRunner) {
    return new SandboxFallbackSpawnRunner(sandboxSpawnRunner,  createFallbackRunner(env));
  }

  private static SpawnRunner createFallbackRunner(CommandEnvironment env) {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    LocalEnvProvider localEnvProvider =
        OS.getCurrent() == OS.DARWIN
            ? new XcodeLocalEnvProvider(env.getClientEnv())
            : new PosixLocalEnvProvider(env.getClientEnv());
    return
        new LocalSpawnRunner(
            env.getExecRoot(),
            localExecutionOptions,
            ResourceManager.instance(),
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
    public String getName() {
      return "sandbox-fallback";
    }

    @Override
    public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
        throws InterruptedException, IOException, ExecException {
      if (!Spawns.mayBeSandboxed(spawn)) {
        return fallbackSpawnRunner.exec(spawn, context);
      } else {
        return sandboxSpawnRunner.exec(spawn, context);
      }
    }
  }
}
