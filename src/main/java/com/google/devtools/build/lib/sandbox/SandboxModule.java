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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.apple.XcodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * This module provides the Sandbox spawn strategy.
 */
public final class SandboxModule extends BlazeModule {

  /** Tracks whether we are issuing the very first build within this Bazel server instance. */
  private static boolean firstBuild = true;

  /** Environment for the running command. */
  private @Nullable CommandEnvironment env;

  /** Path to the location of the sandboxes. */
  private @Nullable Path sandboxBase;

  /** Instance of the sandboxfs process in use, if enabled. */
  private @Nullable SandboxfsProcess sandboxfsProcess;

  /**
   * Collection of spawn runner instantiated during the executor setup.
   *
   * <p>We need this information to clean up the heavy subdirectories of the sandbox base on build
   * completion but to avoid wiping the whole sandbox base itself, which could be problematic across
   * builds.
   */
  private final Set<SpawnRunner> spawnRunners = new HashSet<>();

  /**
   * Whether to remove the sandbox worker directories after a build or not. Useful for debugging
   * to inspect the state of files on failures.
   */
  private boolean shouldCleanupSandboxBase;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(SandboxOptions.class)
        : ImmutableList.of();
  }

  /** Computes the path to the sandbox base tree for the given running command. */
  private static Path computeSandboxBase(SandboxOptions options, CommandEnvironment env)
      throws IOException {
    if (options.sandboxBase.isEmpty()) {
      return env.getOutputBase().getRelative("sandbox");
    } else {
      String dirName =
          String.format(
              "%s-sandbox.%s",
              env.getRuntime().getProductName(),
              Fingerprint.getHexDigest(env.getOutputBase().toString()));
      FileSystem fileSystem = env.getRuntime().getFileSystem();
      Path resolvedSandboxBase = fileSystem.getPath(options.sandboxBase).resolveSymbolicLinks();
      return resolvedSandboxBase.getRelative(dirName);
    }
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    // We can't assert that env is null because the Blaze runtime does not guarantee that
    // afterCommand() will be called if the command fails due to, e.g. a syntax error.
    this.env = env;
    env.getEventBus().register(this);

    // Don't attempt cleanup unless the executor is initialized.
    shouldCleanupSandboxBase = false;
  }

  @Override
  public void executorInit(CommandEnvironment cmdEnv, BuildRequest request, ExecutorBuilder builder)
      throws ExecutorInitException {
    checkNotNull(env, "env not initialized; was beforeCommand called?");
    try {
      setup(cmdEnv, builder);
    } catch (IOException e) {
      throw new ExecutorInitException("Failed to initialize sandbox", e);
    }
  }

  /**
   * Returns true if sandboxfs should be used for this build.
   *
   * <p>If the user set the use of sandboxfs as optional, this only returns true if the configured
   * sandboxfs binary is present and valid. If the user requested the use of sandboxfs as mandatory,
   * this throws an error if the binary is not valid.
   *
   * @param requested whether sandboxfs use was requested or not
   * @param binary path of the sandboxfs binary to use
   * @return true if sandboxfs can and should be used; false otherwise
   * @throws IOException if there are problems trying to determine the status of sandboxfs
   */
  private boolean shouldUseSandboxfs(TriState requested, PathFragment binary) throws IOException {
    switch (requested) {
      case AUTO:
        return RealSandboxfsProcess.isAvailable(binary);

      case NO:
        return false;

      case YES:
        if (!RealSandboxfsProcess.isAvailable(binary)) {
          throw new IOException(
              "sandboxfs explicitly requested but \""
                  + binary
                  + "\" could not be found or is not valid");
        }
        return true;
    }
    throw new IllegalStateException("Not reachable");
  }

  private void setup(CommandEnvironment cmdEnv, ExecutorBuilder builder)
      throws IOException {
    SandboxOptions options = checkNotNull(env.getOptions().getOptions(SandboxOptions.class));
    sandboxBase = computeSandboxBase(options, env);

    // Do not remove the sandbox base when --sandbox_debug was specified so that people can check
    // out the contents of the generated sandbox directories.
    shouldCleanupSandboxBase = !options.sandboxDebug;

    Path mountPoint = sandboxBase.getRelative("sandboxfs");

    if (sandboxfsProcess != null) {
      if (options.sandboxDebug) {
        env.getReporter()
            .handle(
                Event.info(
                    "Unmounting sandboxfs instance left behind on "
                        + mountPoint
                        + " by a previous command"));
      }
      sandboxfsProcess.destroy();
      sandboxfsProcess = null;
    }
    // SpawnExecutionPolicy#getId returns unique base directories for each sandboxed action during
    // the life of a Bazel server instance so we don't need to worry about stale directories from
    // previous builds. However, on the very first build of an instance of the server, we must
    // wipe old contents to avoid reusing stale directories.
    if (firstBuild && sandboxBase.exists()) {
      cmdEnv.getReporter().handle(Event.info("Deleting stale sandbox base " + sandboxBase));
      sandboxBase.deleteTree();
    }
    firstBuild = false;

    PathFragment sandboxfsPath = PathFragment.create(options.sandboxfsPath);
    boolean useSandboxfs;
    try (SilentCloseable c = Profiler.instance().profile("shouldUseSandboxfs")) {
      useSandboxfs = shouldUseSandboxfs(options.useSandboxfs, sandboxfsPath);
    }
    sandboxBase.createDirectoryAndParents();
    if (useSandboxfs) {
      mountPoint.createDirectory();
      Path logFile = sandboxBase.getRelative("sandboxfs.log");

      if (sandboxfsProcess == null) {
        if (options.sandboxDebug) {
          env.getReporter().handle(Event.info("Mounting sandboxfs instance on " + mountPoint));
        }
        sandboxfsProcess = RealSandboxfsProcess.mount(sandboxfsPath, mountPoint, logFile);
      }
    }

    Duration timeoutKillDelay =
        cmdEnv.getOptions().getOptions(LocalExecutionOptions.class).getLocalSigkillGraceSeconds();

    boolean processWrapperSupported = ProcessWrapperSandboxedSpawnRunner.isSupported(cmdEnv);
    boolean linuxSandboxSupported = LinuxSandboxedSpawnRunner.isSupported(cmdEnv);
    boolean darwinSandboxSupported = DarwinSandboxedSpawnRunner.isSupported(cmdEnv);

    // This works on most platforms, but isn't the best choice, so we put it first and let later
    // platform-specific sandboxing strategies become the default.
    if (processWrapperSupported) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new ProcessWrapperSandboxedSpawnRunner(
                  cmdEnv, sandboxBase, cmdEnv.getRuntime().getProductName(), timeoutKillDelay));
      spawnRunners.add(spawnRunner);
      builder.addActionContext(
          new ProcessWrapperSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    if (options.enableDockerSandbox) {
      // This strategy uses Docker to execute spawns. It should work on all platforms that support
      // Docker.
      Path pathToDocker = getPathToDockerClient(cmdEnv);
      // DockerSandboxedSpawnRunner.isSupported is expensive! It runs docker as a subprocess, and
      // docker hangs sometimes.
      if (pathToDocker != null && DockerSandboxedSpawnRunner.isSupported(cmdEnv, pathToDocker)) {
        String defaultImage = options.dockerImage;
        boolean useCustomizedImages = options.dockerUseCustomizedImages;
        SpawnRunner spawnRunner =
            withFallback(
                cmdEnv,
                new DockerSandboxedSpawnRunner(
                    cmdEnv,
                    pathToDocker,
                    sandboxBase,
                    defaultImage,
                    timeoutKillDelay,
                    useCustomizedImages));
        spawnRunners.add(spawnRunner);
        builder.addActionContext(
            new DockerSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
      }
    } else if (options.dockerVerbose) {
      cmdEnv.getReporter().handle(Event.info(
          "Docker sandboxing disabled. Use the '--experimental_enable_docker_sandbox' command "
          + "line option to enable it"));
    }

    // This is the preferred sandboxing strategy on Linux.
    if (linuxSandboxSupported) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              LinuxSandboxedStrategy.create(
                  cmdEnv,
                  sandboxBase,
                  timeoutKillDelay,
                  sandboxfsProcess,
                  options.sandboxfsMapSymlinkTargets));
      spawnRunners.add(spawnRunner);
      builder.addActionContext(new LinuxSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    // This is the preferred sandboxing strategy on macOS.
    if (darwinSandboxSupported) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new DarwinSandboxedSpawnRunner(
                  cmdEnv,
                  sandboxBase,
                  timeoutKillDelay,
                  sandboxfsProcess,
                  options.sandboxfsMapSymlinkTargets));
      spawnRunners.add(spawnRunner);
      builder.addActionContext(new DarwinSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner));
    }

    if (processWrapperSupported || linuxSandboxSupported || darwinSandboxSupported) {
      // This makes the "sandboxed" strategy available via --spawn_strategy=sandboxed,
      // but it is not necessarily the default.
      builder.addStrategyByContext(SpawnActionContext.class, "sandboxed");

      // This makes the "sandboxed" strategy the default Spawn strategy, unless it is
      // overridden by a later BlazeModule.
      builder.addStrategyByMnemonic("", ImmutableList.of("sandboxed"));
    }
  }

  private static Path getPathToDockerClient(CommandEnvironment cmdEnv) {
    String path = cmdEnv.getClientEnv().getOrDefault("PATH", "");

    // TODO(philwo): Does this return the correct result if one of the elements intentionally ends
    // in white space?
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
            return dentry;
          }
        }
      } catch (IOException e) {
        continue;
      }
    }

    return null;
  }

  private static SpawnRunner withFallback(CommandEnvironment env, SpawnRunner sandboxSpawnRunner) {
    return new SandboxFallbackSpawnRunner(sandboxSpawnRunner, createFallbackRunner(env));
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
            localEnvProvider,
            env.getBlazeWorkspace().getBinTools());
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
      if (sandboxSpawnRunner.canExec(spawn)) {
        return sandboxSpawnRunner.exec(spawn, context);
      } else {
        return fallbackSpawnRunner.exec(spawn, context);
      }
    }

    @Override
    public boolean canExec(Spawn spawn) {
      return sandboxSpawnRunner.canExec(spawn) || fallbackSpawnRunner.canExec(spawn);
    }

    @Override
    public void cleanupSandboxBase(Path sandboxBase) throws IOException {
      sandboxSpawnRunner.cleanupSandboxBase(sandboxBase);
      fallbackSpawnRunner.cleanupSandboxBase(sandboxBase);
    }
  }

  /**
   * Unmounts an existing sandboxfs instance unless the user asked not to by providing the {@code
   * --sandbox_debug} flag.
   */
  private void unmountSandboxfs() {
    if (sandboxfsProcess != null) {
      if (shouldCleanupSandboxBase) {
        sandboxfsProcess.destroy();
        sandboxfsProcess = null;
      } else {
        checkNotNull(env, "env not initialized; was beforeCommand called?");
        env.getReporter()
            .handle(Event.info("Leaving sandboxfs mounted because of --sandbox_debug"));
      }
    }
  }

  /** Silently tries to unmount an existing sandboxfs instance, ignoring errors. */
  private void tryUnmountSandboxfsOnShutdown() {
    if (sandboxfsProcess != null) {
      sandboxfsProcess.destroy();
      sandboxfsProcess = null;
    }
  }

  @Subscribe
  public void buildComplete(@SuppressWarnings("unused") BuildCompleteEvent event) {
    unmountSandboxfs();
  }

  @Subscribe
  public void buildInterrupted(@SuppressWarnings("unused") BuildInterruptedEvent event) {
    unmountSandboxfs();
  }

  @Override
  public void afterCommand() {
    checkNotNull(env, "env not initialized; was beforeCommand called?");

    if (shouldCleanupSandboxBase) {
      try {
        checkNotNull(sandboxBase, "shouldCleanupSandboxBase implies sandboxBase has been set");
        for (SpawnRunner spawnRunner : spawnRunners) {
          spawnRunner.cleanupSandboxBase(sandboxBase);
        }
      } catch (IOException e) {
        env.getReporter()
            .handle(Event.warn("Failed to delete contents of sandbox " + sandboxBase + ": " + e));
      }
      shouldCleanupSandboxBase = false;

      checkState(
          sandboxfsProcess == null,
          "sandboxfs instance should have been shut down at this "
              + "point; were the buildComplete/buildInterrupted events sent?");
      sandboxBase = null;
    }

    env.getEventBus().unregister(this);
    env = null;
  }

  @Override
  public void blazeShutdown() {
    tryUnmountSandboxfsOnShutdown();
  }

  @Override
  public void blazeShutdownOnCrash() {
    tryUnmountSandboxfsOnShutdown();
  }
}
