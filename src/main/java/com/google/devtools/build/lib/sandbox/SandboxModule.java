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
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
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
  @Nullable private CommandEnvironment env;

  /** Path to the location of the sandboxes. */
  @Nullable private Path sandboxBase;

  /** Instance of the sandboxfs process in use, if enabled. */
  @Nullable private SandboxfsProcess sandboxfsProcess;

  /**
   * Collection of spawn runner instantiated during the executor setup.
   *
   * <p>We need this information to clean up the heavy subdirectories of the sandbox base on build
   * completion but to avoid wiping the whole sandbox base itself, which could be problematic across
   * builds.
   */
  private final Set<SpawnRunner> spawnRunners = new HashSet<>();

  /**
   * Handler to process expensive tree deletions outside of the critical path.
   *
   * <p>Sandboxing creates one separate tree for each action, and this tree is used to run the
   * action commands in. These trees are disjoint for all actions and have unique identifiers.
   * Therefore, there is no need for their deletion (which can be very expensive) to happen in the
   * critical path -- so if the user so wishes, we process those deletions asynchronously.
   */
  @Nullable private TreeDeleter treeDeleter;

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
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env)
      throws ExecutorInitException {
    checkNotNull(env, "env not initialized; was beforeCommand called?");
    try {
      setup(env, registryBuilder);
    } catch (IOException e) {
      throw new ExecutorInitException("Failed to initialize sandbox", e);
    }
  }

  /**
   * Returns true if windows-sandbox should be used for this build.
   *
   * <p>Returns true if requested in ["auto", "yes"] and binary is valid. Throws an error if state
   * is "yes" and binary is not valid.
   *
   * @param requested whether windows-sandbox use was requested or not
   * @param binary path of the windows-sandbox binary to use, can be absolute or relative path
   * @return true if windows-sandbox can and should be used; false otherwise
   * @throws IOException if there are problems trying to determine the status of windows-sandbox
   */
  private boolean shouldUseWindowsSandbox(TriState requested, PathFragment binary)
      throws IOException {
    switch (requested) {
      case AUTO:
        return WindowsSandboxUtil.isAvailable(binary);

      case NO:
        return false;

      case YES:
        if (!WindowsSandboxUtil.isAvailable(binary)) {
          throw new IOException(
              "windows-sandbox explicitly requested but \""
                  + binary
                  + "\" could not be found or is not valid");
        }
        return true;
    }
    throw new IllegalStateException("Not reachable");
  }

  private void setup(CommandEnvironment cmdEnv, SpawnStrategyRegistry.Builder builder)
      throws IOException {
    SandboxOptions options = checkNotNull(env.getOptions().getOptions(SandboxOptions.class));
    sandboxBase = computeSandboxBase(options, env);

    SandboxHelpers helpers = new SandboxHelpers(options.delayVirtualInputMaterialization);

    // Do not remove the sandbox base when --sandbox_debug was specified so that people can check
    // out the contents of the generated sandbox directories.
    shouldCleanupSandboxBase = !options.sandboxDebug;

    // If there happens to be any live tree deleter from a previous build and it's different than
    // the one we want now, leave it alone (i.e. don't attempt to wait for pending deletions). Its
    // deletions shouldn't overlap any new directories we create during this build (because the
    // identifiers in the subdirectories will be different).
    if (options.asyncTreeDeleteIdleThreads == 0) {
      if (!(treeDeleter instanceof SynchronousTreeDeleter)) {
        treeDeleter = new SynchronousTreeDeleter();
      }
    } else {
      if (!(treeDeleter instanceof AsynchronousTreeDeleter)) {
        treeDeleter = new AsynchronousTreeDeleter();
      }
    }

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
    sandboxBase.createDirectoryAndParents();
    if (options.useSandboxfs != TriState.NO) {
      mountPoint.createDirectory();
      Path logFile = sandboxBase.getRelative("sandboxfs.log");

      if (sandboxfsProcess == null) {
        if (options.sandboxDebug) {
          env.getReporter().handle(Event.info("Mounting sandboxfs instance on " + mountPoint));
        }
        try (SilentCloseable c = Profiler.instance().profile("mountSandboxfs")) {
          sandboxfsProcess = RealSandboxfsProcess.mount(sandboxfsPath, mountPoint, logFile);
        } catch (IOException e) {
          if (options.sandboxDebug) {
            env.getReporter()
                .handle(
                    Event.info(
                        "sandboxfs failed to mount due to " + e.getMessage() + "; ignoring"));
          }
          if (options.useSandboxfs == TriState.YES) {
            throw e;
          }
        }
      }
    }

    PathFragment windowsSandboxPath = PathFragment.create(options.windowsSandboxPath);
    boolean windowsSandboxSupported;
    try (SilentCloseable c = Profiler.instance().profile("shouldUseWindowsSandbox")) {
      windowsSandboxSupported =
          shouldUseWindowsSandbox(options.useWindowsSandbox, windowsSandboxPath);
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
                  helpers,
                  cmdEnv,
                  sandboxBase,
                  timeoutKillDelay,
                  sandboxfsProcess,
                  options.sandboxfsMapSymlinkTargets,
                  treeDeleter));
      spawnRunners.add(spawnRunner);
      builder.registerStrategy(
          new ProcessWrapperSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner),
          "sandboxed",
          "processwrapper-sandbox");
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
                    helpers,
                    cmdEnv,
                    pathToDocker,
                    sandboxBase,
                    defaultImage,
                    timeoutKillDelay,
                    useCustomizedImages,
                    treeDeleter));
        spawnRunners.add(spawnRunner);
        builder.registerStrategy(
            new DockerSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner), "docker");
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
                  helpers,
                  cmdEnv,
                  sandboxBase,
                  timeoutKillDelay,
                  sandboxfsProcess,
                  options.sandboxfsMapSymlinkTargets,
                  treeDeleter));
      spawnRunners.add(spawnRunner);
      builder.registerStrategy(
          new LinuxSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner),
          "sandboxed",
          "linux-sandbox");
    }

    // This is the preferred sandboxing strategy on macOS.
    if (darwinSandboxSupported) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new DarwinSandboxedSpawnRunner(
                  helpers,
                  cmdEnv,
                  sandboxBase,
                  timeoutKillDelay,
                  sandboxfsProcess,
                  options.sandboxfsMapSymlinkTargets,
                  treeDeleter));
      spawnRunners.add(spawnRunner);
      builder.registerStrategy(
          new DarwinSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner),
          "sandboxed",
          "darwin-sandbox");
    }

    if (windowsSandboxSupported) {
      SpawnRunner spawnRunner =
          withFallback(
              cmdEnv,
              new WindowsSandboxedSpawnRunner(
                  helpers, cmdEnv, timeoutKillDelay, windowsSandboxPath));
      spawnRunners.add(spawnRunner);
      builder.registerStrategy(
          new WindowsSandboxedStrategy(cmdEnv.getExecRoot(), spawnRunner),
          "sandboxed",
          "windows-sandbox");
    }

    if (processWrapperSupported
        || linuxSandboxSupported
        || darwinSandboxSupported
        || windowsSandboxSupported) {
      // This makes the "sandboxed" strategy the default Spawn strategy, unless it is
      // overridden by a later BlazeModule.
      builder.setDefaultStrategies(ImmutableList.of("sandboxed"));
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
    return new LocalSpawnRunner(
        env.getExecRoot(),
        localExecutionOptions,
        env.getLocalResourceManager(),
        LocalEnvProvider.forCurrentOs(env.getClientEnv()),
        env.getBlazeWorkspace().getBinTools(),
        // TODO(buchgr): Replace singleton by a command-scoped RunfilesTreeUpdater
        RunfilesTreeUpdater.INSTANCE);
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
    public void cleanupSandboxBase(Path sandboxBase, TreeDeleter treeDeleter) throws IOException {
      sandboxSpawnRunner.cleanupSandboxBase(sandboxBase, treeDeleter);
      fallbackSpawnRunner.cleanupSandboxBase(sandboxBase, treeDeleter);
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

    SandboxOptions options = env.getOptions().getOptions(SandboxOptions.class);
    int asyncTreeDeleteThreads = options != null ? options.asyncTreeDeleteIdleThreads : 0;
    if (treeDeleter != null && asyncTreeDeleteThreads > 0) {
      // If asynchronous deletions were requested, they may still be ongoing so let them be: trying
      // to delete the base tree synchronously could fail as we can race with those other deletions,
      // and scheduling an asynchronous deletion could race with future builds.
      AsynchronousTreeDeleter treeDeleter =
          (AsynchronousTreeDeleter) checkNotNull(this.treeDeleter);
      treeDeleter.setThreads(asyncTreeDeleteThreads);
    }

    if (shouldCleanupSandboxBase) {
      try {
        checkNotNull(sandboxBase, "shouldCleanupSandboxBase implies sandboxBase has been set");
        for (SpawnRunner spawnRunner : spawnRunners) {
          spawnRunner.cleanupSandboxBase(sandboxBase, treeDeleter);
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

  private void commonShutdown() {
    tryUnmountSandboxfsOnShutdown();

    // Try to clean up as much garbage as possible, if there happens to be any. This will delay
    // server termination but it's the nice thing to do. If the user gets impatient, they can always
    // kill us again.
    if (treeDeleter != null) {
      try {
        treeDeleter.shutdown();
      } finally {
        treeDeleter = null; // Avoid potential reexecution if we crash.
      }
    }
  }

  @Override
  public void blazeShutdown() {
    commonShutdown();
  }

  @Override
  public void blazeShutdownOnCrash() {
    commonShutdown();
  }
}
