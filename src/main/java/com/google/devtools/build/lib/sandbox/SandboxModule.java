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

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * This module provides the Sandbox spawn strategy.
 */
public final class SandboxModule extends BlazeModule {

  /** Environment for the running command. */
  private @Nullable CommandEnvironment env;

  /** Path to the location of the sandboxes. */
  private @Nullable Path sandboxBase;

  /** Instance of the sandboxfs process in use, if enabled. */
  private @Nullable SandboxfsProcess sandboxfsProcess;

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
      String dirName = String.format("%s-sandbox.%s", env.getRuntime().getProductName(),
          Fingerprint.md5Digest(env.getOutputBase().toString()));
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
    sandboxfsProcess = null;
    shouldCleanupSandboxBase = false;
  }

  @Override
  public void executorInit(CommandEnvironment cmdEnv, BuildRequest request, ExecutorBuilder builder)
      throws ExecutorInitException {
    checkNotNull(env, "env not initialized; was beforeCommand called?");

    SandboxOptions options = env.getOptions().getOptions(SandboxOptions.class);
    checkNotNull(options, "We were told to initialize the executor but the SandboxOptions are "
        + "not present; were they registered for all build commands?");

    try {
      sandboxBase = computeSandboxBase(options, env);
    } catch (IOException e) {
      throw new ExecutorInitException(
          "--experimental_sandbox_base points to an invalid directory", e);
    }

    ActionContextProvider provider;
    try {
      // Ensure that each build starts with a clean sandbox base directory. Otherwise using the `id`
      // that is provided by SpawnExecutionPolicy#getId to compute a base directory for a sandbox
      // might result in an already existing directory.
      if (sandboxBase.exists()) {
        FileSystemUtils.deleteTree(sandboxBase);
      }

      sandboxBase.createDirectoryAndParents();
      if (options.useSandboxfs) {
        Path mountPoint = sandboxBase.getRelative("sandboxfs");
        mountPoint.createDirectory();
        Path logFile = sandboxBase.getRelative("sandboxfs.log");

        env.getReporter().handle(Event.info("Mounting sandboxfs instance on " + mountPoint));
        sandboxfsProcess = RealSandboxfsProcess.mount(
            PathFragment.create(options.sandboxfsPath), mountPoint, logFile);
        provider = SandboxActionContextProvider.create(cmdEnv, sandboxBase, sandboxfsProcess);
      } else {
        provider = SandboxActionContextProvider.create(cmdEnv, sandboxBase, null);
      }
    } catch (IOException e) {
      throw new ExecutorInitException("Failed to initialize sandbox", e);
    }
    builder.addActionContextProvider(provider);
    builder.addActionContextConsumer(new SandboxActionContextConsumer(cmdEnv));

    // Do not remove the sandbox base when --sandbox_debug was specified so that people can check
    // out the contents of the generated sandbox directories.
    shouldCleanupSandboxBase = !options.sandboxDebug;
  }

  private void unmountSandboxfs(String reason) {
    if (sandboxfsProcess != null) {
      checkNotNull(env, "env not initialized; was beforeCommand called?");
      env.getReporter().handle(Event.info(reason));
      // TODO(jmmv): This can be incredibly slow.  Either fix sandboxfs or do it in the background.
      sandboxfsProcess.destroy();
      sandboxfsProcess = null;
    }
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    unmountSandboxfs("Build complete; unmounting sandboxfs...");
  }

  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    unmountSandboxfs("Build interrupted; unmounting sandboxfs...");
  }

  @Override
  public void afterCommand() {
    checkNotNull(env, "env not initialized; was beforeCommand called?");

    if (shouldCleanupSandboxBase) {
      try {
        FileSystemUtils.deleteTree(sandboxBase);
      } catch (IOException e) {
        env.getReporter().handle(Event.warn("Failed to delete sandbox base " + sandboxBase
            + ": " + e));
      }
      shouldCleanupSandboxBase = false;
    }

    checkState(sandboxfsProcess == null, "sandboxfs instance should have been shut down at this "
        + "point; were the buildComplete/buildInterrupted events sent?");
    sandboxBase = null;

    env.getEventBus().unregister(this);
    env = null;
  }
}
