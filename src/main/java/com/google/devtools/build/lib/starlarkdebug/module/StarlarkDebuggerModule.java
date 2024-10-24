// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdebug.module;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkdebug.server.StarlarkDebugServer;
import com.google.devtools.build.lib.starlarkdebug.server.StarlarkDebugServer.DebugCallback;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import net.starlark.java.eval.Debug;

/** Blaze module for setting up Starlark debugging. */
public final class StarlarkDebuggerModule extends BlazeModule {
  @Override
  public void beforeCommand(CommandEnvironment env) {
    // Conditionally enable debugging
    StarlarkDebuggerOptions buildOptions =
        env.getOptions().getOptions(StarlarkDebuggerOptions.class);
    boolean enabled = buildOptions != null && buildOptions.debugStarlark;
    if (enabled) {
      initializeDebugging(
          env, buildOptions.debugServerPort, buildOptions.verboseLogs, buildOptions.resetAnalysis);
    } else {
      disableDebugging();
    }
  }

  @Override
  public void afterCommand() {
    disableDebugging();
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(StarlarkDebuggerOptions.class)
        : ImmutableList.of();
  }

  @Override
  public void blazeShutdown() {
    disableDebugging();
  }

  @Override
  public void blazeShutdownOnCrash(DetailedExitCode exitCode) {
    disableDebugging();
  }

  private static void initializeDebugging(
      CommandEnvironment env, int debugPort, boolean verboseLogs, boolean resetAnalysis) {
    try {
      DebugCallback callback =
          resetAnalysis ? getBreakpointInvalidatingCallback(env) : DebugCallback.noop();
      StarlarkDebugServer server =
          StarlarkDebugServer.createAndWaitForConnection(
              env.getReporter(), debugPort, verboseLogs, callback);
      Debug.setDebugger(server);
      // we need to block otherwise the build (i.e. analysis) may start and the request to set
      // breakpoints may lose the race to delete skyframe nodes
      callback.maybeBlockBeforeStart();
    } catch (IOException | InterruptedException e) {
      env.getReporter()
          .handle(Event.error("Error while setting up the debug server: " + e.getMessage()));
    }
  }

  private static DebugCallback getBreakpointInvalidatingCallback(CommandEnvironment env) {
    return new DebugCallback() {
      private final CountDownLatch latch = new CountDownLatch(1);

      @Override
      public void beforeDebuggingStart(ImmutableSet<String> breakPointPaths) {
        handle(Event.debug("resetting analysis for: " + breakPointPaths));
        // we delete the FILE nodes for all paths with breakpoints to force re-analysis. Ideally,
        // we should perhaps invalidate bzl-compile (for .bzl files) and package(??) (for BUILD
        // files) but computing the right arguments for those skykeys is a lot harder.
        env.getSkyframeExecutor()
            .getEvaluator()
            .delete(
                skyKey ->
                    Objects.equals(skyKey.functionName(), SkyFunctions.FILE)
                        && breakPointPaths.contains(
                            ((RootedPath) skyKey.argument()).asPath().toString()));
        handle(Event.debug("analysis reset complete"));
        // unblock the build
        latch.countDown();
      }

      @Override
      public void maybeBlockBeforeStart() throws InterruptedException {
        handle(Event.debug("waiting for breakpoints before executing build"));
        latch.await();
      }

      @Override
      public void onClose() {
        latch.countDown();
      }

      private void handle(Event event) {
        env.getReporter().handle(event);
      }
    };
  }

  private static void disableDebugging() {
    Debug.setDebugger(null);
  }
}
