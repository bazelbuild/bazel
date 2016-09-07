// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/** Helper methods that are shared by the different sandboxing strategies in this package. */
final class SandboxHelpers {

  static void lazyCleanup(ExecutorService backgroundWorkers, final SandboxRunner runner) {
    // By deleting the sandbox directory in the background, we avoid having to wait for it to
    // complete before returning from the action, which improves performance.
    backgroundWorkers.execute(
        new Runnable() {
          @Override
          public void run() {
            try {
              while (!Thread.currentThread().isInterrupted()) {
                try {
                  runner.cleanup();
                  return;
                } catch (IOException e2) {
                  // Sleep & retry.
                  Thread.sleep(250);
                }
              }
            } catch (InterruptedException e) {
              // Mark ourselves as interrupted and then exit.
              Thread.currentThread().interrupt();
            }
          }
        });
  }

  static void fallbackToNonSandboxedExecution(
      Spawn spawn, ActionExecutionContext actionExecutionContext, Executor executor)
      throws ExecException {
    StandaloneSpawnStrategy standaloneStrategy =
        Preconditions.checkNotNull(executor.getContext(StandaloneSpawnStrategy.class));
    standaloneStrategy.exec(spawn, actionExecutionContext);
  }

  static void reportSubcommand(Executor executor, Spawn spawn) {
    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(
          Label.print(spawn.getOwner().getLabel())
              + " ["
              + spawn.getResourceOwner().prettyPrint()
              + "]",
          spawn.asShellCommand(executor.getExecRoot()));
    }
  }

  static ImmutableSet<PathFragment> getOutputFiles(Spawn spawn) {
    Builder<PathFragment> outputFiles = ImmutableSet.builder();
    for (PathFragment optionalOutput : spawn.getOptionalOutputFiles()) {
      Preconditions.checkArgument(!optionalOutput.isAbsolute());
      outputFiles.add(optionalOutput);
    }
    for (ActionInput output : spawn.getOutputFiles()) {
      outputFiles.add(new PathFragment(output.getExecPathString()));
    }
    return outputFiles.build();
  }

  static boolean shouldAllowNetwork(BuildRequest buildRequest, Spawn spawn) {
    // If we don't run tests, allow network access.
    if (!buildRequest.shouldRunTests()) {
      return true;
    }

    // If the Spawn specifically requests network access, allow it.
    if (spawn.getExecutionInfo().containsKey("requires-network")) {
      return true;
    }

    // Allow network access, when --java_debug is specified, otherwise we can't connect to the
    // remote debug server of the test.
    if (buildRequest
        .getOptions(BuildConfiguration.Options.class)
        .testArguments
        .contains("--wrapper_script_flag=--debug")) {
      return true;
    }

    return false;
  }

  static void postActionStatusMessage(Executor executor, Spawn spawn) {
    executor
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "sandbox"));
  }

  static Path getSandboxRoot(
      BlazeDirectories blazeDirs, String productName, UUID uuid, AtomicInteger execCounter) {
    return blazeDirs
        .getOutputBase()
        .getRelative(productName + "-sandbox")
        .getRelative(uuid + "-" + execCounter.getAndIncrement());
  }
}
