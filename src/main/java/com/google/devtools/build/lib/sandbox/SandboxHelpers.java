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
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/** Helper methods that are shared by the different sandboxing strategies in this package. */
public final class SandboxHelpers {

  static void fallbackToNonSandboxedExecution(
      Spawn spawn, ActionExecutionContext actionExecutionContext, Executor executor)
      throws ExecException {
    StandaloneSpawnStrategy standaloneStrategy =
        Preconditions.checkNotNull(executor.getContext(StandaloneSpawnStrategy.class));
    standaloneStrategy.exec(spawn, actionExecutionContext);
  }

  static void reportSubcommand(Executor executor, Spawn spawn) {
    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(spawn);
    }
  }

  public static ImmutableSet<PathFragment> getOutputFiles(Spawn spawn) {
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
    // Allow network access, when --java_debug is specified, otherwise we can't connect to the
    // remote debug server of the test. This intentionally overrides the "block-network" execution
    // tag.
    if (buildRequest
        .getOptions(BuildConfiguration.Options.class)
        .testArguments
        .contains("--wrapper_script_flag=--debug")) {
      return true;
    }

    // If the Spawn requests to block network access, do so.
    if (spawn.getExecutionInfo().containsKey("block-network")) {
      return false;
    }

    // Network access is allowed by default.
    return true;
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
