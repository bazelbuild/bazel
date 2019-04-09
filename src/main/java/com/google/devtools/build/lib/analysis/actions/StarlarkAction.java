// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * A Starlark specific SpawnAction.
 *
 * <p>Note: current implementation is empty: StarlarkAction is equivalent to SpawnAction. This
 * refactoring will help isolate Starlark specific implementation without impacting other classes
 * derived from SpawnAction.
 */
public final class StarlarkAction extends SpawnAction {

  /**
   * Constructs a StarlarkAction using direct initialization arguments.
   *
   * <p>All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action
   * @param tools the set of files comprising the tool that does the work (e.g. compiler). This is a
   *     subset of "inputs" and is only used by the WorkerSpawnStrategy
   * @param inputs the set of all files potentially read by this action; must not be subsequently
   *     modified
   * @param outputs the set of all files written by this action; must not be subsequently modified.
   * @param primaryOutput the primary output of this action
   * @param resourceSet the resources consumed by executing this Action
   * @param commandLines the command lines to execute. This includes the main argv vector and any
   *     param file-backed command lines.
   * @param commandLineLimits the command line limits, from the build configuration
   * @param isShellCommand Whether the command line represents a shell command with the given shell
   *     executable. This is used to give better error messages.
   * @param env the action's environment
   * @param executionInfo out-of-band information for scheduling the spawn
   * @param progressMessage the message printed during the progression of the build
   * @param runfilesSupplier {@link RunfilesSupplier}s describing the runfiles for the action
   * @param mnemonic the mnemonic that is reported in the master log
   */
  public StarlarkAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      Artifact primaryOutput,
      ResourceSet resourceSet,
      CommandLines commandLines,
      CommandLineLimits commandLineLimits,
      boolean isShellCommand,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier,
      String mnemonic) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        primaryOutput,
        resourceSet,
        commandLines,
        commandLineLimits,
        isShellCommand,
        env,
        executionInfo,
        progressMessage,
        runfilesSupplier,
        mnemonic,
        /* executeUnconditionally */ false,
        /* extraActionInfoSupplier */ null);
  }

  /** Builder class to construct {@link StarlarkAction} instances. */
  public static class Builder extends SpawnAction.Builder {

    /** Creates a SpawnAction. */
    @Override
    protected SpawnAction createSpawnAction(
        ActionOwner owner,
        NestedSet<Artifact> tools,
        NestedSet<Artifact> inputsAndTools,
        ImmutableList<Artifact> outputs,
        Artifact primaryOutput,
        ResourceSet resourceSet,
        CommandLines commandLines,
        CommandLineLimits commandLineLimits,
        boolean isShellCommand,
        ActionEnvironment env,
        ImmutableMap<String, String> executionInfo,
        CharSequence progressMessage,
        RunfilesSupplier runfilesSupplier,
        String mnemonic) {
      return new StarlarkAction(
          owner,
          tools,
          inputsAndTools,
          outputs,
          primaryOutput,
          resourceSet,
          commandLines,
          commandLineLimits,
          isShellCommand,
          env,
          executionInfo,
          progressMessage,
          runfilesSupplier,
          mnemonic);
    }
  }
}
