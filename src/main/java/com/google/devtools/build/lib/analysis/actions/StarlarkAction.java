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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** A Starlark specific SpawnAction. */
public final class StarlarkAction extends SpawnAction {

  private final Optional<Artifact> unusedInputsList;
  private final Iterable<Artifact> allInputs;

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
   * @param unusedInputsList file containing the list of inputs that were not used by the action.
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
      String mnemonic,
      Optional<Artifact> unusedInputsList) {
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
        /* extraActionInfoSupplier */ null,
        /* resultConsumer */ null);
    this.allInputs = inputs;
    this.unusedInputsList = unusedInputsList;
  }

  @VisibleForTesting
  public Optional<Artifact> getUnusedInputsList() {
    return unusedInputsList;
  }

  @Override
  public boolean isShareable() {
    return !unusedInputsList.isPresent();
  }

  @Override
  public boolean discoversInputs() {
    return unusedInputsList.isPresent();
  }

  @Override
  public Iterable<Artifact> getAllowedDerivedInputs() {
    return getInputs();
  }

  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    // We need to "re-discover" all the original inputs: the unused ones that were removed
    // might now be needed.
    updateInputs(allInputs);
    return allInputs;
  }

  @Override
  protected void afterExecute(ActionExecutionContext actionExecutionContext) throws IOException {
    if (!unusedInputsList.isPresent()) {
      return;
    }
    Map<String, Artifact> usedInputs = new HashMap<>();
    for (Artifact input : allInputs) {
      usedInputs.put(input.getExecPathString(), input);
    }
    try (BufferedReader br =
        new BufferedReader(
            new InputStreamReader(unusedInputsList.get().getPath().getInputStream(), UTF_8))) {
      String line;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) {
          continue;
        }
        usedInputs.remove(line);
      }
    }
    updateInputs(usedInputs.values());
  }

  /** Builder class to construct {@link StarlarkAction} instances. */
  public static class Builder extends SpawnAction.Builder {

    private Optional<Artifact> unusedInputsList = Optional.empty();

    public Builder setUnusedInputsList(Optional<Artifact> unusedInputsList) {
      this.unusedInputsList = unusedInputsList;
      return this;
    }

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
          mnemonic,
          unusedInputsList);
    }
  }
}
