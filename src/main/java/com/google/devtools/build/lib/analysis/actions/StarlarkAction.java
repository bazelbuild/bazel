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
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheAwareAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.StarlarkAction.Code;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/** A Starlark specific SpawnAction. */
public final class StarlarkAction extends SpawnAction implements ActionCacheAwareAction {

  // All the inputs of the Starlark action including those listed in the unused inputs and
  // execluding the shadowed action inputs
  private final NestedSet<Artifact> allStarlarkActionInputs;

  private final Optional<Artifact> unusedInputsList;
  private final Optional<Action> shadowedAction;

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
   * @param shadowedAction the action to use its inputs and environment during execution
   */
  public StarlarkAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
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
      Optional<Artifact> unusedInputsList,
      Optional<Action> shadowedAction) {
    super(
        owner,
        tools,
        shadowedAction.isPresent()
            ? createInputs(shadowedAction.get().getInputs(), inputs)
            : inputs,
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

    this.allStarlarkActionInputs = inputs;
    this.unusedInputsList = unusedInputsList;
    this.shadowedAction = shadowedAction;
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
    return unusedInputsList.isPresent()
        || (shadowedAction.isPresent() && shadowedAction.get().discoversInputs());
  }

  @Override
  public NestedSet<Artifact> getAllowedDerivedInputs() {
    if (shadowedAction.isPresent()) {
      return createInputs(shadowedAction.get().getAllowedDerivedInputs(), getInputs());
    }
    return getInputs();
  }

  /**
   * This method returns null when a required SkyValue is missing and a Skyframe restart is
   * required.
   */
  @Nullable
  @Override
  public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    // If the Starlark action shadows another action and the shadowed action discovers its inputs,
    // we get the shadowed action's discovered inputs and append it to the Starlark action inputs.
    if (shadowedAction.isPresent() && shadowedAction.get().discoversInputs()) {
      Action shadowedActionObj = shadowedAction.get();

      NestedSet<Artifact> oldInputs = getInputs();
      NestedSet<Artifact> inputFilesForExtraAction =
          shadowedActionObj.getInputFilesForExtraAction(actionExecutionContext);
      if (inputFilesForExtraAction == null) {
        return null;
      }
      updateInputs(
          createInputs(
              shadowedActionObj.getInputs(), inputFilesForExtraAction, allStarlarkActionInputs));
      return NestedSetBuilder.wrap(
          Order.STABLE_ORDER, Sets.<Artifact>difference(getInputs().toSet(), oldInputs.toSet()));
    }
    // Otherwise, we need to "re-discover" all the original inputs: the unused ones that were
    // removed might now be needed.
    updateInputs(allStarlarkActionInputs);
    return allStarlarkActionInputs;
  }

  private InputStream getUnusedInputListInputStream(
      ActionExecutionContext actionExecutionContext, List<SpawnResult> spawnResults)
      throws IOException, ExecException {

    // Check if the file is in-memory.
    // Note: SpawnActionContext guarantees that the first list entry exists and corresponds to the
    // executed spawn.
    Artifact unusedInputsListArtifact = unusedInputsList.get();
    InputStream inputStream = spawnResults.get(0).getInMemoryOutput(unusedInputsListArtifact);
    if (inputStream != null) {
      return inputStream;
    }
    // Fallback to reading from disk.
    try {
      return actionExecutionContext
          .getPathResolver()
          .toPath(unusedInputsListArtifact)
          .getInputStream();
    } catch (FileNotFoundException e) {
      String message =
          "Action did not create expected output file listing unused inputs: "
              + unusedInputsListArtifact.getExecPathString();
      throw new UserExecException(
          e, createFailureDetail(message, Code.UNUSED_INPUT_LIST_FILE_NOT_FOUND));
    }
  }

  @Override
  protected void afterExecute(
      ActionExecutionContext actionExecutionContext, List<SpawnResult> spawnResults)
      throws ExecException {
    if (!unusedInputsList.isPresent()) {
      return;
    }

    // Get all the action's inputs after execution which will include the shadowed action
    // discovered inputs
    NestedSet<Artifact> allInputs = getInputs();
    Map<String, Artifact> usedInputs = new HashMap<>();
    for (Artifact input : allInputs.toList()) {
      usedInputs.put(input.getExecPathString(), input);
    }
    try (BufferedReader br =
        new BufferedReader(
            new InputStreamReader(
                getUnusedInputListInputStream(actionExecutionContext, spawnResults), UTF_8))) {
      String line;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) {
          continue;
        }
        usedInputs.remove(line);
      }
    } catch (IOException e) {
      throw new EnvironmentalExecException(
          e,
          createFailureDetail("Unused inputs read failure", Code.UNUSED_INPUT_LIST_READ_FAILURE));
    }
    updateInputs(NestedSetBuilder.wrap(Order.STABLE_ORDER, usedInputs.values()));
  }

  @Override
  Spawn getSpawnForExtraAction() throws CommandLineExpansionException, InterruptedException {
    if (shadowedAction.isPresent()) {
      return getSpawn(createInputs(shadowedAction.get().getInputs(), allStarlarkActionInputs));
    }
    return getSpawn(allStarlarkActionInputs);
  }

  /**
   * This method returns null when a required SkyValue is missing and a Skyframe restart is
   * required.
   */
  @Nullable
  @Override
  public NestedSet<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    if (shadowedAction.isPresent()) {
      NestedSet<Artifact> inputFilesForExtraAction =
          shadowedAction.get().getInputFilesForExtraAction(actionExecutionContext);
      if (inputFilesForExtraAction == null) {
        return null;
      }
      return createInputs(
          shadowedAction.get().getInputFilesForExtraAction(actionExecutionContext),
          allStarlarkActionInputs);
    }
    return allStarlarkActionInputs;
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setStarlarkAction(FailureDetails.StarlarkAction.newBuilder().setCode(detailedCode))
        .build();
  }

  private static NestedSet<Artifact> createInputs(NestedSet<Artifact>... inputsLists) {
    NestedSetBuilder<Artifact> nestedSetBuilder = new NestedSetBuilder<>(Order.STABLE_ORDER);
    for (NestedSet<Artifact> inputs : inputsLists) {
      nestedSetBuilder.addTransitive(inputs);
    }
    return nestedSetBuilder.build();
  }

  /**
   * StarlarkAction can contain `unused_input_list`, which rely on the action cache entry's file
   * list to determine the list of inputs for a subsequent run, taking into account
   * unused_input_list. Hence we need to store the inputs' execPaths in the action cache. The
   * StarlarkAction inputs' execPaths should also be stored in the action cache if it shadows
   * another action that discovers its inputs to avoid re-running input discovery after a shutdown.
   */
  @Override
  public boolean storeInputsExecPathsInActionCache() {
    return unusedInputsList.isPresent()
        || (shadowedAction.isPresent() && shadowedAction.get().discoversInputs());
  }

  /**
   * Return a spawn that is representative of the command that this Action will execute in the given
   * client environment.
   *
   * <p>Overriding this method to add the environment of the shadowed action, if any, to the
   * execution spawn.
   */
  @Override
  public Spawn getSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException, InterruptedException {
    return getSpawn(
        actionExecutionContext.getArtifactExpander(),
        getEffectiveEnvironment(actionExecutionContext.getClientEnv()),
        /*envResolved=*/ true,
        actionExecutionContext.getTopLevelFilesets());
  }

  @Override
  public ImmutableMap<String, String> getEffectiveEnvironment(Map<String, String> clientEnv)
      throws CommandLineExpansionException {
    Map<String, String> environment = Maps.newLinkedHashMapWithExpectedSize(env.size());

    if (shadowedAction.isPresent()) {
      // Put all the variables of the shadowed action's environment
      environment.putAll(shadowedAction.get().getEffectiveEnvironment(clientEnv));
    }

    // This order guarantees that the Starlark action can overwrite any variable in its shadowed
    // action environment with a new value.
    env.resolve(environment, clientEnv);
    return ImmutableMap.copyOf(environment);
  }

  /** Builder class to construct {@link StarlarkAction} instances. */
  public static class Builder extends SpawnAction.Builder {

    private Optional<Artifact> unusedInputsList = Optional.empty();
    private Optional<Action> shadowedAction = Optional.empty();

    public Builder setUnusedInputsList(Optional<Artifact> unusedInputsList) {
      this.unusedInputsList = unusedInputsList;
      return this;
    }

    public Builder setShadowedAction(Optional<Action> shadowedAction) {
      this.shadowedAction = shadowedAction;
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
        @Nullable BuildConfiguration configuration,
        ImmutableMap<String, String> executionInfo,
        CharSequence progressMessage,
        RunfilesSupplier runfilesSupplier,
        String mnemonic) {
      if (unusedInputsList.isPresent()) {
        // Always download unused_inputs_list file from remote cache.
        executionInfo =
            ImmutableMap.<String, String>builderWithExpectedSize(executionInfo.size() + 1)
                .putAll(executionInfo)
                .put(
                    ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS,
                    unusedInputsList.get().getExecPathString())
                .build();
      }
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
          unusedInputsList,
          shadowedAction);
    }
  }
}
