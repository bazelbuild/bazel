// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.extra;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineLimits;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Action used by extra_action rules to create an action that shadows an existing action. Runs a
 * command-line using {@link com.google.devtools.build.lib.actions.SpawnStrategy} for executions.
 */
@AutoCodec
public final class ExtraAction extends SpawnAction {
  private final Action shadowedAction;
  private final boolean createDummyOutput;
  private final NestedSet<Artifact> extraActionInputs;
  private boolean inputsDiscovered = false;

  ExtraAction(
      ActionOwner owner,
      NestedSet<Artifact> extraActionInputs,
      Collection<Artifact.DerivedArtifact> outputs,
      Action shadowedAction,
      boolean createDummyOutput,
      CommandLine argv,
      ActionEnvironment env,
      Map<String, String> executionInfo,
      CharSequence progressMessage,
      String mnemonic) {
    super(
        owner,
        /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        createInputs(
            shadowedAction.getInputs(),
            /* inputFilesForExtraAction= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            extraActionInputs),
        outputs,
        AbstractAction.DEFAULT_RESOURCE_SET,
        CommandLines.of(argv),
        env,
        ImmutableMap.copyOf(executionInfo),
        progressMessage,
        mnemonic,
        OutputPathsMode.OFF);
    this.shadowedAction = shadowedAction;
    this.createDummyOutput = createDummyOutput;

    this.extraActionInputs = extraActionInputs;
    if (createDummyOutput) {
      // Expecting just a single dummy file in the outputs.
      Preconditions.checkArgument(outputs.size() == 1, outputs);
    }
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  ExtraAction(
      ActionOwner owner,
      NestedSet<Artifact> extraActionInputs,
      Object rawOutputs,
      Action shadowedAction,
      boolean createDummyOutput,
      CommandLines commandLines,
      ActionEnvironment environment,
      ImmutableSortedMap<String, String> sortedExecutionInfo,
      CharSequence progressMessage,
      String mnemonic) {
    super(
        owner,
        /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        createInputs(
            shadowedAction.getInputs(),
            /* inputFilesForExtraAction= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            extraActionInputs),
        rawOutputs,
        AbstractAction.DEFAULT_RESOURCE_SET,
        commandLines,
        environment,
        sortedExecutionInfo,
        progressMessage,
        mnemonic,
        OutputPathsMode.OFF);
    this.shadowedAction = shadowedAction;
    this.createDummyOutput = createDummyOutput;
    this.extraActionInputs = extraActionInputs;
  }

  @Override
  protected CommandLineLimits getCommandLineLimits() {
    return CommandLineLimits.UNLIMITED;
  }

  @Override
  public boolean discoversInputs() {
    return shadowedAction.discoversInputs();
  }

  @Override
  protected boolean inputsDiscovered() {
    return inputsDiscovered;
  }

  @Override
  protected void setInputsDiscovered(boolean inputsDiscovered) {
    this.inputsDiscovered = inputsDiscovered;
  }

  /**
   * This method returns null when a required SkyValue is missing and a Skyframe restart is
   * required.
   */
  @Nullable
  @Override
  public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Preconditions.checkState(discoversInputs(), this);

    // We need to update our inputs to take account of any additional
    // inputs the shadowed action may need to do its work.
    NestedSet<Artifact> oldInputs = getInputs();
    NestedSet<Artifact> inputFilesForExtraAction =
        shadowedAction.getInputFilesForExtraAction(actionExecutionContext);
    if (inputFilesForExtraAction == null) {
      return null;
    }
    updateInputs(
        createInputs(shadowedAction.getInputs(), inputFilesForExtraAction, extraActionInputs));
    return NestedSetBuilder.wrap(
        Order.STABLE_ORDER, Sets.difference(getInputs().toSet(), oldInputs.toSet()));
  }

  @Override
  public NestedSet<Artifact> getOriginalInputs() {
    return shadowedAction.getOriginalInputs();
  }

  @Override
  public NestedSet<Artifact> getSchedulingDependencies() {
    return shadowedAction.getSchedulingDependencies();
  }

  private static NestedSet<Artifact> createInputs(
      NestedSet<Artifact> shadowedActionInputs,
      NestedSet<Artifact> inputFilesForExtraAction,
      NestedSet<Artifact> extraActionInputs) {
    return NestedSet.<Artifact>builder(Order.STABLE_ORDER)
        .addTransitive(shadowedActionInputs)
        .addTransitive(inputFilesForExtraAction)
        .addTransitive(extraActionInputs)
        .build();
  }

  @Override
  public NestedSet<Artifact> getAllowedDerivedInputs() {
    return shadowedAction.getAllowedDerivedInputs();
  }

  @Override
  public Spawn getSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException, InterruptedException {
    return super.getSpawn(actionExecutionContext, /* reportOutputs= */ !createDummyOutput);
  }

  @Override
  protected void afterExecute(
      ActionExecutionContext actionExecutionContext,
      List<SpawnResult> spawnResults,
      PathMapper pathMapper)
      throws ExecException {
    // PHASE 3: create dummy output.
    // If the user didn't specify output, we need to create dummy output
    // to make blaze schedule this action.
    if (createDummyOutput) {
      for (Artifact output : getOutputs()) {
        try {
          FileSystemUtils.touchFile(actionExecutionContext.getInputPath(output));
        } catch (IOException e) {
          throw new EnvironmentalExecException(e, Code.EXTRA_ACTION_OUTPUT_CREATION_FAILURE);
        }
      }
    }
  }

  /** Returns the action this extra action is 'shadowing'. */
  public Action getShadowedAction() {
    return shadowedAction;
  }
}
