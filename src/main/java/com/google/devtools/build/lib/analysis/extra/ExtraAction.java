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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Action used by extra_action rules to create an action that shadows an existing action. Runs a
 * command-line using {@link SpawnActionContext} for executions.
 */
public final class ExtraAction extends SpawnAction {
  private final Action shadowedAction;
  private final boolean createDummyOutput;
  private final NestedSet<Artifact> extraActionInputs;

  /**
   * A long way to say (ExtraAction xa) -> xa.getShadowedAction().
   */
  public static final Function<ExtraAction, Action> GET_SHADOWED_ACTION =
      new Function<ExtraAction, Action>() {
        @Nullable
        @Override
        public Action apply(@Nullable ExtraAction extraAction) {
          return extraAction != null ? extraAction.getShadowedAction() : null;
        }
      };

  ExtraAction(
      NestedSet<Artifact> extraActionInputs,
      RunfilesSupplier runfilesSupplier,
      Collection<Artifact.DerivedArtifact> outputs,
      Action shadowedAction,
      boolean createDummyOutput,
      CommandLine argv,
      ActionEnvironment env,
      Map<String, String> executionInfo,
      CharSequence progressMessage,
      String mnemonic) {
    super(
        shadowedAction.getOwner(),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        createInputs(
            shadowedAction.getInputs(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            extraActionInputs),
        outputs,
        Iterables.getFirst(outputs, null),
        AbstractAction.DEFAULT_RESOURCE_SET,
        CommandLines.of(argv),
        CommandLineLimits.UNLIMITED,
        false,
        env,
        ImmutableMap.copyOf(executionInfo),
        progressMessage,
        CompositeRunfilesSupplier.of(shadowedAction.getRunfilesSupplier(), runfilesSupplier),
        mnemonic,
        false,
        null,
        null);
    this.shadowedAction = shadowedAction;
    this.createDummyOutput = createDummyOutput;

    this.extraActionInputs = extraActionInputs;
    if (createDummyOutput) {
      // Expecting just a single dummy file in the outputs.
      Preconditions.checkArgument(outputs.size() == 1, outputs);
    }
  }

  @Override
  public boolean discoversInputs() {
    return shadowedAction.discoversInputs();
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
    // We depend on the outputs of actions doing input discovery and they should know their inputs
    // after having been executed
    Preconditions.checkState(shadowedAction.inputsDiscovered());

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
        Order.STABLE_ORDER, Sets.<Artifact>difference(getInputs().toSet(), oldInputs.toSet()));
  }

  private static NestedSet<Artifact> createInputs(
      NestedSet<Artifact> shadowedActionInputs,
      NestedSet<Artifact> inputFilesForExtraAction,
      NestedSet<Artifact> extraActionInputs) {
    return new NestedSetBuilder<Artifact>(Order.STABLE_ORDER)
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
  protected void afterExecute(
      ActionExecutionContext actionExecutionContext, List<SpawnResult> spawnResults)
      throws IOException {
    // PHASE 3: create dummy output.
    // If the user didn't specify output, we need to create dummy output
    // to make blaze schedule this action.
    if (createDummyOutput) {
      for (Artifact output : getOutputs()) {
        FileSystemUtils.touchFile(actionExecutionContext.getInputPath(output));
      }
    }
  }

  /**
   * Returns the action this extra action is 'shadowing'.
   */
  public Action getShadowedAction() {
    return shadowedAction;
  }
}
