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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Action used by extra_action rules to create an action that shadows an existing action. Runs a
 * command-line using {@link SpawnActionContext} for executions.
 */
public final class ExtraAction extends SpawnAction {
  private final Action shadowedAction;
  private final boolean createDummyOutput;
  private final ImmutableSet<Artifact> extraActionInputs;

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
      ImmutableSet<Artifact> extraActionInputs,
      RunfilesSupplier runfilesSupplier,
      Collection<Artifact> outputs,
      Action shadowedAction,
      boolean createDummyOutput,
      CommandLine argv,
      ActionEnvironment env,
      Map<String, String> executionInfo,
      CharSequence progressMessage,
      String mnemonic) {
    super(
        shadowedAction.getOwner(),
        ImmutableList.<Artifact>of(),
        createInputs(shadowedAction.getInputs(), ImmutableList.<Artifact>of(), extraActionInputs),
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

  @Nullable
  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Preconditions.checkState(discoversInputs(), this);
    // We depend on the outputs of actions doing input discovery and they should know their inputs
    // after having been executed
    Preconditions.checkState(shadowedAction.inputsDiscovered());

    // We need to update our inputs to take account of any additional
    // inputs the shadowed action may need to do its work.
    Iterable<Artifact> oldInputs = getInputs();
    updateInputs(
        createInputs(
            shadowedAction.getInputs(),
            shadowedAction.getInputFilesForExtraAction(actionExecutionContext),
            extraActionInputs));
    return Sets.<Artifact>difference(
        ImmutableSet.<Artifact>copyOf(getInputs()), ImmutableSet.<Artifact>copyOf(oldInputs));
  }

  private static NestedSet<Artifact> createInputs(
      Iterable<Artifact> shadowedActionInputs,
      Iterable<Artifact> inputFilesForExtraAction,
      ImmutableSet<Artifact> extraActionInputs) {
    NestedSetBuilder<Artifact> result = new NestedSetBuilder<>(Order.STABLE_ORDER);
    for (Iterable<Artifact> inputSet : ImmutableList.of(
        shadowedActionInputs, inputFilesForExtraAction)) {
      if (inputSet instanceof NestedSet) {
        result.addTransitive((NestedSet<Artifact>) inputSet);
      } else {
        result.addAll(inputSet);
      }
    }
    return result.addAll(extraActionInputs).build();
  }

  @Override
  public Iterable<Artifact> getAllowedDerivedInputs() {
    return shadowedAction.getAllowedDerivedInputs();
  }

  /**
   * @InheritDoc
   *
   * <p>This method calls in to {@link AbstractAction#getInputFilesForExtraAction} and {@link
   * Action#getExtraActionInfo} of the action being shadowed from the thread executing this
   * ExtraAction. It assumes these methods are safe to call from a different thread than the thread
   * responsible for the execution of the action being shadowed.
   */
  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    // PHASE 2: execution of extra_action.
    ActionResult actionResult = super.execute(actionExecutionContext);

    // PHASE 3: create dummy output.
    // If the user didn't specify output, we need to create dummy output
    // to make blaze schedule this action.
    if (createDummyOutput) {
      for (Artifact output : getOutputs()) {
        try {
          FileSystemUtils.touchFile(actionExecutionContext.getInputPath(output));
        } catch (IOException e) {
          throw new ActionExecutionException(e.getMessage(), e, this, false);
        }
      }
    }

    return actionResult;
  }

  /**
   * Returns the action this extra action is 'shadowing'.
   */
  public Action getShadowedAction() {
    return shadowedAction;
  }
}
