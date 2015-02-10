// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.extra;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.DelegateSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Action used by extra_action rules to create an action that shadows an existing action. Runs a
 * command-line using {@link SpawnActionContext} for executions.
 */
public final class ExtraAction extends SpawnAction {
  private final Action shadowedAction;
  private final boolean createDummyOutput;
  private final Artifact extraActionInfoFile;
  private final ImmutableMap<PathFragment, Artifact> runfilesManifests;
  private final ImmutableSet<Artifact> extraActionInputs;
  private boolean inputsKnown;

  public ExtraAction(ActionOwner owner,
      ImmutableSet<Artifact> extraActionInputs,
      Map<PathFragment, Artifact> runfilesManifests,
      Artifact extraActionInfoFile,
      Collection<Artifact> outputs,
      Action shadowedAction,
      boolean createDummyOutput,
      CommandLine argv,
      Map<String, String> environment,
      String progressMessage,
      String mnemonic) {
    super(owner,
        createInputs(shadowedAction.getInputs(), extraActionInputs),
        outputs,
        AbstractAction.DEFAULT_RESOURCE_SET,
        argv,
        ImmutableMap.copyOf(environment),
        ImmutableMap.<String, String>of(),
        progressMessage,
        getManifests(shadowedAction),
        mnemonic,
        null);
    this.extraActionInfoFile = extraActionInfoFile;
    this.shadowedAction = shadowedAction;
    this.runfilesManifests = ImmutableMap.copyOf(runfilesManifests);
    this.createDummyOutput = createDummyOutput;

    this.extraActionInputs = extraActionInputs;
    inputsKnown = shadowedAction.inputsKnown();
    if (createDummyOutput) {
      // extra action file & dummy file
      Preconditions.checkArgument(outputs.size() == 2);
    }
  }

  private static ImmutableMap<PathFragment, Artifact> getManifests(Action shadowedAction) {
    // If the shadowed action is a SpawnAction, then we also add the input manifests to this
    // action's input manifests.
    // TODO(bazel-team): Also handle other action classes correctly.
    if (shadowedAction instanceof SpawnAction) {
      return ((SpawnAction) shadowedAction).getInputManifests();
    }
    return ImmutableMap.of();
  }

  @Override
  public boolean discoversInputs() {
    return shadowedAction.discoversInputs();
  }

  @Override
  public void discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Preconditions.checkState(discoversInputs(), this);
    if (getContext(actionExecutionContext.getExecutor()).isRemotable(getMnemonic(),
        isRemotable())) {
      // If we're running remotely, we need to update our inputs to take account of any additional
      // inputs the shadowed action may need to do its work.
      if (shadowedAction.discoversInputs() && shadowedAction instanceof AbstractAction) {
        updateInputs(
            ((AbstractAction) shadowedAction).getInputFilesForExtraAction(actionExecutionContext));
      }
    }
  }

  @Override
  public boolean inputsKnown() {
    return inputsKnown;
  }

  private static NestedSet<Artifact> createInputs(
      Iterable<Artifact> shadowedActionInputs, ImmutableSet<Artifact> extraActionInputs) {
    NestedSetBuilder<Artifact> result = new NestedSetBuilder<>(Order.STABLE_ORDER);
    if (shadowedActionInputs instanceof NestedSet) {
      result.addTransitive((NestedSet<Artifact>) shadowedActionInputs);
    } else {
      result.addAll(shadowedActionInputs);
    }
    return result.addAll(extraActionInputs).build();
  }

  private void updateInputs(Iterable<Artifact> shadowedActionInputs) {
    synchronized (this) {
      setInputs(createInputs(shadowedActionInputs, extraActionInputs));
      inputsKnown = true;
    }
  }

  @Override
  public void updateInputsFromCache(ArtifactResolver artifactResolver,
      Collection<PathFragment> inputPaths) {
    // We update the inputs directly from the shadowed action.
    Set<PathFragment> extraActionPathFragments =
        ImmutableSet.copyOf(Artifact.asPathFragments(extraActionInputs));
    shadowedAction.updateInputsFromCache(artifactResolver,
        Collections2.filter(inputPaths, Predicates.in(extraActionPathFragments)));
    Preconditions.checkState(shadowedAction.inputsKnown(), "%s %s", this, shadowedAction);
    updateInputs(shadowedAction.getInputs());
  }

  /**
   * @InheritDoc
   *
   * This method calls in to {@link AbstractAction#getInputFilesForExtraAction} and
   * {@link Action#getExtraActionInfo} of the action being shadowed from the thread executing this
   * ExtraAction. It assumes these methods are safe to call from a different thread than the thread
   * responsible for the execution of the action being shadowed.
   */
  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    // PHASE 1: generate .xa file containing protocol buffer describing
    // the action being shadowed

    // We call the getExtraActionInfo command only at execution time
    // so actions can store information only known at execution time into the
    // protocol buffer.
    ExtraActionInfo info = shadowedAction.getExtraActionInfo().build();
    try (OutputStream out = extraActionInfoFile.getPath().getOutputStream()) {
      info.writeTo(out);
    } catch (IOException e) {
      throw new ActionExecutionException(e.getMessage(), e, this, false);
    }
    Executor executor = actionExecutionContext.getExecutor();

    // PHASE 2: execution of extra_action.

    if (getContext(executor).isRemotable(getMnemonic(), isRemotable())) {
      try {
        getContext(executor).exec(getExtraActionSpawn(), actionExecutionContext);
      } catch (ExecException e) {
        throw e.toActionExecutionException(this);
      }
    } else {
      super.execute(actionExecutionContext);
    }

    // PHASE 3: create dummy output.
    // If the user didn't specify output, we need to create dummy output
    // to make blaze schedule this action.
    if (createDummyOutput) {
      for (Artifact output : getOutputs()) {
        try {
          FileSystemUtils.touchFile(output.getPath());
        } catch (IOException e) {
          throw new ActionExecutionException(e.getMessage(), e, this, false);
        }
      }
    }
    synchronized (this) {
      inputsKnown = true;
    }
  }

  /**
   * The spawn command for ExtraAction needs to be slightly modified from
   * regular SpawnActions:
   * -the extraActionInfo file needs to be added to the list of inputs.
   * -the extraActionInfo file that is an output file of this task is created
   * before the SpawnAction so should not be listed as one of its outputs.
   */
  // TODO(bazel-team): Add more tests that execute this code path!
  private Spawn getExtraActionSpawn() {
    final Spawn base = super.getSpawn();
    return new DelegateSpawn(base) {
      @Override public Iterable<? extends ActionInput> getInputFiles() {
        return Iterables.concat(base.getInputFiles(), ImmutableSet.of(extraActionInfoFile));
      }

      @Override public List<? extends ActionInput> getOutputFiles() {
        return Lists.newArrayList(
            Iterables.filter(getOutputs(), new Predicate<Artifact>() {
              @Override
              public boolean apply(Artifact item) {
                return item != extraActionInfoFile;
              }
            }));
      }

      @Override public ImmutableMap<PathFragment, Artifact> getRunfilesManifests() {
        ImmutableMap.Builder<PathFragment, Artifact> builder = ImmutableMap.builder();
        builder.putAll(super.getRunfilesManifests());
        builder.putAll(runfilesManifests);
        return builder.build();
      }

      @Override public String getMnemonic() { return ExtraAction.this.getMnemonic(); }
    };
  }

  /**
   * Returns the action this extra action is 'shadowing'.
   */
  public Action getShadowedAction() {
    return shadowedAction;
  }
}
