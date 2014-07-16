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

package com.google.devtools.build.lib.view.extra;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DelegateSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Action used by extra_action rules to create an action that shadows an existing action. Runs a
 * command-line using {@link SpawnActionContext} for executions.
 */
public final class ExtraAction extends SpawnAction {
  private final Action shadowedAction;
  private final boolean createDummyOutput;
  private final Artifact extraActionInfoFile;
  private final ImmutableMap<PathFragment, Artifact> runfilesManifests;

  public ExtraAction(ActionOwner owner,
      Iterable<Artifact> inputs,
      Map<PathFragment, Artifact> runfilesManifests,
      Artifact extraActionInfoFile,
      Collection<Artifact> outputs,
      Action shadowedAction,
      BuildConfiguration configuration,
      boolean createDummyOutput,
      CommandLine argv,
      Map<String, String> environment,
      String progressMessage,
      String mnemonic) {
    super(owner, inputs, outputs, configuration, AbstractAction.DEFAULT_RESOURCE_SET,
        argv, environment, progressMessage, mnemonic);
    this.extraActionInfoFile = extraActionInfoFile;
    this.shadowedAction = shadowedAction;
    this.runfilesManifests = ImmutableMap.copyOf(runfilesManifests);
    this.createDummyOutput = createDummyOutput;

    if (createDummyOutput) {
      // extra action file & dummy file
      Preconditions.checkArgument(outputs.size() == 2);
    }
  }

  /**
   * @InheritDoc
   *
   * This method calls in to {@link
   * AbstractAction#getAdditionalFilesForExtraAction} and
   * {@link Action#getExtraActionInfo} of the action being shadowed from the
   * thread executing this ExtraAction. It assumes these methods are safe to
   * call from a differentthread than the thread responsible for the execution
   * of the action being shadowed.
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

    // If we're running remotely, we need to ask the original action for any additional
    // files are required to do its work as we need to mirror that behavior.
    if (getContext(executor).isRemotable(getMnemonic(), isRemotable())) {
      List<String> extraFiles = new ArrayList<>();
      if (shadowedAction instanceof AbstractAction) {
        AbstractAction abstractShadowedAction = (AbstractAction) shadowedAction;
        extraFiles.addAll(abstractShadowedAction.getAdditionalFilesForExtraAction(
            actionExecutionContext, this));
      }
      extraFiles.add(extraActionInfoFile.getExecPathString());
      try {
        getContext(executor).exec(getSpawn(extraFiles), actionExecutionContext);
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
  }

  /**
   * The spawn command for ExtraAction needs to be slightly modified from
   * regular SpawnActions:
   * -the extraActionInfo file as well as any other additional file required for
   * the action being shadowed need to be added to the list of inputs (e.g.
   * c++ header files).
   * -the extraActionInfo file that is an output file of this task is created
   * before the SpawnAction so should not be listed as one of its outputs.
   */
  // TODO(bazel-team): Add more tests that execute this code path!
  private Spawn getSpawn(final List<String> extraInputFiles) {
    final Spawn base = super.getSpawn();
    return new DelegateSpawn(base) {
      @Override public Iterable<? extends ActionInput> getInputFiles() {
        return Iterables.concat(base.getInputFiles(), ActionInputHelper.fromPaths(extraInputFiles));
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
      @Override public Artifact getPrimaryInput() { return ExtraAction.this.getPrimaryInput(); }
    };
  }

  /**
   * Returns the action this extra action is 'shadowing'.
   */
  public Action getShadowedAction() {
    return shadowedAction;
  }
}
