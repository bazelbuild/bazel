// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.ActionTemplate.InputFileReader;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.LostArtifacts;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.collect.nestedset.NestedSetOwnerScope;
import com.google.devtools.build.lib.concurrent.MoreFutures;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindException;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The SkyFunction for {@link ActionTemplateExpansionValue}.
 *
 * <p>Given an action template, this function resolves its input TreeArtifact, then expands the
 * action template into a list of actions using the expanded {@link TreeFileArtifact}s under the
 * input TreeArtifact.
 */
public class ActionTemplateExpansionFunction implements SkyFunction {
  private final ActionKeyContext actionKeyContext;
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final ActionRewindStrategy actionRewindStrategy;

  @VisibleForTesting
  ActionTemplateExpansionFunction(
      ActionKeyContext actionKeyContext,
      SkyframeActionExecutor skyframeActionExecutor,
      ActionRewindStrategy actionRewindStrategy) {
    this.actionKeyContext = actionKeyContext;
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.actionRewindStrategy = actionRewindStrategy;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ActionTemplateExpansionFunctionException, InterruptedException {
    ActionTemplateExpansionKey key = (ActionTemplateExpansionKey) skyKey.argument();
    ActionLookupValue value = (ActionLookupValue) env.getValue(key.getActionLookupKey());
    if (value == null) {
      // Because of the phase boundary separating analysis and execution, all needed
      // ActionLookupValues must have already been evaluated, so a missing ActionLookupValue is
      // unexpected. However, we tolerate this case.
      BugReport.sendBugReport(new IllegalStateException("Unexpected absent value for " + key));
      return null;
    }
    ActionTemplate<?> actionTemplate = value.getActionTemplate(key.getActionIndex());

    ImmutableList<TreeArtifact> inputTreeArtifacts = actionTemplate.getInputTreeArtifacts();

    // The values of inputs to the expanded actions are not used by this function. They are
    // requested so that they are built in parallel with the input tree artifacts instead of only
    // after the template has been expanded.
    ImmutableList.Builder<SkyKey> otherInputKeysBuilder = ImmutableList.builder();
    // Following b/143205147, we unwrap the top layer of the NestedSet and evaluate the first layer
    // of the NestedSet as direct Artifact(s) and transitive NestedSet(s).
    ArtifactNestedSetKey.visitDirectDeps(
        actionTemplate.getInputs(),
        leaf -> otherInputKeysBuilder.add(Artifact.key(leaf)),
        otherInputKeysBuilder::add);
    ImmutableList<SkyKey> otherInputKeys = otherInputKeysBuilder.build();

    SkyframeLookupResult result =
        env.getValuesAndExceptions(Iterables.concat(inputTreeArtifacts, otherInputKeys));

    if (env.valuesMissing()) {
      return null;
    }

    var inputMetadata = new ActionInputMap(inputTreeArtifacts.size());
    var inputFileReader = new TemplateInputFileReader(actionTemplate, inputMetadata, result);
    ImmutableList<ActionAnalysisMetadata> actions;
    try {
      ImmutableList.Builder<TreeFileArtifact> inputTreeFileArtifacts = ImmutableList.builder();
      for (TreeArtifact inputTreeArtifact : inputTreeArtifacts) {
        TreeArtifactValue treeArtifactValue =
            (TreeArtifactValue)
                result.getOrThrow(inputTreeArtifact, ActionExecutionException.class);
        // b/507424770#comment10: To handle the case of a wrongly bubbled up exception causing a
        // null value, we return null here so that we don't crash with an NPE.
        if (treeArtifactValue == null) {
          return null;
        }
        inputTreeFileArtifacts.addAll(treeArtifactValue.getChildren());
        ActionInputMapHelper.addToMap(
            inputMetadata, inputTreeArtifact, treeArtifactValue, MetadataConsumerForMetrics.NO_OP);
      }
      // Check other inputs for errors to avoid propagating both a value and an error (b/580921749).
      for (SkyKey depKey : otherInputKeys) {
        if (result.get(depKey) == null) {
          return null;
        }
      }
      // Expand the action template using the list of expanded input TreeFileArtifacts.
      // TODO(rduan): Add a check to verify the inputs of expanded actions are subsets of inputs
      // of the ActionTemplate.
      try (var _ = NestedSetOwnerScope.enter(key)) {
        actions =
            generateAndValidateActionsFromTemplate(
                actionTemplate,
                inputTreeFileArtifacts.build(),
                key,
                inputFileReader,
                env.getListener());
      }
    } catch (ActionExecutionException e) {
      ActionExecutionException failure = e;
      if (inputFileReader.lostInputs != null) {
        // The expansion failed since it read an input that is no longer available. Rewind the
        // action generating it.
        try {
          return actionRewindStrategy
              .prepareRewindPlanForLostTemplateExpansionInputs(
                  key,
                  actionTemplate,
                  ImmutableSet.copyOf(Iterables.concat(inputTreeArtifacts, otherInputKeys)),
                  inputFileReader.lostInputs,
                  inputMetadata,
                  env)
              .toNullIfMissingDependenciesElseReset();
        } catch (ActionRewindException rewindException) {
          failure =
              new ActionExecutionException(
                  rewindException,
                  actionTemplate,
                  /* catastrophe= */ false,
                  rewindException.getDetailedExitCode());
        }
      }
      env.getListener()
          .handle(
              Event.error(
                  actionTemplate.getOwner().getLocation(),
                  actionTemplate.describe() + " failed: " + failure.getMessage()));
      throw new ActionTemplateExpansionFunctionException(
          new AlreadyReportedActionExecutionException(failure));
    } catch (ActionConflictException e) {
      e.reportTo(env.getListener());
      throw new ActionTemplateExpansionFunctionException(e);
    }
    try {
      checkActionAndArtifactConflicts(actions, key);
    } catch (ActionConflictException e) {
      e.reportTo(env.getListener());
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (Actions.ArtifactGeneratedByOtherRuleException
        | Actions.SourceArtifactUsedAsOutputException e) {
      throw new IllegalStateException(
          "Actions generated by template "
              + actionTemplate.describe()
              + " did not all output tree file artifacts belonging to the correct output tree"
              + " artifact + ("
              + skyKey
              + ")",
          e);
    }

    return new ActionTemplateExpansionValue(actions);
  }

  /**
   * Reads the input files of an {@link ActionTemplate} and records the ones that are no longer
   * available.
   */
  private final class TemplateInputFileReader implements InputFileReader {
    private final ActionTemplate<?> actionTemplate;
    // Initially contains the metadata of the files in the input tree artifacts. The metadata of
    // other files is added when they are read.
    private final ActionInputMap inputMetadata;
    private final SkyframeLookupResult inputValues;
    @Nullable private ImmutableSetMultimap<String, ActionInput> lostInputs;

    TemplateInputFileReader(
        ActionTemplate<?> actionTemplate,
        ActionInputMap inputMetadata,
        SkyframeLookupResult inputValues) {
      this.actionTemplate = actionTemplate;
      this.inputMetadata = inputMetadata;
      this.inputValues = inputValues;
    }

    @Override
    public byte[] read(Artifact file) throws IOException, InterruptedException {
      if (!(file instanceof TreeFileArtifact) && inputMetadata.getInputMetadata(file) == null) {
        addMetadataOfDirectInput(file);
      }
      try (SilentCloseable c =
          skyframeActionExecutor
              .getOutputService()
              .getRewoundActionSynchronizer()
              .enterActionTemplateInputRead(file, inputMetadata)) {
        ListenableFuture<Void> prefetch =
            skyframeActionExecutor
                .getActionInputPrefetcher()
                .prefetchFiles(
                    /* action= */ null,
                    /* spawn= */ null,
                    () -> ImmutableList.of(file),
                    inputMetadata,
                    Priority.CRITICAL,
                    Reason.INPUTS);
        try {
          MoreFutures.waitForFutureAndGetWithCheckedException(
              prefetch, /* cancelOnInterrupt= */ true, IOException.class);
        } catch (BulkTransferException e) {
          LostArtifacts lostArtifacts = e.getLostArtifacts(inputMetadata::getInput);
          if (!lostArtifacts.isEmpty()) {
            lostInputs = lostArtifacts.byDigest();
          }
          throw e;
        }
        return FileSystemUtils.readContent(file.getPath());
      }
    }

    private void addMetadataOfDirectInput(Artifact file) throws FileNotFoundException {
      // Only the keys of the direct inputs of the template have been requested. Since the
      // expansion depends on them, it's reevaluated when the contents of the file change and the
      // action generating the file can be rewound if the contents are lost.
      SkyValue value = inputValues.get(Artifact.key(file));
      checkArgument(value != null, "%s is not a direct input of %s", file, actionTemplate);
      if (value instanceof MissingArtifactValue) {
        throw new FileNotFoundException("missing input file");
      }
      ActionInputMapHelper.addToMap(inputMetadata, file, value, MetadataConsumerForMetrics.NO_OP);
    }
  }

  /** Exception thrown by {@link ActionTemplateExpansionFunction}. */
  private static final class ActionTemplateExpansionFunctionException extends SkyFunctionException {
    ActionTemplateExpansionFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ActionExecutionException e) {
      // Reading the contents of inputs can fail for reasons that don't show up in the dependencies,
      // such as remote cache evictions.
      super(e, Transience.TRANSIENT);
    }
  }

  private static ImmutableList<ActionAnalysisMetadata> generateAndValidateActionsFromTemplate(
      ActionTemplate<?> actionTemplate,
      ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
      ActionTemplateExpansionKey key,
      InputFileReader inputFileReader,
      EventHandler eventHandler)
      throws ActionConflictException, ActionExecutionException, InterruptedException {
    Collection<Artifact> outputs = actionTemplate.getOutputs();
    for (Artifact output : outputs) {
      Preconditions.checkState(
          output.isTreeArtifact(),
          "%s declares an output which is not a tree artifact: %s",
          actionTemplate,
          output);
    }
    ImmutableList<? extends Action> actions =
        actionTemplate.generateActionsForInputArtifacts(
            inputTreeFileArtifacts, key, inputFileReader, eventHandler);
    for (Action action : actions) {
      for (Artifact output : action.getOutputs()) {
        Preconditions.checkState(
            output.getArtifactOwner().equals(key),
            "%s generated an action with an output owned by the wrong owner %s not %s (%s)",
            actionTemplate,
            output.getArtifactOwner(),
            key,
            action);
        Preconditions.checkState(
            output.hasParent(),
            "%s generated an action which outputs a non-TreeFileArtifact %s (%s)",
            actionTemplate,
            output,
            action);
        Artifact outputTree =
            output.getParent().isSubTreeArtifact()
                ? output.getParent().getParent()
                : output.getParent();
        Preconditions.checkState(
            outputs.contains(outputTree),
            "%s generated an action with an output %s under an undeclared tree not in %s (%s)",
            actionTemplate,
            output,
            outputs,
            action);
      }
    }
    return ImmutableList.copyOf(actions); // Just a cast, no copy performed.
  }

  private void checkActionAndArtifactConflicts(
      ImmutableList<ActionAnalysisMetadata> actions, ActionTemplateExpansionKey key)
      throws ActionConflictException,
          InterruptedException,
          Actions.ArtifactGeneratedByOtherRuleException,
          Actions.SourceArtifactUsedAsOutputException {
    Actions.assignOwnersAndThrowIfConflict(actionKeyContext, actions, key);
    Map<ActionAnalysisMetadata, ActionConflictException> artifactPrefixConflictMap =
        findArtifactPrefixConflicts(getMapForConsistencyCheck(actions));

    if (!artifactPrefixConflictMap.isEmpty()) {
      throw artifactPrefixConflictMap.values().iterator().next();
    }
  }

  private static ImmutableMap<Artifact, ActionAnalysisMetadata> getMapForConsistencyCheck(
      List<? extends ActionAnalysisMetadata> actions) {
    if (actions.isEmpty()) {
      return ImmutableMap.of();
    }
    HashMap<Artifact, ActionAnalysisMetadata> result =
        Maps.newHashMapWithExpectedSize(actions.size() * actions.get(0).getOutputs().size());
    for (ActionAnalysisMetadata action : actions) {
      for (Artifact output : action.getOutputs()) {
        result.put(output, action);
      }
    }
    return ImmutableMap.copyOf(result);
  }

  /**
   * Finds Artifact prefix conflicts between generated artifacts. An artifact prefix conflict
   * happens if one action generates an artifact whose path is a prefix of another artifact's path.
   * Those two artifacts cannot exist simultaneously in the output tree.
   *
   * @param generatingActions a map between generated artifacts and their associated generating
   *     actions.
   * @return a map between actions that generated the conflicting artifacts and their associated
   *     {@link ActionConflictException}.
   */
  private static Map<ActionAnalysisMetadata, ActionConflictException> findArtifactPrefixConflicts(
      Map<Artifact, ActionAnalysisMetadata> generatingActions) {
    return Actions.findArtifactPrefixConflicts(
        new MapBasedImmutableActionGraph(generatingActions), generatingActions.keySet());
  }

  private static class MapBasedImmutableActionGraph implements ActionGraph {
    private final Map<Artifact, ActionAnalysisMetadata> generatingActions;

    MapBasedImmutableActionGraph(Map<Artifact, ActionAnalysisMetadata> generatingActions) {
      this.generatingActions = ImmutableMap.copyOf(generatingActions);
    }

    @Nullable
    @Override
    public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
      return generatingActions.get(artifact);
    }
  }
}
