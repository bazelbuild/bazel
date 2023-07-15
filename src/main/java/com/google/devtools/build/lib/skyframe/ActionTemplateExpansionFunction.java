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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
  private final BugReporter bugReporter;

  ActionTemplateExpansionFunction(ActionKeyContext actionKeyContext) {
    this(actionKeyContext, BugReporter.defaultInstance());
  }

  @VisibleForTesting
  ActionTemplateExpansionFunction(ActionKeyContext actionKeyContext, BugReporter bugReporter) {
    this.actionKeyContext = actionKeyContext;
    this.bugReporter = bugReporter;
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

    TreeArtifactValue treeArtifactValue =
        (TreeArtifactValue) env.getValue(actionTemplate.getInputTreeArtifact());

    // Input TreeArtifact is not ready yet.
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts = treeArtifactValue.getChildren();
    ImmutableList<ActionAnalysisMetadata> actions;
    try {
      // Expand the action template using the list of expanded input TreeFileArtifacts.
      // TODO(rduan): Add a check to verify the inputs of expanded actions are subsets of inputs
      // of the ActionTemplate.
      actions = generateAndValidateActionsFromTemplate(actionTemplate, inputTreeFileArtifacts, key);
    } catch (ActionExecutionException e) {
      env.getListener()
          .handle(
              Event.error(
                  actionTemplate.getOwner().getLocation(),
                  actionTemplate.describe() + " failed: " + e.getMessage()));
      throw new ActionTemplateExpansionFunctionException(
          new AlreadyReportedActionExecutionException(e));
    }
    try {
      checkActionAndArtifactConflicts(actions, key);
      // It is currently not possible for Starlark actions to create action template actions, so
      // no exceptions here are expected. However, they may be possible in the future.
    } catch (ActionConflictException e) {
      bugReporter.sendBugReport(
          new IllegalStateException("Unexpected action conflict for " + skyKey, e));
      e.reportTo(env.getListener());
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (ArtifactPrefixConflictException e) {
      bugReporter.sendBugReport(
          new IllegalStateException("Unexpected artifact prefix conflict for " + skyKey, e));
      env.getListener().handle(Event.error(e.getMessage()));
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (Actions.ArtifactGeneratedByOtherRuleException e) {
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

  /** Exception thrown by {@link ActionTemplateExpansionFunction}. */
  private static final class ActionTemplateExpansionFunctionException extends SkyFunctionException {
    ActionTemplateExpansionFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ArtifactPrefixConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  private static ImmutableList<ActionAnalysisMetadata> generateAndValidateActionsFromTemplate(
      ActionTemplate<?> actionTemplate,
      ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts,
      ActionTemplateExpansionKey key)
      throws ActionExecutionException {
    Set<Artifact> outputs = actionTemplate.getOutputs();
    for (Artifact output : outputs) {
      Preconditions.checkState(
          output.isTreeArtifact(),
          "%s declares an output which is not a tree artifact: %s",
          actionTemplate,
          output);
    }
    ImmutableList<? extends Action> actions =
        actionTemplate.generateActionsForInputArtifacts(inputTreeFileArtifacts, key);
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
        Preconditions.checkState(
            outputs.contains(output.getParent()),
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
          ArtifactPrefixConflictException,
          InterruptedException,
          Actions.ArtifactGeneratedByOtherRuleException {
    Actions.assignOwnersAndThrowIfConflict(actionKeyContext, actions, key);
    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> artifactPrefixConflictMap =
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
   *     {@link ArtifactPrefixConflictException}.
   */
  private static Map<ActionAnalysisMetadata, ArtifactPrefixConflictException>
      findArtifactPrefixConflicts(Map<Artifact, ActionAnalysisMetadata> generatingActions) {
    return Actions.findArtifactPrefixConflicts(
        new MapBasedImmutableActionGraph(generatingActions),
        generatingActions.keySet(),
        /*strictConflictChecks=*/ true);
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
