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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.ActionTemplate.ActionTemplateExpansionException;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.ArtifactSkyKey;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
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

  ActionTemplateExpansionFunction(ActionKeyContext actionKeyContext) {
    this.actionKeyContext = actionKeyContext;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ActionTemplateExpansionFunctionException, InterruptedException {
    ActionTemplateExpansionKey key = (ActionTemplateExpansionKey) skyKey.argument();
    ActionLookupValue value = (ActionLookupValue) env.getValue(key.getActionLookupKey());
    if (value == null) {
      // Because of the phase boundary separating analysis and execution, all needed
      // ActionLookupValues must have already been evaluated, so a missing ActionLookupValue is
      // unexpected. However, we tolerate this case.
      return null;
    }
    ActionTemplate<?> actionTemplate = value.getActionTemplate(key.getActionIndex());

    // Requests the TreeArtifactValue object for the input TreeArtifact.
    SkyKey artifactValueKey = ArtifactSkyKey.key(actionTemplate.getInputTreeArtifact(), true);
    TreeArtifactValue treeArtifactValue = (TreeArtifactValue) env.getValue(artifactValueKey);

    // Input TreeArtifact is not ready yet.
    if (env.valuesMissing()) {
      return null;
    }
    Iterable<TreeFileArtifact> inputTreeFileArtifacts = treeArtifactValue.getChildren();
    GeneratingActions generatingActions;
    try {
      // Expand the action template using the list of expanded input TreeFileArtifacts.
      // TODO(rduan): Add a check to verify the inputs of expanded actions are subsets of inputs
      // of the ActionTemplate.
      generatingActions =
          checkActionAndArtifactConflicts(
              actionTemplate.generateActionForInputArtifacts(inputTreeFileArtifacts, key));
    } catch (ActionConflictException e) {
      e.reportTo(env.getListener());
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (ArtifactPrefixConflictException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (ActionTemplateExpansionException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new ActionTemplateExpansionFunctionException(e);
    }

    return new ActionTemplateExpansionValue(generatingActions);
  }

  /** Exception thrown by {@link ActionTemplateExpansionFunction}. */
  public static final class ActionTemplateExpansionFunctionException extends SkyFunctionException {
    ActionTemplateExpansionFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ArtifactPrefixConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ActionTemplateExpansionException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  private GeneratingActions checkActionAndArtifactConflicts(Iterable<? extends Action> actions)
      throws ActionConflictException, ArtifactPrefixConflictException {
    GeneratingActions generatingActions =
        Actions.findAndThrowActionConflict(actionKeyContext, ImmutableList.copyOf(actions));
    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> artifactPrefixConflictMap =
        Actions.findArtifactPrefixConflicts(
            ActionLookupValue.getMapForConsistencyCheck(
                generatingActions.getGeneratingActionIndex(), generatingActions.getActions()));

    if (!artifactPrefixConflictMap.isEmpty()) {
      throw artifactPrefixConflictMap.values().iterator().next();
    }
    return generatingActions;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
