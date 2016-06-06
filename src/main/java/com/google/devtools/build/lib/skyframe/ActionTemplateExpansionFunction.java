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
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * The SkyFunction for {@link ActionTemplateExpansionValue}.
 *
 * <p>Given an action template, this function resolves its input TreeArtifact, then expands the
 * action template into a list of actions using the expanded {@link TreeFileArtifact}s under the
 * input TreeArtifact.
 */
public class ActionTemplateExpansionFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ActionTemplateExpansionFunctionException {
    ActionTemplateExpansionKey key = (ActionTemplateExpansionKey) skyKey.argument();
    SpawnActionTemplate actionTemplate = key.getActionTemplate();

    // Requests the TreeArtifactValue object for the input TreeArtifact.
    SkyKey artifactValueKey = ArtifactValue.key(actionTemplate.getInputTreeArtifact(), true);
    TreeArtifactValue treeArtifactValue = (TreeArtifactValue) env.getValue(artifactValueKey);

    // Input TreeArtifact is not ready yet.
    if (env.valuesMissing()) {
      return null;
    }
    Iterable<TreeFileArtifact> inputTreeFileArtifacts = treeArtifactValue.getChildren();
    Iterable<Action> expandedActions;
    try {
      // Expand the action template using the list of expanded input TreeFileArtifacts.
      expandedActions = ImmutableList.<Action>copyOf(
          actionTemplate.generateActionForInputArtifacts(inputTreeFileArtifacts, key));
    } catch (ActionConflictException e) {
      e.reportTo(env.getListener());
      throw new ActionTemplateExpansionFunctionException(e);
    } catch (ArtifactPrefixConflictException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new ActionTemplateExpansionFunctionException(e);
    }

    return new ActionTemplateExpansionValue(expandedActions);
  }

  /** Exception thrown by {@link ActionTemplateExpansionFunction}. */
  public static final class ActionTemplateExpansionFunctionException extends SkyFunctionException {
    ActionTemplateExpansionFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    ActionTemplateExpansionFunctionException(ArtifactPrefixConflictException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
