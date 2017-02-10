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
import com.google.devtools.build.lib.analysis.actions.ActionTemplate;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Value that stores expanded actions from ActionTemplate.
 */
public final class ActionTemplateExpansionValue extends ActionLookupValue {
  private final Iterable<Action> expandedActions;

  ActionTemplateExpansionValue(Iterable<Action> expandedActions) {
    super(ImmutableList.<ActionAnalysisMetadata>copyOf(expandedActions));
    this.expandedActions = ImmutableList.copyOf(expandedActions);
  }

  Iterable<Action> getExpandedActions() {
    return expandedActions;
  }

  static SkyKey key(ActionTemplate<?> actionTemplate) {
    return SkyKey.create(
        SkyFunctions.ACTION_TEMPLATE_EXPANSION,
        createActionTemplateExpansionKey(actionTemplate));
  }

  static ActionTemplateExpansionKey createActionTemplateExpansionKey(
      ActionTemplate<?> actionTemplate) {
    return new ActionTemplateExpansionKey(actionTemplate);
  }


  static final class ActionTemplateExpansionKey extends ActionLookupKey {
    private final ActionTemplate<?> actionTemplate;

    ActionTemplateExpansionKey(ActionTemplate<?> actionTemplate) {
      Preconditions.checkNotNull(
          actionTemplate,
          "Passed in action template cannot be null: %s",
          actionTemplate);
      this.actionTemplate = actionTemplate;
    }

    @Override
    SkyFunctionName getType() {
      return SkyFunctions.ACTION_TEMPLATE_EXPANSION;
    }


    @Override
    public Label getLabel() {
      return actionTemplate.getOwner().getLabel();
    }

    /** Returns the associated {@link ActionTemplate} */
    ActionTemplate<?> getActionTemplate() {
      return actionTemplate;
    }

    @Override
    public int hashCode() {
      return actionTemplate.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
     if (this == obj) {
       return true;
     }
      if (obj == null) {
        return false;
      }
      if (!(obj instanceof ActionTemplateExpansionKey)) {
        return false;
      }
      ActionTemplateExpansionKey other = (ActionTemplateExpansionKey) obj;
      return actionTemplate.equals(other.actionTemplate);
    }
  }
}
