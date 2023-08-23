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
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;

/** Value that stores expanded actions from ActionTemplate. */
public final class ActionTemplateExpansionValue extends BasicActionLookupValue {

  ActionTemplateExpansionValue(ImmutableList<ActionAnalysisMetadata> generatingActions) {
    super(generatingActions);
  }

  public static ActionTemplateExpansionKey key(ActionLookupKey actionLookupKey, int actionIndex) {
    return ActionTemplateExpansionKey.of(actionLookupKey, actionIndex);
  }

  /** Key for {@link ActionTemplateExpansionValue} nodes. */
  @AutoCodec
  public static final class ActionTemplateExpansionKey implements ActionLookupKey {
    private static final Interner<ActionTemplateExpansionKey> interner =
        BlazeInterners.newWeakInterner();

    private final ActionLookupKey actionLookupKey;
    private final int actionIndex;

    private ActionTemplateExpansionKey(ActionLookupKey actionLookupKey, int actionIndex) {
      this.actionLookupKey = actionLookupKey;
      this.actionIndex = actionIndex;
    }

    @VisibleForTesting
    @AutoCodec.Instantiator
    public static ActionTemplateExpansionKey of(ActionLookupKey actionLookupKey, int actionIndex) {
      return interner.intern(new ActionTemplateExpansionKey(actionLookupKey, actionIndex));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ACTION_TEMPLATE_EXPANSION;
    }

    @Override
    public Label getLabel() {
      return actionLookupKey.getLabel();
    }

    @Override
    public BuildConfigurationKey getConfigurationKey() {
      return actionLookupKey.getConfigurationKey();
    }

    public ActionLookupKey getActionLookupKey() {
      return actionLookupKey;
    }

    /**
     * Index of the action in question in the node keyed by {@link #getActionLookupKey}. Should be
     * passed to {@link com.google.devtools.build.lib.actions.ActionLookupValue#getAction}.
     */
    public int getActionIndex() {
      return actionIndex;
    }

    @Override
    public int hashCode() {
      return 37 * actionLookupKey.hashCode() + actionIndex;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof ActionTemplateExpansionKey)) {
        return false;
      }
      ActionTemplateExpansionKey that = (ActionTemplateExpansionKey) obj;
      return this.actionIndex == that.actionIndex
          && this.actionLookupKey.equals(that.actionLookupKey);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("actionLookupKey", actionLookupKey)
          .add("actionIndex", actionIndex)
          .toString();
    }
  }
}
