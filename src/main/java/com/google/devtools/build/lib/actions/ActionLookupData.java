// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;

/** Data that uniquely identifies an action. */
public class ActionLookupData {
  private final SkyKey actionLookupNode;
  private final int actionIndex;

  public ActionLookupData(SkyKey actionLookupNode, int actionIndex) {
    Preconditions.checkState(
        actionLookupNode.argument() instanceof ActionLookupValue.ActionLookupKey, actionLookupNode);
    this.actionLookupNode = actionLookupNode;
    this.actionIndex = actionIndex;
  }

  public SkyKey getActionLookupNode() {
    return actionLookupNode;
  }

  /**
   * Index of the action in question in the node keyed by {@link #getActionLookupNode}. Should be
   * passed to {@link ActionLookupValue#getAction}.
   */
  public int getActionIndex() {
    return actionIndex;
  }

  public Label getLabelForErrors() {
    return ((ActionLookupValue.ActionLookupKey) actionLookupNode.argument()).getLabel();
  }

  @Override
  public int hashCode() {
    return 37 * actionLookupNode.hashCode() + actionIndex;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ActionLookupData)) {
      return false;
    }
    ActionLookupData that = (ActionLookupData) obj;
    return this.actionIndex == that.actionIndex
        && this.actionLookupNode.equals(that.actionLookupNode);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("actionLookupNode", actionLookupNode)
        .add("actionIndex", actionIndex)
        .toString();
  }
}
