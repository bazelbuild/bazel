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
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.ExecutionPhaseSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;

/** Data that uniquely identifies an action. */
public class ActionLookupData implements ExecutionPhaseSkyKey {

  private final ActionLookupKey actionLookupKey;
  private final int actionIndex;

  private ActionLookupData(ActionLookupKey actionLookupKey, int actionIndex) {
    this.actionLookupKey = Preconditions.checkNotNull(actionLookupKey);
    this.actionIndex = actionIndex;
  }

  /**
   * Creates a key for the result of action execution. Does <i>not</i> intern its results, so should
   * only be called once per {@code (actionLookupKey, actionIndex)} pair.
   */
  public static ActionLookupData create(ActionLookupKey actionLookupKey, int actionIndex) {
    // If the label is null, this is not a typical action (it may be the build info action or a
    // coverage action, for example). Don't try to share it.
    return actionLookupKey.getLabel() == null
        ? createUnshareable(actionLookupKey, actionIndex)
        : new ActionLookupData(actionLookupKey, actionIndex);
  }

  /**
   * Similar to {@link #create}, but the key will return {@code false} for {@link
   * #valueIsShareable}.
   */
  public static ActionLookupData createUnshareable(
      ActionLookupKey actionLookupKey, int actionIndex) {
    return new UnshareableActionLookupData(actionLookupKey, actionIndex);
  }

  public ActionLookupKey getActionLookupKey() {
    return actionLookupKey;
  }

  /**
   * Index of the action in question in the node keyed by {@link #getActionLookupKey}. Should be
   * passed to {@link ActionLookupValue#getAction}.
   */
  public final int getActionIndex() {
    return actionIndex;
  }

  public final Label getLabel() {
    return actionLookupKey.getLabel();
  }

  @Override
  public final int hashCode() {
    int hash = 1;
    hash = 37 * hash + actionLookupKey.hashCode();
    hash = 37 * hash + Integer.hashCode(actionIndex);
    hash = 37 * hash + Boolean.hashCode(valueIsShareable());
    return hash;
  }

  @Override
  public final boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ActionLookupData)) {
      return false;
    }
    ActionLookupData that = (ActionLookupData) obj;
    return this.actionIndex == that.actionIndex
        && this.actionLookupKey.equals(that.actionLookupKey)
        && valueIsShareable() == that.valueIsShareable();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("actionLookupKey", actionLookupKey)
        .add("actionIndex", actionIndex)
        .toString();
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.ACTION_EXECUTION;
  }

  private static final class UnshareableActionLookupData extends ActionLookupData {
    private UnshareableActionLookupData(ActionLookupKey actionLookupKey, int actionIndex) {
      super(actionLookupKey, actionIndex);
    }

    @Override
    public boolean valueIsShareable() {
      return false;
    }
  }
}
