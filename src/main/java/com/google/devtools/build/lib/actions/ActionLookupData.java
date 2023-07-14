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

/**
 * Data that uniquely identifies an action.
 *
 * <p>{@link #getActionLookupKey} returns the {@link ActionLookupKey} to look up an {@link
 * ActionLookupValue}. {@link #getActionIndex} returns the index of the action within {@link
 * ActionLookupValue#getActions}.
 *
 * <p>To save memory, a custom subclass without an {@code int} field is used for the most common
 * action indices [0-9].
 */
public abstract class ActionLookupData implements ExecutionPhaseSkyKey {

  /**
   * Creates a key for the result of action execution. Does <i>not</i> intern its results, so should
   * only be called once per {@code (actionLookupKey, actionIndex)} pair.
   */
  public static ActionLookupData create(ActionLookupKey actionLookupKey, int actionIndex) {
    if (!actionLookupKey.mayOwnShareableActions()) {
      return createUnshareable(actionLookupKey, actionIndex);
    }
    switch (actionIndex) {
      case 0:
        return new ActionLookupData0(actionLookupKey);
      case 1:
        return new ActionLookupData1(actionLookupKey);
      case 2:
        return new ActionLookupData2(actionLookupKey);
      case 3:
        return new ActionLookupData3(actionLookupKey);
      case 4:
        return new ActionLookupData4(actionLookupKey);
      case 5:
        return new ActionLookupData5(actionLookupKey);
      case 6:
        return new ActionLookupData6(actionLookupKey);
      case 7:
        return new ActionLookupData7(actionLookupKey);
      case 8:
        return new ActionLookupData8(actionLookupKey);
      case 9:
        return new ActionLookupData9(actionLookupKey);
      default:
        return new ActionLookupDataN(actionLookupKey, actionIndex);
    }
  }

  /**
   * Similar to {@link #create}, but the key will return {@code false} for {@link
   * #valueIsShareable}.
   */
  public static ActionLookupData createUnshareable(
      ActionLookupKey actionLookupKey, int actionIndex) {
    return new UnshareableActionLookupData(actionLookupKey, actionIndex);
  }

  private final ActionLookupKey actionLookupKey;

  private ActionLookupData(ActionLookupKey actionLookupKey) {
    this.actionLookupKey = Preconditions.checkNotNull(actionLookupKey);
  }

  public final ActionLookupKey getActionLookupKey() {
    return actionLookupKey;
  }

  /**
   * Index of the action in question in the node keyed by {@link #getActionLookupKey}. Should be
   * passed to {@link ActionLookupValue#getAction}.
   */
  public abstract int getActionIndex();

  public final Label getLabel() {
    return actionLookupKey.getLabel();
  }

  @Override
  public final int hashCode() {
    int hash = 1;
    hash = 37 * hash + actionLookupKey.hashCode();
    hash = 37 * hash + Integer.hashCode(getActionIndex());
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
    return getActionIndex() == that.getActionIndex()
        && actionLookupKey.equals(that.actionLookupKey)
        && valueIsShareable() == that.valueIsShareable();
  }

  @Override
  public final String toString() {
    return MoreObjects.toStringHelper(this)
        .add("actionLookupKey", actionLookupKey)
        .add("actionIndex", getActionIndex())
        .toString();
  }

  @Override
  public final SkyFunctionName functionName() {
    return SkyFunctions.ACTION_EXECUTION;
  }

  private static final class ActionLookupData0 extends ActionLookupData {
    private ActionLookupData0(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 0;
    }
  }

  private static final class ActionLookupData1 extends ActionLookupData {
    private ActionLookupData1(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 1;
    }
  }

  private static final class ActionLookupData2 extends ActionLookupData {
    private ActionLookupData2(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 2;
    }
  }

  private static final class ActionLookupData3 extends ActionLookupData {
    private ActionLookupData3(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 3;
    }
  }

  private static final class ActionLookupData4 extends ActionLookupData {
    private ActionLookupData4(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 4;
    }
  }

  private static final class ActionLookupData5 extends ActionLookupData {
    private ActionLookupData5(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 5;
    }
  }

  private static final class ActionLookupData6 extends ActionLookupData {
    private ActionLookupData6(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 6;
    }
  }

  private static final class ActionLookupData7 extends ActionLookupData {
    private ActionLookupData7(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 7;
    }
  }

  private static final class ActionLookupData8 extends ActionLookupData {
    private ActionLookupData8(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 8;
    }
  }

  private static final class ActionLookupData9 extends ActionLookupData {
    private ActionLookupData9(ActionLookupKey actionLookupKey) {
      super(actionLookupKey);
    }

    @Override
    public int getActionIndex() {
      return 9;
    }
  }

  private static class ActionLookupDataN extends ActionLookupData {
    private final int actionIndex;

    private ActionLookupDataN(ActionLookupKey actionLookupKey, int actionIndex) {
      super(actionLookupKey);
      this.actionIndex = actionIndex;
    }

    @Override
    public final int getActionIndex() {
      return actionIndex;
    }
  }

  private static final class UnshareableActionLookupData extends ActionLookupDataN {
    private UnshareableActionLookupData(ActionLookupKey actionLookupKey, int actionIndex) {
      super(actionLookupKey, actionIndex);
    }

    @Override
    public boolean valueIsShareable() {
      return false;
    }
  }
}
