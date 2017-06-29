// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.LegacySkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Base class for all values which can provide the generating action of an artifact. The primary
 * instance of such lookup values is {@code ConfiguredTargetValue}. Values that hold the generating
 * actions of target completion values and build info artifacts also fall into this category.
 */
public class ActionLookupValue implements SkyValue {
  protected final List<ActionAnalysisMetadata> actions;
  private final ImmutableMap<Artifact, Integer> generatingActionIndex;

  private static Actions.GeneratingActions filterSharedActionsAndThrowRuntimeIfConflict(
      List<ActionAnalysisMetadata> actions) {
    try {
      return Actions.filterSharedActionsAndThrowActionConflict(actions);
    } catch (ActionConflictException e) {
      // Programming bug.
      throw new IllegalStateException(e);
    }
  }

  @VisibleForTesting
  public ActionLookupValue(
      List<ActionAnalysisMetadata> actions, boolean removeActionsAfterEvaluation) {
    this(filterSharedActionsAndThrowRuntimeIfConflict(actions), removeActionsAfterEvaluation);
  }

  protected ActionLookupValue(ActionAnalysisMetadata action, boolean removeActionAfterEvaluation) {
    this(ImmutableList.of(action), removeActionAfterEvaluation);
  }

  protected ActionLookupValue(
      Actions.GeneratingActions generatingActions, boolean removeActionsAfterEvaluation) {
    if (removeActionsAfterEvaluation) {
      this.actions = new ArrayList<>(generatingActions.getActions());
    } else {
      this.actions = ImmutableList.copyOf(generatingActions.getActions());
    }
    this.generatingActionIndex = generatingActions.getGeneratingActionIndex();
  }

  /**
   * Returns the action that generates {@code artifact}, if known to this value, or null. This
   * method should be avoided. Call it only when the action is really needed, and it is known to be
   * present, either because the execution phase has not started, or because {@link
   * Action#canRemoveAfterExecution} is known to be false for the action being requested.
   */
  @Nullable
  public ActionAnalysisMetadata getGeneratingActionDangerousReadJavadoc(Artifact artifact) {
    Integer actionIndex = getGeneratingActionIndex(artifact);
    if (actionIndex == null) {
      return null;
    }
    return getActionAnalysisMetadata(actionIndex);
  }

  /**
   * Returns the index of the action that generates {@code artifact} in this value, or null if this
   * value does not have a generating action for this artifact. The index together with the key for
   * this {@link ActionLookupValue} uniquely identifies the action.
   *
   * <p>Unlike {@link #getAction}, this may be called after action execution.
   */
  @Nullable
  public Integer getGeneratingActionIndex(Artifact artifact) {
    return generatingActionIndex.get(artifact);
  }

  /**
   * Returns the {@link Action} with index {@code index} in this value. Never null. Should only be
   * called during action execution by {@code ArtifactFunction} and {@code ActionExecutionFunction}
   * -- after an action has executed, calling this with its index may crash.
   */
  @SuppressWarnings("unchecked") // We test to make sure it's an Action.
  public Action getAction(int index) {
    ActionAnalysisMetadata result = getActionAnalysisMetadata(index);
    Preconditions.checkState(result instanceof Action, "Not action: %s %s %s", result, index, this);
    return (Action) result;
  }

  private ActionAnalysisMetadata getActionAnalysisMetadata(int index) {
    return Preconditions.checkNotNull(actions.get(index), "null action: %s %s", index, this);
  }

  /**
   * Returns the {@link ActionAnalysisMetadata} at index {@code index} if it is present and
   * <i>not</i> an {@link Action}. Tree artifacts need their {@code ActionTemplate}s in order to
   * generate the correct actions, but in general most actions are not needed after they are
   * executed and may not even be available.
   */
  public ActionAnalysisMetadata getIfPresentAndNotAction(int index) {
    ActionAnalysisMetadata actionAnalysisMetadata = actions.get(index);
    if (!(actionAnalysisMetadata instanceof Action)) {
      return actionAnalysisMetadata;
    }
    return null;
  }

  /** To be used only when checking consistency of the action graph -- not by other values. */
  public Map<Artifact, ActionAnalysisMetadata> getMapForConsistencyCheck() {
    return getMapForConsistencyCheck(generatingActionIndex, actions);
  }

  protected ToStringHelper getStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("actions", actions)
        .add("generatingActionIndex", generatingActionIndex);
  }

  @Override
  public String toString() {
    return getStringHelper().toString();
  }

  @VisibleForTesting
  public static SkyKey key(ActionLookupKey ownerKey) {
    return ownerKey.getSkyKey();
  }

  public int getNumActions() {
    return actions.size();
  }

  public static Map<Artifact, ActionAnalysisMetadata> getMapForConsistencyCheck(
      Map<Artifact, Integer> generatingActionIndex,
      final List<? extends ActionAnalysisMetadata> actions) {
    return Maps.transformValues(generatingActionIndex, actions::get);
  }

  /**
   * If this object was initialized with {@code removeActionsAfterEvaluation} and {@link
   * Action#canRemoveAfterExecution()} is true for {@code action}, then remove this action from this
   * object's index as a memory-saving measure. The {@code artifact -> index} mapping remains
   * intact, so this action's execution value can still be addressed by its inputs.
   */
  @ThreadSafe
  public void actionEvaluated(int actionIndex, Action action) {
    if (!action.canRemoveAfterExecution()) {
      return;
    }
    if (actions instanceof ArrayList) {
      // This method may concurrently mutate an ArrayList, which is unsafe on its face. However,
      // ArrayList mutation on different indices that does not affect the size of the ArrayList is
      // safe, and that is what does this code does.
      ArrayList<ActionAnalysisMetadata> actionArrayList =
          (ArrayList<ActionAnalysisMetadata>) actions;
      ActionAnalysisMetadata oldAction = actionArrayList.set(actionIndex, null);
      Preconditions.checkState(
          action.equals(oldAction), "Not same: %s %s %s %s", action, oldAction, this, actionIndex);
    }
  }

  /**
   * ArtifactOwner is not a SkyKey, but we wish to convert any ArtifactOwner into a SkyKey as simply
   * as possible. To that end, all subclasses of ActionLookupValue "own" artifacts with
   * ArtifactOwners that are subclasses of ActionLookupKey. This allows callers to easily find the
   * value key, while remaining agnostic to what ActionLookupValues actually exist.
   *
   * <p>The methods of this class should only be called by {@link ActionLookupValue#key}.
   */
  public abstract static class ActionLookupKey implements ArtifactOwner {
    @Override
    public Label getLabel() {
      return null;
    }

    /**
     * Subclasses must override this to specify their specific value type, unless they override
     * {@link #getSkyKey}, in which case they are free not to implement this method.
     */
    protected abstract SkyFunctionName getType();

    protected SkyKey getSkyKeyInternal() {
      return LegacySkyKey.create(getType(), this);
    }

    /**
     * Prefer {@link ActionLookupValue#key} to calling this method directly.
     *
     * <p>Subclasses may override {@link #getSkyKeyInternal} if the {@link SkyKey} argument should
     * not be this {@link ActionLookupKey} itself.
     */
    public final SkyKey getSkyKey() {
      SkyKey result = getSkyKeyInternal();
      Preconditions.checkState(
          result.argument() instanceof ActionLookupKey,
          "Not ActionLookupKey for %s: %s",
          this,
          result);
      return result;
    }
  }
}
