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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;

/**
 * Base class for all values which can provide the generating action of an artifact. The primary
 * instance of such lookup values is {@link ConfiguredTargetValue}. Values that hold the generating
 * actions of target completion values and build info artifacts also fall into this category.
 */
public class ActionLookupValue implements SkyValue {
  protected final ImmutableMap<Artifact, ActionAnalysisMetadata> generatingActionMap;

  private static ImmutableMap<Artifact, ActionAnalysisMetadata>
      filterSharedActionsAndThrowRuntimeIfConflict(
      Iterable<ActionAnalysisMetadata> actions) {
    try {
      return Actions.filterSharedActionsAndThrowActionConflict(actions);
    } catch (ActionConflictException e) {
      // Programming bug.
      throw new IllegalStateException(e);
    }
  }

  ActionLookupValue(Iterable<ActionAnalysisMetadata> actions) {
    this(filterSharedActionsAndThrowRuntimeIfConflict(actions));
  }

  ActionLookupValue(ActionAnalysisMetadata action) {
    this(ImmutableList.of(action));
  }

  ActionLookupValue(Map<Artifact, ActionAnalysisMetadata> generatingActionMap) {
    this.generatingActionMap = ImmutableMap.copyOf(generatingActionMap);
  }

  public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
    return generatingActionMap.get(artifact);
  }

  /** To be used only when checking consistency of the action graph -- not by other values. */
  ImmutableMap<Artifact, ActionAnalysisMetadata> getMapForConsistencyCheck() {
    return generatingActionMap;
  }

  @VisibleForTesting
  public static SkyKey key(ActionLookupKey ownerKey) {
    return ownerKey.getSkyKey();
  }

  /**
   * ArtifactOwner is not a SkyKey, but we wish to convert any ArtifactOwner into a SkyKey as
   * simply as possible. To that end, all subclasses of ActionLookupValue "own" artifacts with
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
    abstract SkyFunctionName getType();

    SkyKey getSkyKeyInternal() {
      return SkyKey.create(getType(), this);
    }

    /**
     * Prefer {@link ActionLookupValue#key} to calling this method directly.
     *
     * <p>Subclasses may override {@link #getSkyKeyInternal} if the {@link SkyKey} argument should
     * not be this {@link ActionLookupKey} itself.
     */
    final SkyKey getSkyKey() {
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
