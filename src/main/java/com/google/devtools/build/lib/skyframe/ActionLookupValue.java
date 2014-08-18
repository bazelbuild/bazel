// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.HashMap;
import java.util.Map;

/**
 * Base class for all values which can provide the generating action of an artifact. The primary
 * instance of such lookup values is {@link ConfiguredTargetValue}. Values that hold the generating
 * actions of target completion values and build info artifacts also fall into this category.
 */
public class ActionLookupValue implements SkyValue {
  protected final ImmutableMap<Artifact, Action> generatingActionMap;

  ActionLookupValue(Iterable<Action> actions) {
    // Duplicate/shared actions get passed in all the time. Blaze is weird. We can't double-register
    // the generated artifacts in an immutable map builder, so we double-register them in a more
    // forgiving map, and then use that map to create the immutable one.
    Map<Artifact, Action> generatingActions = new HashMap<>();
    for (Action action : actions) {
      for (Artifact artifact : action.getOutputs()) {
        generatingActions.put(artifact, action);
      }
    }
    generatingActionMap = ImmutableMap.copyOf(generatingActions);
  }

  ActionLookupValue(Action action) {
    this(ImmutableList.of(action));
  }

  Action getGeneratingAction(Artifact artifact) {
    return generatingActionMap.get(artifact);
  }

  /** To be used only when checking consistency of the action graph -- not by other values. */
  ImmutableMap<Artifact, Action> getMapForConsistencyCheck() {
    return generatingActionMap;
  }

  /**
   * To be used only when setting the owners of deserialized artifacts whose owners were unknown at
   * creation time -- not by other callers or values.
   */
  Iterable<Action> getActionsForFindingArtifactOwners() {
    return generatingActionMap.values();
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
  protected abstract static class ActionLookupKey implements ArtifactOwner {
    @Override
    public Label getLabel() {
      return null;
    }

    /**
     * Subclasses must override this to specify their specific value type, unless they override
     * {@link #getSkyKey}, in which case they are free not to implement this method.
     */
    abstract SkyFunctionName getType();

    /**
     * Prefer {@link ActionLookupValue#key} to calling this method directly.
     *
     * <p>Subclasses may override if the value key contents should not be the key itself.
     */
    SkyKey getSkyKey() {
      return new SkyKey(getType(), this);
    }
  }
}
