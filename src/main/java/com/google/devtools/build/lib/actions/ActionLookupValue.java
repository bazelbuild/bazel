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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** Base class for all values which can provide the generating action of an artifact. */
public abstract class ActionLookupValue implements SkyValue {

  /** Returns a list of actions registered by this {@link SkyValue}. */
  public abstract ImmutableList<ActionAnalysisMetadata> getActions();

  /**
   * Returns the {@link Action} with index {@code index} in this value. Never null. Should only be
   * called during action execution by {@code ArtifactFunction} and {@code ActionExecutionFunction}
   * -- after an action has executed, calling this with its index may crash.
   */
  public Action getAction(int index) {
    ActionAnalysisMetadata result = getActions().get(index);
    // Avoid Preconditions.checkState which would box the int arg.
    if (!(result instanceof Action)) {
      throw new IllegalStateException(String.format("Not action: %s %s %s", result, index, this));
    }
    return (Action) result;
  }

  public ActionTemplate<?> getActionTemplate(int index) {
    ActionAnalysisMetadata result = getActions().get(index);
    // Avoid Preconditions.checkState which would box the int arg.
    if (!(result instanceof ActionTemplate)) {
      throw new IllegalStateException(
          String.format("Not action template: %s %s %s", result, index, this));
    }
    return (ActionTemplate<?>) result;
  }

  /** Returns the number of {@link Action} objects present in this value. */
  public int getNumActions() {
    return getActions().size();
  }

  /** Returns a source artifact if the underlying configured target is an input file. */
  @Nullable
  public SourceArtifact getSourceArtifact() {
    return null;
  }

  /**
   * All subclasses of ActionLookupValue "own" artifacts with {@link ArtifactOwner}s that are
   * subclasses of ActionLookupKey. This allows callers to easily find the value key, while
   * remaining agnostic to what ActionLookupValues actually exist.
   */
  public abstract static class ActionLookupKey implements ArtifactOwner, SkyKey {
    @Override
    public Label getLabel() {
      return null;
    }
  }
}
