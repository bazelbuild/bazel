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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import java.util.Set;

/** Holds the result(s) of an action's execution. */
@AutoValue
public abstract class ActionResult {

  /** An empty ActionResult used by Actions that don't have any metadata to return. */
  public static final ActionResult EMPTY = ActionResult.create(ImmutableSet.of());

  /** Returns the SpawnResults for the action. */
  public abstract Set<SpawnResult> spawnResults();

  /** Returns a builder that can be used to construct a {@link ActionResult} object. */
  public static Builder builder() {
    return new AutoValue_ActionResult.Builder();
  }

  /** Creates an ActionResult given a set of SpawnResults. */
  public static ActionResult create(Set<SpawnResult> spawnResults) {
    if (spawnResults == null) {
      return EMPTY;
    } else {
      return builder().setSpawnResults(spawnResults).build();
    }
  }

  /** Builder for a {@link ActionResult} instance, which is immutable once built. */
  @AutoValue.Builder
  public abstract static class Builder {

    /** Sets the SpawnResults for the action. */
    public abstract Builder setSpawnResults(Set<SpawnResult> spawnResults);

    /** Builds and returns an ActionResult object. */
    public abstract ActionResult build();
  }
}
