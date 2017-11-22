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

package com.google.devtools.build.lib.rules.cpp;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.SpawnResult;
import java.util.List;
import javax.annotation.Nullable;

/** Contains information about the result of a CppCompileAction's execution. */
@AutoValue
public abstract class CppCompileActionResult {

  /** Returns the SpawnResults created by the action, if any. */
  public abstract List<SpawnResult> spawnResults();

  /**
   * Gets the optional CppCompileActionContext.Reply for the action.
   *
   * <p>Could be null if there is no reply (e.g. if there is no .d file documenting which #include
   * statements are actually required.)
   */
  @Nullable
  public abstract CppCompileActionContext.Reply contextReply();

  /** Returns a builder that can be used to construct a {@link CppCompileActionResult} object. */
  public static Builder builder() {
    return new AutoValue_CppCompileActionResult.Builder();
  }

  /** Builder for a {@link CppCompileActionResult} instance, which is immutable once built. */
  @AutoValue.Builder
  public abstract static class Builder {

    /** Returns the SpawnResults for the action, if any. */
    abstract List<SpawnResult> spawnResults();

    /** Sets the SpawnResults for the action. */
    public abstract Builder setSpawnResults(List<SpawnResult> spawnResults);

    /** Sets the CppCompileActionContext.Reply for the action. */
    public abstract Builder setContextReply(CppCompileActionContext.Reply reply);

    abstract CppCompileActionResult realBuild();

    /**
     * Returns an immutable CppCompileActionResult object.
     *
     * <p>The list of SpawnResults is also made immutable here.
     */
    public CppCompileActionResult build() {
      return this.setSpawnResults(ImmutableList.copyOf(spawnResults())).realBuild();
    }
  }
}
