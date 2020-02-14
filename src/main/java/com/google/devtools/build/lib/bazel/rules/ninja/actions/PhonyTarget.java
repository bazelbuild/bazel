// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Helper class to represent "evaluated" phony target: it contains the NestedSet with all non-phony
 * inputs to the phony target (with Artifacts for the real computation, and PathFragments for the
 * tests); and it contains the flag whether this phony target is always dirty, i.e. must be rebuild
 * each time.
 *
 * <p>Always-dirty phony targets are those which do not have any inputs: "build alias: phony". All
 * usual direct dependants of those actions automatically also always-dirty (but not the transitive
 * dependants: they should check whether their computed inputs have changed). As phony targets are
 * not performing any actions, <b>all phony transitive dependants of always-dirty phony targets are
 * themselves always-dirty.</b> That is why we can compute the always-dirty flag for the phony
 * targets, and use it for marking their direct non-phony dependants as actions to be executed
 * unconditionally.
 *
 * @param <T> either Artifact or PathFragment
 */
@Immutable
public final class PhonyTarget<T> {
  private final NestedSet<T> inputs;
  private final boolean isAlwaysDirty;

  public PhonyTarget(NestedSet<T> inputs, boolean isAlwaysDirty) {
    this.inputs = inputs;
    this.isAlwaysDirty = isAlwaysDirty;
  }

  public NestedSet<T> getInputs() {
    return inputs;
  }

  public boolean isAlwaysDirty() {
    return isAlwaysDirty;
  }
}
