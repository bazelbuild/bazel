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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.function.Consumer;

/**
 * Helper class to represent "evaluated" phony target.
 */
public class PhonyTarget {
  private final ImmutableList<PathFragment> phonyClosureNames;
  private final ImmutableList<PathFragment> directUsualInputs;

  public PhonyTarget(
      ImmutableList<PathFragment> phonyClosureNames,
      ImmutableList<PathFragment> directUsualInputs) {
    this.phonyClosureNames = phonyClosureNames;
    this.directUsualInputs = directUsualInputs;
  }

  public ImmutableList<PathFragment> getPhonyClosureNames() {
    return phonyClosureNames;
  }

  public ImmutableList<PathFragment> getDirectUsualInputs() {
    return directUsualInputs;
  }

  public void visitUsualInputs(ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap,
      Consumer<ImmutableList<PathFragment>> consumer) {
    consumer.accept(directUsualInputs);
    // phonyTarget.getPhonyClosureNames() is flattened transitive closure,
    // only visit first layer.
    phonyClosureNames.forEach(name ->
        consumer.accept(
            Preconditions.checkNotNull(phonyTargetsMap.get(name)).getDirectUsualInputs()));
  }
}
