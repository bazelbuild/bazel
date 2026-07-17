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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A value corresponding to a glob which uses {@link NestedSet} as the container to store matching
 * {@link PathFragment}s.
 *
 * <p>Used by {@link GlobFunctionWithRecursiveGlobbing} as a way to save memory when bubbling sub
 * glob node matches to parent glob nodes. All sub-glob node matches are stored only as a reference
 * in its parent {@link GlobValueWithNestedSet#matches} container.
 */
@Immutable
@ThreadSafe
public final class GlobValueWithNestedSet extends GlobValue {

  public static final GlobValueWithNestedSet EMPTY =
      new GlobValueWithNestedSet(NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  private final NestedSet<PathFragment> matches;

  /**
   * Create a GlobValue wrapping {@code matches}. {@code matches} must have order {@link
   * Order#STABLE_ORDER}.
   */
  public GlobValueWithNestedSet(NestedSet<PathFragment> matches) {
    this.matches = Preconditions.checkNotNull(matches);
    Preconditions.checkState(
        matches.getOrder() == Order.STABLE_ORDER,
        "Only STABLE_ORDER is supported, but got %s",
        matches.getOrder());
  }

  /**
   * Returns glob matches stored in {@link NestedSet}. The matches will be in a deterministic but
   * unspecified order. If a particular order is required, the returned iterable should be sorted.
   */
  public NestedSet<PathFragment> getMatchesInNestedSet() {
    return matches;
  }

  @Override
  public ImmutableSet<PathFragment> getMatches() {
    return matches.toSet();
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (!(other instanceof GlobValueWithNestedSet)) {
      return false;
    }
    // shallowEquals() may fail to detect that two equivalent (according to toString())
    // NestedSets are equal, but will always detect when two NestedSets are different.
    // This makes this implementation of equals() overly strict, but we only call this
    // method when doing change pruning, which can accept false negatives.
    return getMatchesInNestedSet()
        .shallowEquals(((GlobValueWithNestedSet) other).getMatchesInNestedSet());
  }

  @Override
  public int hashCode() {
    return matches.shallowHashCode();
  }
}
