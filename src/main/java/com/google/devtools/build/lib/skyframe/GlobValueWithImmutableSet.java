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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A value corresponding to a glob which uses {@link ImmutableSet} as the container to store
 * matching {@link PathFragment}s.
 */
@Immutable
@ThreadSafe
public final class GlobValueWithImmutableSet extends GlobValue {

  public static final GlobValueWithImmutableSet EMPTY =
      new GlobValueWithImmutableSet(ImmutableSet.of());

  private final ImmutableSet<PathFragment> matches;

  /** Creates a {@link GlobValueWithImmutableSet} wrapping {@code matches}. */
  public GlobValueWithImmutableSet(ImmutableSet<PathFragment> matches) {
    this.matches = Preconditions.checkNotNull(matches);
  }

  /** Returns an unordered {@link ImmutableSet} containing all glob matches. */
  @Override
  public ImmutableSet<PathFragment> getMatches() {
    return matches;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (!(other instanceof GlobValueWithImmutableSet)) {
      return false;
    }
    return matches.equals(((GlobValueWithImmutableSet) other).matches);
  }

  @Override
  public int hashCode() {
    return matches.hashCode();
  }
}
