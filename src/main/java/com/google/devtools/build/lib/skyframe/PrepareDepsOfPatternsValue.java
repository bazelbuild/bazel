// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.Serializable;
import java.util.Objects;

/**
 * The value returned by {@link PrepareDepsOfPatternsFunction}. Although that function is
 * invoked primarily for its side effect (i.e. ensuring the graph contains targets matching the
 * pattern sequence and their transitive dependencies), this value contains the
 * {@link TargetPatternKey} arguments of the {@link PrepareDepsOfPatternFunction}s evaluated in
 * service of it.
 *
 * <p>Because the returned value may remain the same when the side-effects of this function
 * evaluation change, this value and the {@link PrepareDepsOfPatternsFunction} which computes it
 * are incompatible with change pruning. It should only be requested by consumers who do not
 * require reevaluation when {@link PrepareDepsOfPatternsFunction} is reevaluated. Safe consumers
 * include, e.g., top-level consumers, and other functions which invoke {@link
 * PrepareDepsOfPatternsFunction} solely for its side-effects and which do not behave differently
 * depending on those side-effects.
 */
@Immutable
@ThreadSafe
public final class PrepareDepsOfPatternsValue implements SkyValue {

  private final ImmutableList<TargetPatternKey> targetPatternKeys;

  PrepareDepsOfPatternsValue(ImmutableList<TargetPatternKey> targetPatternKeys) {
    this.targetPatternKeys = targetPatternKeys;
  }

  public ImmutableList<TargetPatternKey> getTargetPatternKeys() {
    return targetPatternKeys;
  }

  @ThreadSafe
  public static SkyKey key(ImmutableList<String> patterns, String offset) {
    return SkyKey.create(
        SkyFunctions.PREPARE_DEPS_OF_PATTERNS, new TargetPatternSequence(patterns, offset));
  }

  /** The argument value for {@link SkyKey}s of {@link PrepareDepsOfPatternsFunction}. */
  @ThreadSafe
  public static class TargetPatternSequence implements Serializable {
    private final ImmutableList<String> patterns;
    private final String offset;

    public TargetPatternSequence(ImmutableList<String> patterns, String offset) {
      this.patterns = Preconditions.checkNotNull(patterns);
      this.offset = Preconditions.checkNotNull(offset);
    }

    public ImmutableList<String> getPatterns() {
      return patterns;
    }

    public String getOffset() {
      return offset;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof TargetPatternSequence)) {
        return false;
      }
      TargetPatternSequence that = (TargetPatternSequence) o;
      return Objects.equals(offset, that.offset) && Objects.equals(patterns, that.patterns);
    }

    @Override
    public int hashCode() {
      return Objects.hash(patterns, offset);
    }
  }
}
