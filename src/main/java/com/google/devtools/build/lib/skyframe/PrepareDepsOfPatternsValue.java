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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.query2.common.UniverseSkyKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * The value returned by {@link PrepareDepsOfPatternsFunction}. Although that function is invoked
 * primarily for its side effect (i.e. ensuring the graph contains targets matching the pattern
 * sequence and their transitive dependencies), this value contains the {@link TargetPatternKey}
 * arguments of the {@link PrepareDepsOfPatternFunction}s evaluated in service of it.
 *
 * <p>Because the returned value may remain the same when the side-effects of this function
 * evaluation change, this value and the {@link PrepareDepsOfPatternsFunction} which computes it are
 * incompatible with change pruning. It should only be requested by consumers who do not require
 * reevaluation when {@link PrepareDepsOfPatternsFunction} is reevaluated. Safe consumers include,
 * e.g., top-level consumers, and other functions which invoke {@link PrepareDepsOfPatternsFunction}
 * solely for its side-effects and which do not behave differently depending on those side-effects.
 */
@Immutable
@ThreadSafe
public final class PrepareDepsOfPatternsValue implements SkyValue {
  private final ImmutableList<TargetPatternKey> targetPatternKeys;

  public PrepareDepsOfPatternsValue(ImmutableList<TargetPatternKey> targetPatternKeys) {
    this.targetPatternKeys = Preconditions.checkNotNull(targetPatternKeys);
  }

  public ImmutableList<TargetPatternKey> getTargetPatternKeys() {
    return targetPatternKeys;
  }

  @ThreadSafe
  public static TargetPatternSequence key(ImmutableList<String> patterns, PathFragment offset) {
    return TargetPatternSequence.create(patterns, offset);
  }

  /** The argument value for {@link SkyKey}s of {@link PrepareDepsOfPatternsFunction}. */
  @ThreadSafe
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class TargetPatternSequence implements UniverseSkyKey {
    private static final Interner<TargetPatternSequence> interner =
        BlazeInterners.newWeakInterner();

    private final ImmutableList<String> patterns;
    private final PathFragment offset;

    private TargetPatternSequence(ImmutableList<String> patterns, PathFragment offset) {
      this.patterns = Preconditions.checkNotNull(patterns);
      this.offset = Preconditions.checkNotNull(offset);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static TargetPatternSequence create(ImmutableList<String> patterns, PathFragment offset) {
      return interner.intern(new TargetPatternSequence(patterns, offset));
    }

    @Override
    public ImmutableList<String> getPatterns() {
      return patterns;
    }

    public PathFragment getOffset() {
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

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("patterns", patterns)
          .add("offset", offset)
          .toString();
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PREPARE_DEPS_OF_PATTERNS;
    }
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof PrepareDepsOfPatternsValue
        && targetPatternKeys.equals(((PrepareDepsOfPatternsValue) other).getTargetPatternKeys());
  }

  @Override
  public int hashCode() {
    return targetPatternKeys.hashCode();
  }
}
