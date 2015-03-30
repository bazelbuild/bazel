// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.Serializable;
import java.util.Objects;

/**
 * The value returned by {@link PrepareDepsOfPatternsFunction}. Because that function is
 * invoked only for its side effect (i.e. ensuring the graph contains targets matching the
 * pattern sequence and their transitive dependencies), this value carries no information.
 */
@Immutable
@ThreadSafe
public final class PrepareDepsOfPatternsValue implements SkyValue {
  public static final PrepareDepsOfPatternsValue INSTANCE = new PrepareDepsOfPatternsValue();

  private PrepareDepsOfPatternsValue() {
  }

  @ThreadSafe
  public static SkyKey key(ImmutableList<String> patterns, FilteringPolicy policy,
      String offset) {
    return new SkyKey(SkyFunctions.PREPARE_DEPS_OF_PATTERNS,
        new TargetPatternSequence(patterns, policy, offset));
  }

  /** The argument value for SkyKeys of {@link PrepareDepsOfPatternsFunction}. */
  @ThreadSafe
  public static class TargetPatternSequence implements Serializable {
    private final ImmutableList<String> patterns;
    private final FilteringPolicy policy;
    private final String offset;

    public TargetPatternSequence(ImmutableList<String> patterns,
        FilteringPolicy policy, String offset) {
      this.patterns = patterns;
      this.policy = policy;
      this.offset = offset;
    }

    public ImmutableList<String> getPatterns() {
      return patterns;
    }

    public FilteringPolicy getPolicy() {
      return policy;
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
      return offset.equals(that.offset) && patterns.equals(that.patterns)
          && policy.equals(that.policy);
    }

    @Override
    public int hashCode() {
      return Objects.hash(patterns, policy, offset);
    }
  }
}
