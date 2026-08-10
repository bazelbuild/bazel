// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;

/**
 * SkyValue returned by {@link BuildOptionsScopeFunction}.
 *
 * @param starlarkFlags the Starlark flags this value answers the scoping question for, i.e. the
 *     labels in the {@link Key} it was computed from.
 * @param projectScopes the {@link Scope} of each flag in {@code starlarkFlags} that has {@link
 *     Scope.ScopeType#PROJECT} scope. Flags with any other scope type are absent: scoping only ever
 *     resets project-scoped flags, so no consumer needs their scope.
 */
@AutoCodec
public record BuildOptionsScopeValue(
    ImmutableSet<Label> starlarkFlags, ImmutableMap<Label, Scope> projectScopes)
    implements SkyValue {

  /** Answers the scoping question for no flag at all. */
  public static final BuildOptionsScopeValue EMPTY =
      new BuildOptionsScopeValue(ImmutableSet.of(), ImmutableMap.of());

  /**
   * Returns whether this value answers the scoping question for all of {@code flags}.
   *
   * <p>Callers that hold a value computed for a superset of the flags they care about can use it
   * instead of asking Skyframe for one computed for the exact set.
   */
  public boolean covers(Set<Label> flags) {
    return starlarkFlags.containsAll(flags);
  }

  /** Key for {@link BuildOptionsScopeValue}. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();
    private final ImmutableSet<Label> starlarkOptionLabels;
    private final int hashCode;

    public Key(ImmutableSet<Label> starlarkOptionLabels) {
      this.starlarkOptionLabels = starlarkOptionLabels;
      this.hashCode = starlarkOptionLabels.hashCode();
    }

    public static Key create(ImmutableSet<Label> starlarkOptionLabels) {
      return interner.intern(new Key(starlarkOptionLabels));
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_OPTIONS_SCOPE;
    }

    public ImmutableSet<Label> starlarkOptionLabels() {
      return starlarkOptionLabels;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      return obj instanceof Key other && starlarkOptionLabels.equals(other.starlarkOptionLabels);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public String toString() {
      return "Key[starlarkOptionLabels=%s]".formatted(starlarkOptionLabels);
    }
  }
}
