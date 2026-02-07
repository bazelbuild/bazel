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

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;

/** SkyValue returned by {@link BuildOptionsScopeFunction}. */
public final class BuildOptionsScopeValue implements SkyValue {

  private final LinkedHashMap<Label, Scope> fullyResolvedScopes;

  /** Key for {@link BuildOptionsScopeValue}. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();
    private final BuildOptions buildOptions;
    private final List<Label> flagsNeedingScopeLookup;

    public Key(BuildOptions buildOptions, List<Label> flagsNeedingScopeLookup) {
      this.buildOptions = buildOptions;
      this.flagsNeedingScopeLookup = flagsNeedingScopeLookup;
    }

    public static Key create(BuildOptions buildOptions, List<Label> flagsNeedingScopeLookup) {
      return interner.intern(new Key(buildOptions, flagsNeedingScopeLookup));
    }

    public BuildOptions getBuildOptions() {
      return buildOptions;
    }

    /** Returns the list of flags that need scope resolution via Skyframe. */
    public List<Label> getFlagsWithIncompleteScopeInfo() {
      return flagsNeedingScopeLookup;
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_OPTIONS_SCOPE;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(buildOptions, key.buildOptions)
          && Objects.equals(flagsNeedingScopeLookup, key.flagsNeedingScopeLookup);
    }

    @Override
    public int hashCode() {
      return Objects.hash(buildOptions, flagsNeedingScopeLookup);
    }
  }

  public static BuildOptionsScopeValue create(
      BuildOptions inputBuildOptions,
      List<Label> scopedFlags,
      LinkedHashMap<Label, Scope> fullyResolvedScopes) {
    return new BuildOptionsScopeValue(fullyResolvedScopes);
  }

  public BuildOptionsScopeValue(LinkedHashMap<Label, Scope> fullyResolvedScopes) {
    this.fullyResolvedScopes = fullyResolvedScopes;
  }

  /**
   * Returns the map of {@link Label} of scoped flags to their {@link Scope} including both {@link
   * Scope.ScopeType} and {@link Scope.ScopeDefinition}.
   */
  public LinkedHashMap<Label, Scope> getFullyResolvedScopes() {
    return fullyResolvedScopes;
  }
}
