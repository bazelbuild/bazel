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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.Serializable;
import java.util.Objects;
import java.util.Set;

/**
 * A Skyframe value representing a {@link BuildConfiguration}.
 */
// TODO(bazel-team): mark this immutable when BuildConfiguration is immutable.
// @Immutable
@ThreadSafe
public class BuildConfigurationValue implements SkyValue {
  private static final Interner<Key> keyInterner = BlazeInterners.newWeakInterner();

  private final BuildConfiguration configuration;

  BuildConfigurationValue(BuildConfiguration configuration) {
    this.configuration = configuration;
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  /**
   * Returns the key for a requested configuration.
   *
   * @param fragments the fragments the configuration should contain
   * @param buildOptions the build options the fragments should be built from
   */
  @ThreadSafe
  public static SkyKey key(Set<Class<? extends BuildConfiguration.Fragment>> fragments,
      BuildOptions buildOptions) {
    return keyInterner.intern(
        new Key(
            ImmutableSortedSet.copyOf(BuildConfiguration.lexicalFragmentSorter, fragments),
            buildOptions));
  }

  static final class Key implements SkyKey, Serializable {
    private final ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> fragments;
    private final BuildOptions buildOptions;
    private final boolean enableActions;
    // If hashCode really is -1, we'll recompute it from scratch each time. Oh well.
    private volatile int hashCode = -1;

    Key(ImmutableSortedSet<Class<? extends Fragment>> fragments, BuildOptions buildOptions) {
      this.fragments = fragments;
      this.buildOptions = Preconditions.checkNotNull(buildOptions);
      // Cache this value for quicker access on .equals() / .hashCode(). We don't cache it inside
      // BuildOptions because BuildOptions is mutable, so a cached value there could fall out of
      // date while the BuildOptions is being prepared for this key.
      this.enableActions = buildOptions.enableActions();
    }

    ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> getFragments() {
      return fragments;
    }

    BuildOptions getBuildOptions() {
      return buildOptions;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_CONFIGURATION;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Key)) {
        return false;
      }
      Key otherConfig = (Key) o;
      return buildOptions.equals(otherConfig.buildOptions)
          && Objects.equals(fragments, otherConfig.fragments)
          && enableActions == otherConfig.enableActions;
    }

    @Override
    public int hashCode() {
      if (hashCode == -1) {
        hashCode = Objects.hash(fragments, buildOptions, enableActions);
      }
      return hashCode;
    }
  }
}
