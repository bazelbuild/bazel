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

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.LegacySkyKey;
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
    return LegacySkyKey.create(
        SkyFunctions.BUILD_CONFIGURATION, new Key(fragments, buildOptions));
  }

  static final class Key implements Serializable {
    private final Set<Class<? extends BuildConfiguration.Fragment>> fragments;
    private final BuildOptions buildOptions;
    private final boolean enableActions;

    Key(Set<Class<? extends BuildConfiguration.Fragment>> fragments,
        BuildOptions buildOptions) {
      this.fragments = fragments;
      this.buildOptions = Preconditions.checkNotNull(buildOptions);
      // Cache this value for quicker access on .equals() / .hashCode(). We don't cache it inside
      // BuildOptions because BuildOptions is mutable, so a cached value there could fall out of
      // date while the BuildOptions is being prepared for this key.
      this.enableActions = buildOptions.enableActions();
    }

    Set<Class<? extends BuildConfiguration.Fragment>> getFragments() {
      return fragments;
    }

    BuildOptions getBuildOptions() {
      return buildOptions;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof Key)) {
        return false;
      }
      Key otherConfig = (Key) o;
      return Objects.equals(fragments, otherConfig.fragments)
          && Objects.equals(buildOptions, otherConfig.buildOptions)
          && enableActions == otherConfig.enableActions;
    }

    @Override
    public int hashCode() {
      return Objects.hash(fragments, buildOptions, enableActions);
    }
  }
}
