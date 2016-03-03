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


import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.Serializable;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A Skyframe node representing a build configuration fragment.
 */
@Immutable
@ThreadSafe
public class ConfigurationFragmentValue implements SkyValue {

  @Nullable
  private final BuildConfiguration.Fragment fragment;

  ConfigurationFragmentValue(BuildConfiguration.Fragment fragment) {
    this.fragment = fragment;
  }

  public BuildConfiguration.Fragment getFragment() {
    return fragment;
  }

  @ThreadSafe
  public static SkyKey key(BuildOptions buildOptions, Class<? extends Fragment> fragmentType,
      RuleClassProvider ruleClassProvider) {
    BuildOptions optionsKey =
        buildOptions.trim(
            BuildConfiguration.getOptionsClasses(
                ImmutableList.<Class<? extends BuildConfiguration.Fragment>>of(fragmentType),
                ruleClassProvider));
    return SkyKey.create(
        SkyFunctions.CONFIGURATION_FRAGMENT,
        new ConfigurationFragmentKey(optionsKey, fragmentType));
  }

  static final class ConfigurationFragmentKey implements Serializable {
    private final BuildOptions buildOptions;
    private final Class<? extends Fragment> fragmentType;

    public ConfigurationFragmentKey(BuildOptions buildOptions,
        Class<? extends Fragment> fragmentType) {
      this.buildOptions = Preconditions.checkNotNull(buildOptions);
      this.fragmentType = Preconditions.checkNotNull(fragmentType);
    }

    public BuildOptions getBuildOptions() {
      return buildOptions;
    }

    public Class<? extends Fragment> getFragmentType() {
      return fragmentType;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof ConfigurationFragmentKey)) {
        return false;
      }
      ConfigurationFragmentKey confObject = (ConfigurationFragmentKey) o;
      return Objects.equals(fragmentType, confObject.fragmentType)
          && Objects.equals(buildOptions, confObject.buildOptions);
    }

    @Override
    public int hashCode() {
      return Objects.hash(buildOptions, fragmentType);
    }
  }
}
