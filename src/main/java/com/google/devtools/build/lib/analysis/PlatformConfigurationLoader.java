// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;

/** A loader that creates {@link PlatformConfiguration} instances based on command-line options. */
public class PlatformConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(PlatformOptions.class);
  }

  @Override
  public PlatformConfiguration create(BuildOptions buildOptions)
      throws InvalidConfigurationException {
    PlatformOptions platformOptions = buildOptions.get(PlatformOptions.class);

    return new PlatformConfiguration(
        platformOptions.computeHostPlatform(),
        ImmutableList.copyOf(platformOptions.extraExecutionPlatforms),
        platformOptions.computeTargetPlatform(),
        ImmutableList.copyOf(platformOptions.extraToolchains),
        platformOptions.targetFilterToAdditionalExecConstraints);
  }

  @Override
  public Class<? extends Fragment> creates() {
    return PlatformConfiguration.class;
  }
}
