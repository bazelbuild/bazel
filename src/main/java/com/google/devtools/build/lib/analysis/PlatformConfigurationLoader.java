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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;

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

    // Handle default values for the host and target platform.
    // TODO(https://github.com/bazelbuild/bazel/issues/6849): After migration, set the defaults
    // directly.
    Label hostPlatform;
    if (platformOptions.hostPlatform != null) {
      hostPlatform = platformOptions.hostPlatform;
    } else if (platformOptions.autoConfigureHostPlatform) {
      // Use the auto-configured host platform.
      hostPlatform = PlatformOptions.DEFAULT_HOST_PLATFORM;
    } else {
      // Use the legacy host platform.
      hostPlatform = PlatformOptions.LEGACY_DEFAULT_HOST_PLATFORM;
    }

    Label targetPlatform;
    if (!platformOptions.platforms.isEmpty()) {
      targetPlatform = Iterables.getFirst(platformOptions.platforms, null);
    } else if (platformOptions.autoConfigureHostPlatform) {
      // Default to the host platform, whatever it is.
      targetPlatform = hostPlatform;
    } else {
      // Use the legacy target platform
      targetPlatform = PlatformOptions.LEGACY_DEFAULT_TARGET_PLATFORM;
    }

    return new PlatformConfiguration(
        hostPlatform,
        ImmutableList.copyOf(platformOptions.extraExecutionPlatforms),
        targetPlatform,
        ImmutableList.copyOf(platformOptions.extraToolchains),
        ImmutableList.copyOf(platformOptions.enabledToolchainTypes));
  }

  @Override
  public Class<? extends BuildConfiguration.Fragment> creates() {
    return PlatformConfiguration.class;
  }
}
