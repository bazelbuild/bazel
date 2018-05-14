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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/**
 * Grab bag file for test configuration fragments and fragment factories.
 */
public class TestConfigFragments {
  /**
   * A {@link PatchTransition} that appends a given value to {@link
   * BuildConfiguration.Options#hostCpu}.
   */
  @AutoCodec
  @VisibleForSerialization
  static class HostCpuTransition implements PatchTransition {

    private final String patchMessage;

    HostCpuTransition(String patchMessage) {
      this.patchMessage = patchMessage;
    }

    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions toOptions = options.clone();
      BuildConfiguration.Options coreOptions =
          toOptions.get(BuildConfiguration.Options.class);
      String prefix = coreOptions.hostCpu.startsWith("$") ? coreOptions.hostCpu : "";
      coreOptions.hostCpu = prefix + "$" + patchMessage;
      return toOptions;
    }
  }

  /**
   * A {@link ConfigurationFragmentFactory} that trivially returns a given fragment.
   */
  private static class SimpleFragmentFactory implements ConfigurationFragmentFactory {
    private final BuildConfiguration.Fragment fragment;

    public SimpleFragmentFactory(BuildConfiguration.Fragment fragment) {
      this.fragment = fragment;
    }

    @Override
    public BuildConfiguration.Fragment create(BuildOptions buildOptions) {
      return fragment;
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return fragment.getClass();
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of();
    }
  }

  @AutoCodec
  static class Hook1Fragment extends BuildConfiguration.Fragment {
    @Override
    public PatchTransition topLevelConfigurationHook(Target toTarget) {
      return new HostCpuTransition("CONFIG HOOK 1");
    }
  }

  /** Factory for a test fragment with a top-level configuration hook. */
  public static SimpleFragmentFactory FragmentWithTopLevelConfigHook1Factory =
      new SimpleFragmentFactory(new Hook1Fragment());

  /**
   * The class definition for the BuildConfiguration.Fragment needs to be different than the one of
   * its peer above. This is because Bazel indexes configuration fragments by class name. So we need
   * to make sure all fragment definitions retain distinct class names.
   */
  @AutoCodec
  static class Hook2Fragment extends BuildConfiguration.Fragment {
    @Override
    public PatchTransition topLevelConfigurationHook(Target toTarget) {
      return new HostCpuTransition("CONFIG HOOK 2");
    }
  }

  /** Factory for a test fragment with a top-level configuration hook. */
  public static SimpleFragmentFactory FragmentWithTopLevelConfigHook2Factory =
      new SimpleFragmentFactory(new Hook2Fragment());
}
