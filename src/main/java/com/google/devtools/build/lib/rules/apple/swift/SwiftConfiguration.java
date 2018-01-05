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

package com.google.devtools.build.lib.rules.apple.swift;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A configuration containing flags required for Swift tools. This is used primarily by swift_*
 * family of rules written in Skylark.
 */
@AutoCodec
@SkylarkModule(
  name = "swift",
  doc = "A configuration fragment for Swift tools.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
@Immutable
public class SwiftConfiguration extends BuildConfiguration.Fragment {
  public static final ObjectCodec<SwiftConfiguration> CODEC = new SwiftConfiguration_AutoCodec();

  private final boolean enableWholeModuleOptimization;
  private final ImmutableList<String> copts;

  public SwiftConfiguration(SwiftCommandLineOptions options) {
    this(options.enableWholeModuleOptimization, ImmutableList.copyOf(options.copts));
  }

  @AutoCodec.Constructor
  SwiftConfiguration(boolean enableWholeModuleOptimization, ImmutableList<String> copts) {
    this.enableWholeModuleOptimization = enableWholeModuleOptimization;
    this.copts = copts;
  }

  /** Returns whether to enable Whole Module Optimization. */
  @SkylarkCallable(
    name = "enable_whole_module_optimization",
    doc = "<code>True</code> if <em>whole module optimization</em> should be enabled when "
        + "compiling Swift code; otherwise, <code>False</code>."
  )
  public boolean enableWholeModuleOptimization() {
    return enableWholeModuleOptimization;
  }

  /** Returns a list of options to use for compiling Swift. */
  @SkylarkCallable(
    name = "copts",
    doc = "A list of compiler options that should be passed to <code>swiftc</code> when compiling "
        + "Swift code.")
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /** Loads {@link SwiftConfiguration} from build options. */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public SwiftConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException, InterruptedException {
      SwiftCommandLineOptions options = buildOptions.get(SwiftCommandLineOptions.class);

      return new SwiftConfiguration(options);
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return SwiftConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(SwiftCommandLineOptions.class);
    }
  }
}
