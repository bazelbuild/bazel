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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Loader for C++ configurations.
 */
public class CppConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public Class<? extends Fragment> creates() {
    return CppConfiguration.class;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.of(CppOptions.class);
  }

  private final CpuTransformer cpuTransformer;

  /**
   * Creates a new CrosstoolConfigurationLoader instance with the given configuration provider. The
   * configuration provider is used to perform caller-specific configuration file lookup.
   */
  public CppConfigurationLoader(CpuTransformer cpuTransformer) {
    this.cpuTransformer = cpuTransformer;
  }

  @Override
  public CppConfiguration create(ConfigurationEnvironment env, BuildOptions options)
      throws InvalidConfigurationException, InterruptedException {
    CppOptions cppOptions = options.get(CppOptions.class);
    // To be deleted soon, keeping there only to split a change removing package loading from
    // CppConfiguration into 2, one is user invisible, second one is user visible (redirect chaser
    // doesn't respect visibility, so after removal it can happen that what used to load will
    // not load anymore.
    Label crosstoolTopLabel =
        RedirectChaser.followRedirects(env, cppOptions.crosstoolTop, "crosstool_top");
    if (crosstoolTopLabel == null) {
      return null;
    }
    return CppConfiguration.create(cpuTransformer, crosstoolTopLabel, options);
  }
}
