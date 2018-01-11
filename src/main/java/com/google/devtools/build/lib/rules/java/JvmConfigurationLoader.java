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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;

/**
 * A provider to load jvm configurations from the package path.
 *
 * <p>The {@code --javabase} option should point to a {@code java_runtime_suite} or a
 * {@code java_runtime} rule. If it's the former, the actual runtime is selected from the ones
 * listed in the {@code java_runtime_suite.runtimes} and the  {@code java_runtime_suite.default}
 * attributes based on the current CPU, if it's the latter, that runtime will be used as it is.
 */
public final class JvmConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public Jvm create(ConfigurationEnvironment env, BuildOptions buildOptions) {
    return new Jvm(buildOptions.get(JavaOptions.class).javaBase);
  }

  @Override
  public Class<? extends Fragment> creates() {
    return Jvm.class;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(JavaOptions.class);
  }
}
