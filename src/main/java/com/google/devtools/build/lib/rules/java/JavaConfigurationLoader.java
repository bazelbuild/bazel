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
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;

/**
 * A loader that creates JavaConfiguration instances based on JavaBuilder configurations and
 * command-line options.
 */
public class JavaConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(JavaOptions.class);
  }


  @Override
  public JavaConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
      throws InvalidConfigurationException, InterruptedException {
    JavaOptions javaOptions = buildOptions.get(JavaOptions.class);
    return create(javaOptions, javaOptions.javaToolchain);
  }

  @Override
  public Class<? extends Fragment> creates() {
    return JavaConfiguration.class;
  }
  
  private JavaConfiguration create(JavaOptions javaOptions, Label javaToolchain)
      throws InvalidConfigurationException {
    boolean generateJavaDeps =
        javaOptions.javaDeps || javaOptions.javaClasspath != JavaClasspathMode.OFF;
    return new JavaConfiguration(generateJavaDeps, javaOptions.jvmOpts, javaOptions, javaToolchain);
  }
}
