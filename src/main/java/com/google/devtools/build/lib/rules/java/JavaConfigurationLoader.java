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
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;

/**
 * A loader that creates JavaConfiguration instances based on JavaBuilder configurations and
 * command-line options.
 */
public class JavaConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    // TODO(bazel-team): either require CppOptions only for dependency trees that use the JAVA_CPU
    // make variable or break out CppConfiguration.getTargetCpu() into its own distinct fragment.
    return ImmutableSet.of(JavaOptions.class, CppOptions.class);
  }


  @Override
  public JavaConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
      throws InvalidConfigurationException {
    CppConfiguration cppConfiguration = env.getFragment(buildOptions, CppConfiguration.class);
    if (cppConfiguration == null) {
      return null;
    }

    JavaOptions javaOptions = buildOptions.get(JavaOptions.class);

    Label javaToolchain = RedirectChaser.followRedirects(env, javaOptions.javaToolchain,
        "java_toolchain");
    return create(javaOptions, javaToolchain, cppConfiguration.getTargetCpu());
  }

  @Override
  public Class<? extends Fragment> creates() {
    return JavaConfiguration.class;
  }
  
  public JavaConfiguration create(JavaOptions javaOptions, Label javaToolchain, String javaCpu)
          throws InvalidConfigurationException {

    boolean generateJavaDeps = javaOptions.javaDeps ||
        javaOptions.experimentalJavaClasspath != JavaClasspathMode.OFF;

    return new JavaConfiguration(
        generateJavaDeps, javaOptions.jvmOpts, javaOptions, javaToolchain, javaCpu);
  }
}
