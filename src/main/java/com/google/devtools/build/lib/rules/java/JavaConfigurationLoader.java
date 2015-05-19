// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.syntax.Label;

/**
 * A loader that creates JavaConfiguration instances based on JavaBuilder configurations and
 * command-line options.
 */
public class JavaConfigurationLoader implements ConfigurationFragmentFactory {
  private final JavaCpuSupplier cpuSupplier;

  public JavaConfigurationLoader(JavaCpuSupplier cpuSupplier) {
    this.cpuSupplier = cpuSupplier;
  }

  @Override
  public JavaConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
      throws InvalidConfigurationException {
    JavaOptions javaOptions = buildOptions.get(JavaOptions.class);

    Label javaToolchain = RedirectChaser.followRedirects(env, javaOptions.javaToolchain,
        "java_toolchain");
    return create(javaOptions, javaToolchain, cpuSupplier.getJavaCpu(buildOptions, env));
  }

  @Override
  public Class<? extends Fragment> creates() {
    return JavaConfiguration.class;
  }
  
  public JavaConfiguration create(JavaOptions javaOptions, Label javaToolchain, String javaCpu)
          throws InvalidConfigurationException {

    boolean generateJavaDeps = javaOptions.javaDeps ||
        javaOptions.experimentalJavaClasspath != JavaClasspathMode.OFF;

    ImmutableList<String> defaultJavaBuilderJvmOpts =
        ImmutableList.copyOf(JavaHelper.tokenizeJavaOptions(javaOptions.javaBuilderJvmOpts));

    return new JavaConfiguration(generateJavaDeps, javaOptions.jvmOpts, javaOptions,
        javaToolchain, javaCpu, defaultJavaBuilderJvmOpts);
  }
}
