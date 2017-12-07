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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.util.Objects;

/**
 * Contains parameters which uniquely describe a crosstool configuration
 * and methods for comparing two crosstools against each other.
 *
 * <p>Two crosstools which contain equivalent values of these parameters are
 * considered equal.
 */
public final class CrosstoolConfigurationIdentifier implements CrosstoolConfigurationOptions {
  /** The CPU associated with this crosstool configuration. */
  private final String cpu;

  /** The compiler (e.g. gcc) associated with this crosstool configuration. */
  private final String compiler;

  /** The version of libc (e.g. glibc-2.11) associated with this crosstool configuration. */
  private final String libc;

  /** Creates a new {@link CrosstoolConfigurationIdentifier} with the given parameters. */
  CrosstoolConfigurationIdentifier(String cpu, String compiler, String libc) {
    this.cpu = Preconditions.checkNotNull(cpu);
    this.compiler = compiler;
    this.libc = libc;
  }

  /**
   * Creates a new crosstool configuration from the given crosstool release and
   * configuration options.
   */
  public static CrosstoolConfigurationIdentifier fromOptions(BuildOptions buildOptions) {
    Options options = buildOptions.get(BuildConfiguration.Options.class);
    CppOptions cppOptions = buildOptions.get(CppOptions.class);
    return new CrosstoolConfigurationIdentifier(
        options.cpu, cppOptions.cppCompiler, cppOptions.glibc);
  }

  public static CrosstoolConfigurationIdentifier fromToolchain(CToolchain toolchain) {
    return new CrosstoolConfigurationIdentifier(
        toolchain.getTargetCpu(), toolchain.getCompiler(), toolchain.getTargetLibc());
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof CrosstoolConfigurationIdentifier)) {
      return false;
    }
    CrosstoolConfigurationIdentifier otherCrosstool = (CrosstoolConfigurationIdentifier) other;
    return Objects.equals(cpu, otherCrosstool.cpu)
        && Objects.equals(compiler, otherCrosstool.compiler)
        && Objects.equals(libc, otherCrosstool.libc);
  }

  @Override
  public int hashCode() {
    return Objects.hash(cpu, compiler, libc);
  }


  /**
   * Returns a series of command line flags which specify the configuration options.
   * Any of these options may be null, in which case its flag is omitted.
   *
   * <p>The appended string will be along the lines of
   * " --cpu='cpu' --compiler='compiler' --glibc='libc'".
   */
  public String describeFlags() {
    StringBuilder message = new StringBuilder();
    if (getCpu() != null) {
      message.append(" --cpu='").append(getCpu()).append("'");
    }
    if (getCompiler() != null) {
      message.append(" --compiler='").append(getCompiler()).append("'");
    }
    if (getLibc() != null) {
      message.append(" --glibc='").append(getLibc()).append("'");
    }
    return message.toString();
  }

  /** Returns true if the specified toolchain is a candidate for use with this crosstool. */
  public boolean isCandidateToolchain(CToolchain toolchain) {
    return (toolchain.getTargetCpu().equals(getCpu())
        && (getLibc() == null || toolchain.getTargetLibc().equals(getLibc()))
        && (getCompiler() == null || toolchain.getCompiler().equals(
            getCompiler())));
  }

  @Override
  public String toString() {
    return describeFlags();
  }

  @Override
  public String getCpu() {
    return cpu;
  }

  @Override
  public String getCompiler() {
    return compiler;
  }

  @Override
  public String getLibc() {
    return libc;
  }
}
