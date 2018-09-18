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

  /** Creates a new {@link CrosstoolConfigurationIdentifier} with the given parameters. */
  CrosstoolConfigurationIdentifier(String cpu, String compiler) {
    this.cpu = Preconditions.checkNotNull(cpu);
    this.compiler = compiler;
  }

  public static CrosstoolConfigurationIdentifier fromToolchain(CToolchain toolchain) {
    return new CrosstoolConfigurationIdentifier(toolchain.getTargetCpu(), toolchain.getCompiler());
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof CrosstoolConfigurationIdentifier)) {
      return false;
    }
    CrosstoolConfigurationIdentifier otherCrosstool = (CrosstoolConfigurationIdentifier) other;
    return Objects.equals(cpu, otherCrosstool.cpu)
        && Objects.equals(compiler, otherCrosstool.compiler);
  }

  @Override
  public int hashCode() {
    return Objects.hash(cpu, compiler);
  }

  /**
   * Returns a series of command line flags which specify the configuration options. Any of these
   * options may be null, in which case its flag is omitted.
   *
   * <p>The appended string will be along the lines of " --cpu='cpu' --compiler='compiler'".
   */
  public String describeFlags() {
    StringBuilder message = new StringBuilder();
    if (getCpu() != null) {
      message.append(" --cpu='").append(getCpu()).append("'");
    }
    if (getCompiler() != null) {
      message.append(" --compiler='").append(getCompiler()).append("'");
    }
    return message.toString();
  }

  /** Returns true if the specified toolchain is a candidate for use with this crosstool. */
  public boolean isCandidateToolchain(CToolchain toolchain) {
    return (toolchain.getTargetCpu().equals(getCpu())
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
}
