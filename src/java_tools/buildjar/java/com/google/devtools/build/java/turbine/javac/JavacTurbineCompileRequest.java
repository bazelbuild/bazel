// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;

import java.nio.file.Path;

import javax.annotation.Nullable;

/** The input to a {@link JavacTurbineCompiler} compilation. */
class JavacTurbineCompileRequest {

  private final ImmutableList<Path> classPath;
  private final ImmutableList<Path> bootClassPath;
  private final ImmutableList<Path> processorClassPath;
  private final ImmutableList<String> javacOptions;
  @Nullable private final StrictJavaDepsPlugin strictJavaDepsPlugin;

  JavacTurbineCompileRequest(
      ImmutableList<Path> classPath,
      ImmutableList<Path> bootClassPath,
      ImmutableList<Path> processorClassPath,
      ImmutableList<String> javacOptions,
      @Nullable StrictJavaDepsPlugin strictJavaDepsPlugin) {
    this.classPath = classPath;
    this.bootClassPath = bootClassPath;
    this.processorClassPath = processorClassPath;
    this.javacOptions = javacOptions;
    this.strictJavaDepsPlugin = strictJavaDepsPlugin;
  }

  /** The class path; correspond's to javac -classpath. */
  ImmutableList<Path> classPath() {
    return classPath;
  }

  /** The boot class path; corresponds to javac -bootclasspath. */
  ImmutableList<Path> bootClassPath() {
    return bootClassPath;
  }

  /** The class path to search for processors; corresponds to javac -processorpath. */
  ImmutableList<Path> processorClassPath() {
    return processorClassPath;
  }

  /** Miscellaneous javac options. */
  ImmutableList<String> javacOptions() {
    return javacOptions;
  }

  /**
   * The build's {@link StrictJavaDepsPlugin}, or {@code null} if Strict Java Deps is not enabled.
   */
  @Nullable
  StrictJavaDepsPlugin strictJavaDepsPlugin() {
    return strictJavaDepsPlugin;
  }

  static JavacTurbineCompileRequest.Builder builder() {
    return new Builder();
  }

  static class Builder {
    private ImmutableList<Path> classPath;
    private ImmutableList<Path> bootClassPath;
    private ImmutableList<Path> processorClassPath;
    private ImmutableList<String> javacOptions;
    @Nullable private StrictJavaDepsPlugin strictDepsPlugin;

    private Builder() {}

    JavacTurbineCompileRequest build() {
      return new JavacTurbineCompileRequest(
          classPath, bootClassPath, processorClassPath, javacOptions, strictDepsPlugin);
    }

    Builder setClassPath(ImmutableList<Path> classPath) {
      this.classPath = classPath;
      return this;
    }

    Builder setBootClassPath(ImmutableList<Path> bootClassPath) {
      this.bootClassPath = bootClassPath;
      return this;
    }

    Builder setProcessorClassPath(ImmutableList<Path> processorClassPath) {
      this.processorClassPath = processorClassPath;
      return this;
    }

    Builder setJavacOptions(ImmutableList<String> javacOptions) {
      this.javacOptions = javacOptions;
      return this;
    }

    Builder setStrictDepsPlugin(@Nullable StrictJavaDepsPlugin strictDepsPlugin) {
      this.strictDepsPlugin = strictDepsPlugin;
      return this;
    }
  }
}
