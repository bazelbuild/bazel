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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A class that provides compilation information in Java rules, for perusal of aspects and tools.
 */
@SkylarkModule(
  name = "JavaCompilationInfo",
  doc = "Provides access to compilation information for Java rules"
)
@Immutable
public class JavaCompilationInfoProvider implements TransitiveInfoProvider {
  private final ImmutableList<String> javacOpts;
  private final NestedSet<Artifact> runtimeClasspath;
  private final NestedSet<Artifact> compilationClasspath;
  private final ImmutableList<Artifact> bootClasspath;

  /**
   * Builder for {@link JavaCompilationInfoProvider}.
   */
  public static class Builder {
    private ImmutableList<String> javacOpts;
    private NestedSet<Artifact> runtimeClasspath;
    private NestedSet<Artifact> compilationClasspath;
    private ImmutableList<Artifact> bootClasspath;

    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      this.javacOpts = javacOpts;
      return this;
    }

    public Builder setRuntimeClasspath(NestedSet<Artifact> runtimeClasspath) {
      this.runtimeClasspath = runtimeClasspath;
      return this;
    }

    public Builder setCompilationClasspath(NestedSet<Artifact> compilationClasspath) {
      this.compilationClasspath = compilationClasspath;
      return this;
    }

    public Builder setBootClasspath(ImmutableList<Artifact> bootClasspath) {
      this.bootClasspath = bootClasspath;
      return this;
    }

    public JavaCompilationInfoProvider build() {
      return new JavaCompilationInfoProvider(
          javacOpts, runtimeClasspath, compilationClasspath, bootClasspath);
    }
  }

  @SkylarkCallable(name = "javac_options", structField = true, doc = "Options to java compiler")
  public ImmutableList<String> getJavacOpts() {
    return javacOpts;
  }

  @SkylarkCallable(
    name = "runtime_classpath",
    structField = true,
    doc = "Run-time classpath for this Java target"
  )
  public NestedSet<Artifact> getRuntimeClasspath() {
    return runtimeClasspath;
  }

  @SkylarkCallable(
    name = "compilation_classpath",
    structField = true,
    doc = "Compilation classpath for this Java target"
  )
  public NestedSet<Artifact> getCompilationClasspath() {
    return compilationClasspath;
  }

  @SkylarkCallable(
    name = "boot_classpath",
    structField = true,
    doc = "Boot classpath for this Java target"
  )
  public ImmutableList<Artifact> getBootClasspath() {
    return bootClasspath;
  }

  private JavaCompilationInfoProvider(
      ImmutableList<String> javacOpts,
      NestedSet<Artifact> runtimeClasspath,
      NestedSet<Artifact> compileTimeClasspath,
      ImmutableList<Artifact> bootClasspath) {
    this.javacOpts = javacOpts;
    this.runtimeClasspath = runtimeClasspath;
    this.compilationClasspath = compileTimeClasspath;
    this.bootClasspath = bootClasspath;
  }
}
