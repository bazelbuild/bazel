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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCompilationInfoProviderApi;

/**
 * A class that provides compilation information in Java rules, for perusal of aspects and tools.
 */
@Immutable
@AutoCodec
public class JavaCompilationInfoProvider
    implements TransitiveInfoProvider, JavaCompilationInfoProviderApi<Artifact> {
  private final ImmutableList<String> javacOpts;
  private final NestedSet<Artifact> runtimeClasspath;
  private final NestedSet<Artifact> compilationClasspath;
  private final ImmutableList<Artifact> bootClasspath;

  /** Builder for {@link JavaCompilationInfoProvider}. */
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

  @Override
  public ImmutableList<String> getJavacOpts() {
    return javacOpts;
  }

  @Override
  public NestedSet<Artifact> getRuntimeClasspath() {
    return runtimeClasspath;
  }

  @Override
  public NestedSet<Artifact> getCompilationClasspath() {
    return compilationClasspath;
  }

  @Override
  public ImmutableList<Artifact> getBootClasspath() {
    return bootClasspath;
  }

  @AutoCodec.VisibleForSerialization
  JavaCompilationInfoProvider(
      ImmutableList<String> javacOpts,
      NestedSet<Artifact> runtimeClasspath,
      NestedSet<Artifact> compilationClasspath,
      ImmutableList<Artifact> bootClasspath) {
    this.javacOpts = javacOpts;
    this.runtimeClasspath = runtimeClasspath;
    this.compilationClasspath = compilationClasspath;
    this.bootClasspath = bootClasspath;
  }
}
