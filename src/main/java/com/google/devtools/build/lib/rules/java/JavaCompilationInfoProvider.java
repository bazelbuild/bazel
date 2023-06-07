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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCompilationInfoProviderApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;

/**
 * A class that provides compilation information in Java rules, for perusal of aspects and tools.
 */
@Immutable
public final class JavaCompilationInfoProvider
    implements JavaInfoInternalProvider, JavaCompilationInfoProviderApi<Artifact> {
  private final ImmutableList<String> javacOpts;
  @Nullable private final NestedSet<Artifact> runtimeClasspath;
  @Nullable private final NestedSet<Artifact> compilationClasspath;
  private final BootClassPathInfo bootClasspath;

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Builder for {@link JavaCompilationInfoProvider}. */
  public static class Builder {
    private ImmutableList<String> javacOpts;
    private NestedSet<Artifact> runtimeClasspath;
    private NestedSet<Artifact> compilationClasspath;
    private BootClassPathInfo bootClasspath = BootClassPathInfo.empty();

    @CanIgnoreReturnValue
    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      this.javacOpts = javacOpts;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setRuntimeClasspath(@Nullable NestedSet<Artifact> runtimeClasspath) {
      this.runtimeClasspath = runtimeClasspath;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setCompilationClasspath(@Nullable NestedSet<Artifact> compilationClasspath) {
      this.compilationClasspath = compilationClasspath;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBootClasspath(BootClassPathInfo bootClasspath) {
      this.bootClasspath = Preconditions.checkNotNull(bootClasspath);
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
  @Nullable
  public Depset /*<Artifact>*/ getRuntimeClasspath() {
    return runtimeClasspath == null ? null : Depset.of(Artifact.class, runtimeClasspath);
  }

  @Override
  @Nullable
  public Depset /*<Artifact>*/ getCompilationClasspath() {
    return compilationClasspath == null ? null : Depset.of(Artifact.class, compilationClasspath);
  }

  @Override
  public ImmutableList<Artifact> getBootClasspath() {
    return bootClasspath.bootclasspath().toList();
  }

  public NestedSet<Artifact> getBootClasspathAsNestedSet() {
    return bootClasspath.bootclasspath();
  }

  private JavaCompilationInfoProvider(
      ImmutableList<String> javacOpts,
      @Nullable NestedSet<Artifact> runtimeClasspath,
      @Nullable NestedSet<Artifact> compilationClasspath,
      BootClassPathInfo bootClasspath) {
    this.javacOpts = javacOpts;
    this.runtimeClasspath = runtimeClasspath;
    this.compilationClasspath = compilationClasspath;
    this.bootClasspath = Preconditions.checkNotNull(bootClasspath);
  }
}
