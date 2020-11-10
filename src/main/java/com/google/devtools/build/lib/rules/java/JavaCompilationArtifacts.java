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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A collection of artifacts for java compilations. It concisely describes the outputs of a
 * java-related rule, with runtime jars, compile-time jars, unfiltered compile-time jars (these are
 * run through ijar if they are dependent upon by another target), source ijars, and instrumentation
 * manifests. Not all rules generate all kinds of artifacts. Each java-related rule should add both
 * a runtime jar and either a compile-time jar or an unfiltered compile-time jar.
 *
 * <p>An instance of this class only collects the data for the current target, not for the
 * transitive closure of targets, so these still need to be collected using some other mechanism,
 * such as the {@link JavaCompilationArgsProvider}.
 */
@AutoCodec
@Immutable
@AutoValue
public abstract class JavaCompilationArtifacts {
  @AutoCodec public static final JavaCompilationArtifacts EMPTY = new Builder().build();

  public abstract ImmutableList<Artifact> getRuntimeJars();

  public abstract ImmutableList<Artifact> getCompileTimeJars();

  abstract ImmutableList<Artifact> getFullCompileTimeJars();

  @Nullable
  public abstract Artifact getCompileTimeDependencyArtifact();

  /** Returns a builder for a {@link JavaCompilationArtifacts}. */
  public static Builder builder() {
    return new Builder();
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static JavaCompilationArtifacts create(
      ImmutableList<Artifact> runtimeJars,
      ImmutableList<Artifact> compileTimeJars,
      ImmutableList<Artifact> fullCompileTimeJars,
      Artifact compileTimeDependencyArtifact) {
    return new AutoValue_JavaCompilationArtifacts(
        ImmutableList.copyOf(runtimeJars),
        ImmutableList.copyOf(compileTimeJars),
        ImmutableList.copyOf(fullCompileTimeJars),
        compileTimeDependencyArtifact);
  }

  /** A builder for {@link JavaCompilationArtifacts}. */
  public static final class Builder {
    private final Set<Artifact> runtimeJars = new LinkedHashSet<>();
    private final Set<Artifact> compileTimeJars = new LinkedHashSet<>();
    private final Set<Artifact> fullCompileTimeJars = new LinkedHashSet<>();
    private Artifact compileTimeDependencies;

    public JavaCompilationArtifacts build() {
      validate();
      return create(
          ImmutableList.copyOf(runtimeJars),
          ImmutableList.copyOf(compileTimeJars),
          ImmutableList.copyOf(fullCompileTimeJars),
          compileTimeDependencies);
    }

    private void validate() {
      checkState(
          fullCompileTimeJars.size() == compileTimeJars.size(),
          "Expected the same number of interface and implementation jars:\n%s\n%s\n",
          compileTimeJars,
          fullCompileTimeJars);
    }

    public Builder addRuntimeJar(Artifact jar) {
      this.runtimeJars.add(jar);
      return this;
    }

    public Builder addRuntimeJars(Iterable<Artifact> jars) {
      Iterables.addAll(this.runtimeJars, jars);
      return this;
    }

    public Builder addInterfaceJarWithFullJar(Artifact ijar, Artifact fullJar) {
      this.compileTimeJars.add(ijar);
      this.fullCompileTimeJars.add(fullJar);
      return this;
    }

    public Builder addCompileTimeJarAsFullJar(Artifact jar) {
      this.compileTimeJars.add(jar);
      this.fullCompileTimeJars.add(jar);
      return this;
    }

    Builder addInterfaceJarsWithFullJars(
        Iterable<Artifact> compileTimeJars, Iterable<Artifact> fullCompileTimeJars) {
      Iterables.addAll(this.compileTimeJars, compileTimeJars);
      Iterables.addAll(this.fullCompileTimeJars, fullCompileTimeJars);
      validate();
      return this;
    }

    public Builder setCompileTimeDependencies(@Nullable Artifact compileTimeDependencies) {
      this.compileTimeDependencies = compileTimeDependencies;
      return this;
    }
  }
}
