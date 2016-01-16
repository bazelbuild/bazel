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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.LinkedHashSet;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A collection of artifacts for java compilations. It concisely describes the
 * outputs of a java-related rule, with runtime jars, compile-time jars,
 * unfiltered compile-time jars (these are run through ijar if they are
 * dependent upon by another target), source ijars, and instrumentation
 * manifests. Not all rules generate all kinds of artifacts. Each java-related
 * rule should add both a runtime jar and either a compile-time jar or an
 * unfiltered compile-time jar.
 *
 * <p>An instance of this class only collects the data for the current target,
 * not for the transitive closure of targets, so these still need to be
 * collected using some other mechanism, such as the {@link
 * JavaCompilationArgsProvider}.
 */
@Immutable
public final class JavaCompilationArtifacts {

  public static final JavaCompilationArtifacts EMPTY = new Builder().build();

  private final ImmutableList<Artifact> runtimeJars;
  private final ImmutableList<Artifact> compileTimeJars;
  private final ImmutableList<Artifact> instrumentationMetadata;
  private final Artifact compileTimeDependencyArtifact;
  private final Artifact instrumentedJar;

  private JavaCompilationArtifacts(
      ImmutableList<Artifact> runtimeJars,
      ImmutableList<Artifact> compileTimeJars,
      ImmutableList<Artifact> instrumentationMetadata,
      Artifact compileTimeDependencyArtifact,
      Artifact instrumentedJar) {
    this.runtimeJars = runtimeJars;
    this.compileTimeJars = compileTimeJars;
    this.instrumentationMetadata = instrumentationMetadata;
    this.compileTimeDependencyArtifact = compileTimeDependencyArtifact;
    this.instrumentedJar = instrumentedJar;
  }

  public ImmutableList<Artifact> getRuntimeJars() {
    return runtimeJars;
  }

  public ImmutableList<Artifact> getCompileTimeJars() {
    return compileTimeJars;
  }

  public ImmutableList<Artifact> getInstrumentationMetadata() {
    return instrumentationMetadata;
  }

  public Artifact getCompileTimeDependencyArtifact() {
    return compileTimeDependencyArtifact;
  }

  public Artifact getInstrumentedJar() {
    return instrumentedJar;
  }

  /**
   * A builder for {@link JavaCompilationArtifacts}.
   */
  public static final class Builder {
    private final Set<Artifact> runtimeJars = new LinkedHashSet<>();
    private final Set<Artifact> compileTimeJars = new LinkedHashSet<>();
    private final Set<Artifact> instrumentationMetadata = new LinkedHashSet<>();
    private Artifact compileTimeDependencies;
    private Artifact instrumentedJar;

    public JavaCompilationArtifacts build() {
      return new JavaCompilationArtifacts(
          ImmutableList.copyOf(runtimeJars),
          ImmutableList.copyOf(compileTimeJars),
          ImmutableList.copyOf(instrumentationMetadata),
          compileTimeDependencies,
          instrumentedJar);
    }

    public Builder addRuntimeJar(Artifact jar) {
      this.runtimeJars.add(jar);
      return this;
    }

    public Builder addRuntimeJars(Iterable<Artifact> jars) {
      Iterables.addAll(this.runtimeJars, jars);
      return this;
    }

    public Builder addCompileTimeJar(Artifact jar) {
      this.compileTimeJars.add(jar);
      return this;
    }

    public Builder addCompileTimeJars(Iterable<Artifact> jars) {
      Iterables.addAll(this.compileTimeJars, jars);
      return this;
    }

    public Builder addInstrumentationMetadata(Artifact instrumentationMetadata) {
      this.instrumentationMetadata.add(instrumentationMetadata);
      return this;
    }

    public Builder setCompileTimeDependencies(@Nullable Artifact compileTimeDependencies) {
      this.compileTimeDependencies = compileTimeDependencies;
      return this;
    }

    public Builder setInstrumentedJar(@Nullable Artifact instrumentedJar) {
      this.instrumentedJar = instrumentedJar;
      return this;
    }
  }
}
