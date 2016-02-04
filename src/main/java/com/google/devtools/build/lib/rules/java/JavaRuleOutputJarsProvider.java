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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Provides information about jar files produced by a Java rule.
 */
@Immutable
public final class JavaRuleOutputJarsProvider implements TransitiveInfoProvider {

  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(ImmutableList.<OutputJar>of(), null);

  /**
   * A collection of artifacts associated with a jar output.
   */
  public static class OutputJar {
    @Nullable private final Artifact classJar;
    @Nullable private final Artifact iJar;
    @Nullable private final Artifact srcJar;

    public OutputJar(
        @Nullable Artifact classJar, @Nullable Artifact iJar, @Nullable Artifact srcJar) {
      this.classJar = classJar;
      this.iJar = iJar;
      this.srcJar = srcJar;
    }

    @Nullable
    public Artifact getClassJar() {
      return classJar;
    }

    @Nullable
    public Artifact getIJar() {
      return iJar;
    }

    @Nullable
    public Artifact getSrcJar() {
      return srcJar;
    }
  }

  final Iterable<OutputJar> outputJars;
  @Nullable final Artifact jdeps;

  private JavaRuleOutputJarsProvider(Iterable<OutputJar> outputJars,
      @Nullable Artifact jdeps) {
    this.outputJars = outputJars;
    this.jdeps = jdeps;
  }

  public Iterable<OutputJar> getOutputJars() {
    return outputJars;
  }

  @Nullable
  public Artifact getJdeps() {
    return jdeps;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builderassociated.
   */
  public static class Builder {
    ImmutableList.Builder<OutputJar> outputJars = ImmutableList.builder();
    Artifact jdeps;

    public Builder addOutputJar(
        @Nullable Artifact classJar,
        @Nullable Artifact iJar,
        @Nullable Artifact sourceJar) {
      Preconditions.checkState(classJar != null || iJar != null || sourceJar != null);
      outputJars.add(new OutputJar(classJar, iJar, sourceJar));
      return this;
    }

    public Builder addOutputJar(OutputJar outputJar) {
      outputJars.add(outputJar);
      return this;
    }

    public Builder setJdeps(Artifact jdeps) {
      this.jdeps = jdeps;
      return this;
    }

    public JavaRuleOutputJarsProvider build() {
      return new JavaRuleOutputJarsProvider(outputJars.build(), jdeps);
    }
  }
}
