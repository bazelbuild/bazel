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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Provides information about jar files produced by a Java rule.
 */
@Immutable
@SkylarkModule(name = "JavaOutputJars", doc = "Information about outputs of a Java rule")
public final class JavaRuleOutputJarsProvider implements TransitiveInfoProvider {

  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(ImmutableList.<OutputJar>of(), null);

  /**
   * A collection of artifacts associated with a jar output.
   */
  @SkylarkModule(
    name = "JavaOutput",
    doc = "Java classes jar, together with their associated source and interface archives"
  )
  @Immutable
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
    @SkylarkCallable(
      name = "class_jar",
      doc = "A classes jar file",
      allowReturnNones = true,
      structField = true
    )
    public Artifact getClassJar() {
      return classJar;
    }

    @Nullable
    @SkylarkCallable(
      name = "ijar",
      doc = "A interface jar file",
      allowReturnNones = true,
      structField = true
    )
    public Artifact getIJar() {
      return iJar;
    }

    @Nullable
    @SkylarkCallable(
      name = "source_jar",
      doc = "A sources archive file",
      allowReturnNones = true,
      structField = true
    )
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

  @SkylarkCallable(name = "jars", doc = "A list of jars the rule outputs.", structField = true)
  public Iterable<OutputJar> getOutputJars() {
    return outputJars;
  }

  @Nullable
  @SkylarkCallable(
    name = "jdeps",
    doc = "The jdeps file for rule outputs.",
    structField = true,
    allowReturnNones = true
  )
  public Artifact getJdeps() {
    return jdeps;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder for {@link JavaRuleOutputJarsProvider}.
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
