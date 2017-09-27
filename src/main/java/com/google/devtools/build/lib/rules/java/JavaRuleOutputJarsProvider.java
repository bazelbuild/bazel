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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Provides information about jar files produced by a Java rule. */
@Immutable
@SkylarkModule(
  name = "java_output_jars",
  category = SkylarkModuleCategory.NONE,
  doc = "Information about outputs of a Java rule."
)
public final class JavaRuleOutputJarsProvider implements TransitiveInfoProvider {

  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(ImmutableList.<OutputJar>of(), null);

  /** A collection of artifacts associated with a jar output. */
  @SkylarkModule(
    name = "java_output",
    category = SkylarkModuleCategory.NONE,
    doc = "Java classes jar, together with their associated source and interface archives."
  )
  @Immutable
  public static class OutputJar {
    @Nullable private final Artifact classJar;
    @Nullable private final Artifact iJar;
    @Nullable private final ImmutableList<Artifact> srcJars;

    public OutputJar(
        @Nullable Artifact classJar,
        @Nullable Artifact iJar,
        @Nullable Iterable<Artifact> srcJars) {
      this.classJar = classJar;
      this.iJar = iJar;
      this.srcJars = ImmutableList.copyOf(srcJars);
    }

    @Nullable
    @SkylarkCallable(
      name = "class_jar",
      doc = "A classes jar file.",
      allowReturnNones = true,
      structField = true
    )
    public Artifact getClassJar() {
      return classJar;
    }

    @Nullable
    @SkylarkCallable(
      name = "ijar",
      doc = "A interface jar file.",
      allowReturnNones = true,
      structField = true
    )
    public Artifact getIJar() {
      return iJar;
    }

    @Nullable
    @SkylarkCallable(
      name = "source_jar",
      doc = "A sources archive file. Deprecated. Kept for migration reasons. "
          + "Please use source_jars instead.",
      allowReturnNones = true,
      structField = true
    )
    @Deprecated
    public Artifact getSrcJar() {
      return Iterables.getOnlyElement(srcJars, null);
    }

    @Nullable
    @SkylarkCallable(
      name = "source_jars",
      doc = "A list of sources archive files.",
      allowReturnNones = true,
      structField = true
    )
    public SkylarkList<Artifact> getSrcJarsSkylark() {
      return SkylarkList.createImmutable(srcJars);
    }

    public Iterable<Artifact> getSrcJars() {
      return srcJars;
    }
  }

  final ImmutableList<OutputJar> outputJars;
  @Nullable final Artifact jdeps;

  private JavaRuleOutputJarsProvider(ImmutableList<OutputJar> outputJars,
      @Nullable Artifact jdeps) {
    this.outputJars = outputJars;
    this.jdeps = jdeps;
  }

  @SkylarkCallable(name = "jars", doc = "A list of jars the rule outputs.", structField = true)
  public ImmutableList<OutputJar> getOutputJars() {
    return outputJars;
  }

  /**
   * Collects all class output jars from {@link #outputJars}
   */
  public Iterable<Artifact> getAllClassOutputJars() {
    return outputJars.stream().map(OutputJar::getClassJar).collect(Collectors.toList());
  }

  /**
   * Collects all source output jars from {@link #outputJars}
   */
  public Iterable<Artifact> getAllSrcOutputJars() {
    return outputJars
        .stream()
        .map(OutputJar::getSrcJars)
        .reduce(ImmutableList.of(), Iterables::concat);
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
        @Nullable ImmutableList<Artifact> sourceJars) {
      Preconditions.checkState(classJar != null || iJar != null || !sourceJars.isEmpty());
      outputJars.add(new OutputJar(classJar, iJar, sourceJars));
      return this;
    }

    public Builder addOutputJars(Iterable<OutputJar> outputJars) {
      this.outputJars.addAll(outputJars);
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
