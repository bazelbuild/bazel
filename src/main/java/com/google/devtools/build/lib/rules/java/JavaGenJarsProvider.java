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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaAnnotationProcessingApi;
import com.google.devtools.build.lib.syntax.Depset;
import java.util.List;
import javax.annotation.Nullable;

/** The collection of gen jars from the transitive closure. */
@Immutable
@AutoCodec
public final class JavaGenJarsProvider
    implements TransitiveInfoProvider, JavaAnnotationProcessingApi<Artifact> {

  private final boolean usesAnnotationProcessing;
  @Nullable private final Artifact genClassJar;
  @Nullable private final Artifact genSourceJar;

  private final NestedSet<Artifact> processorClasspath;
  private final ImmutableList<String> processorClassNames;

  private final NestedSet<Artifact> transitiveGenClassJars;
  private final NestedSet<Artifact> transitiveGenSourceJars;

  static JavaGenJarsProvider create(
      boolean usesAnnotationProcessing,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar,
      JavaPluginInfoProvider plugins,
      List<JavaGenJarsProvider> transitiveJavaGenJars) {
    NestedSetBuilder<Artifact> classJarsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> sourceJarsBuilder = NestedSetBuilder.stableOrder();

    if (genClassJar != null) {
      classJarsBuilder.add(genClassJar);
    }
    if (genSourceJar != null) {
      sourceJarsBuilder.add(genSourceJar);
    }
    for (JavaGenJarsProvider dep : transitiveJavaGenJars) {
      classJarsBuilder.addTransitive(dep.getTransitiveGenClassJars());
      sourceJarsBuilder.addTransitive(dep.getTransitiveGenSourceJars());
    }
    return new JavaGenJarsProvider(
        usesAnnotationProcessing,
        genClassJar,
        genSourceJar,
        plugins.plugins().processorClasspath(),
        plugins.plugins().processorClasses().toList(),
        classJarsBuilder.build(),
        sourceJarsBuilder.build());
  }

  // Package-private for @AutoCodec
  JavaGenJarsProvider(
      boolean usesAnnotationProcessing,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar,
      NestedSet<Artifact> processorClasspath,
      ImmutableList<String> processorClassNames,
      NestedSet<Artifact> transitiveGenClassJars,
      NestedSet<Artifact> transitiveGenSourceJars) {
    this.usesAnnotationProcessing = usesAnnotationProcessing;
    this.genClassJar = genClassJar;
    this.genSourceJar = genSourceJar;
    this.processorClasspath = processorClasspath;
    this.processorClassNames = processorClassNames;
    this.transitiveGenClassJars = transitiveGenClassJars;
    this.transitiveGenSourceJars = transitiveGenSourceJars;
  }

  @Override
  public boolean usesAnnotationProcessing() {
    return usesAnnotationProcessing;
  }

  @Override
  @Nullable
  public Artifact getGenClassJar() {
    return genClassJar;
  }

  @Override
  @Nullable
  public Artifact getGenSourceJar() {
    return genSourceJar;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveGenClassJarsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveGenClassJars);
  }

  NestedSet<Artifact> getTransitiveGenClassJars() {
    return transitiveGenClassJars;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveGenSourceJarsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveGenSourceJars);
  }

  NestedSet<Artifact> getTransitiveGenSourceJars() {
    return transitiveGenSourceJars;
  }

  @Override
  public Depset /*<Artifact>*/ getProcessorClasspathForStarlark() {
    return Depset.of(Artifact.TYPE, processorClasspath);
  }

  NestedSet<Artifact> getProcessorClasspath() {
    return processorClasspath;
  }

  @Override
  public ImmutableList<String> getProcessorClassNames() {
    return processorClassNames;
  }
}
