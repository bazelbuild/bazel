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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/** The collection of gen jars from the transitive closure. */
@Immutable
@SkylarkModule(
  name = "java_annotation_processing",
  category = SkylarkModuleCategory.NONE,
  doc = "Information about jars that are a result of annotation processing for a Java rule."
)
public final class JavaGenJarsProvider implements TransitiveInfoProvider {

  private final boolean usesAnnotationProcessing;
  @Nullable
  private final Artifact genClassJar;
  @Nullable
  private final Artifact genSourceJar;

  private final NestedSet<Artifact> processorClasspath;
  private final ImmutableList<String> processorClassNames;

  private final NestedSet<Artifact> transitiveGenClassJars;
  private final NestedSet<Artifact> transitiveGenSourceJars;

  public JavaGenJarsProvider(
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

  @SkylarkCallable(
    name = "enabled",
    structField = true,
    doc = "Returns true if the Java rule uses annotation processing."
  )
  public boolean usesAnnotationProcessing() {
    return usesAnnotationProcessing;
  }

  @SkylarkCallable(
    name = "class_jar",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a jar File that is a result of annotation processing for this rule."
  )
  @Nullable
  public Artifact getGenClassJar() {
    return genClassJar;
  }

  @SkylarkCallable(
    name = "source_jar",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a source archive resulting from annotation processing of this rule."
  )
  @Nullable
  public Artifact getGenSourceJar() {
    return genSourceJar;
  }

  @SkylarkCallable(
    name = "transitive_class_jars",
    structField = true,
    doc =
        "Returns a transitive set of class file jars resulting from annotation "
            + "processing of this rule and its dependencies."
  )
  public NestedSet<Artifact> getTransitiveGenClassJars() {
    return transitiveGenClassJars;
  }

  @SkylarkCallable(
    name = "transitive_source_jars",
    structField = true,
    doc =
        "Returns a transitive set of source archives resulting from annotation processing "
            + "of this rule and its dependencies."
  )
  public NestedSet<Artifact> getTransitiveGenSourceJars() {
    return transitiveGenSourceJars;
  }

  @SkylarkCallable(
    name = "processor_classpath",
    structField = true,
    doc = "Returns a classpath of annotation processors applied to this rule."
  )
  public NestedSet<Artifact> getProcessorClasspath() {
    return processorClasspath;
  }

  @SkylarkCallable(
    name = "processor_classnames",
    structField = true,
    doc =
      "Returns class names of annotation processors applied to this rule."
  )
  public ImmutableList<String> getProcessorClassNames() {
    return processorClassNames;
  }
}
