// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/**
 * Interface for an info object containing information about jars that are a result of
 * annotation processing for a Java rule.
 */
@SkylarkModule(
    name = "java_annotation_processing",
    category = SkylarkModuleCategory.NONE,
    doc = "Information about jars that are a result of annotation processing for a Java rule."
)
public interface JavaAnnotationProcessingApi<FileTypeT extends FileApi> {

  @SkylarkCallable(
    name = "enabled",
    structField = true,
    doc = "Returns true if the Java rule uses annotation processing."
  )
  public boolean usesAnnotationProcessing();

  @SkylarkCallable(
    name = "class_jar",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a jar File that is a result of annotation processing for this rule."
  )
  @Nullable
  public FileTypeT getGenClassJar();

  @SkylarkCallable(
    name = "source_jar",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a source archive resulting from annotation processing of this rule."
  )
  @Nullable
  public FileTypeT getGenSourceJar();

  @SkylarkCallable(
    name = "transitive_class_jars",
    structField = true,
    doc =
        "Returns a transitive set of class file jars resulting from annotation "
            + "processing of this rule and its dependencies."
  )
  public NestedSet<FileTypeT> getTransitiveGenClassJars();

  @SkylarkCallable(
    name = "transitive_source_jars",
    structField = true,
    doc =
        "Returns a transitive set of source archives resulting from annotation processing "
            + "of this rule and its dependencies."
  )
  public NestedSet<FileTypeT> getTransitiveGenSourceJars();

  @SkylarkCallable(
    name = "processor_classpath",
    structField = true,
    doc = "Returns a classpath of annotation processors applied to this rule."
  )
  public NestedSet<FileTypeT> getProcessorClasspath();

  @SkylarkCallable(
    name = "processor_classnames",
    structField = true,
    doc =
      "Returns class names of annotation processors applied to this rule."
  )
  public ImmutableList<String> getProcessorClassNames();
}
