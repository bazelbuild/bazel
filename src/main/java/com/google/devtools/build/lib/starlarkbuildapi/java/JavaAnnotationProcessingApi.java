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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/**
 * Interface for an info object containing information about jars that are a result of annotation
 * processing for a Java rule.
 */
@StarlarkBuiltin(
    name = "java_annotation_processing",
    category = DocCategory.BUILTIN,
    doc = "Information about jars that are a result of annotation processing for a Java rule.")
public interface JavaAnnotationProcessingApi<FileTypeT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(
      name = "enabled",
      structField = true,
      doc = "Deprecated. Returns true if annotation processing was applied on this target.")
  boolean usesAnnotationProcessing() throws EvalException;

  @StarlarkMethod(
      name = "class_jar",
      structField = true,
      allowReturnNones = true,
      doc =
          "Deprecated: Please use <code>JavaInfo.java_outputs.generated_class_jar</code> instead.")
  @Nullable
  FileTypeT getGenClassJar() throws EvalException;

  @StarlarkMethod(
      name = "source_jar",
      structField = true,
      allowReturnNones = true,
      doc =
          "Deprecated: Please use <code>JavaInfo.java_outputs.generated_source_jar</code> instead.")
  @Nullable
  FileTypeT getGenSourceJar() throws EvalException;

  @StarlarkMethod(
      name = "transitive_class_jars",
      structField = true,
      doc =
          "Deprecated. Returns a transitive set of class file jars resulting from annotation "
              + "processing of this rule and its dependencies.")
  Depset /*<FileTypeT>*/ getTransitiveGenClassJarsForStarlark() throws EvalException;

  @StarlarkMethod(
      name = "transitive_source_jars",
      structField = true,
      doc =
          "Deprecated. Returns a transitive set of source archives resulting from annotation "
              + "processing of this rule and its dependencies.")
  Depset /*<FileTypeT>*/ getTransitiveGenSourceJarsForStarlark() throws EvalException;

  @StarlarkMethod(
      name = "processor_classpath",
      structField = true,
      doc =
          "Deprecated: Please use <code>JavaInfo.plugins</code> instead. Returns a classpath of"
              + " annotation processors applied to this rule.")
  Depset /*<FileTypeT>*/ getProcessorClasspathForStarlark() throws EvalException;

  @StarlarkMethod(
      name = "processor_classnames",
      structField = true,
      doc =
          "Deprecated: Please use <code>JavaInfo.plugins</code> instead. Returns class names of"
              + " annotation processors applied to this rule.")
  ImmutableList<String> getProcessorClassNamesList() throws EvalException;
}
