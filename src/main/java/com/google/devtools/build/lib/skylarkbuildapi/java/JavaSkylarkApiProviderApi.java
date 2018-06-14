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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * Provides access to information about Java rules. Every Java-related target provides
 * this struct, accessible as a java field on a Target.
 */
@SkylarkModule(
    name = "JavaSkylarkApiProvider",
    title = "java",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Provides access to information about Java rules. Every Java-related target provides "
            + "this struct, accessible as a <code>java</code> field on a "
            + "<a href=\"Target.html\">target</a>."
)
public interface JavaSkylarkApiProviderApi<FileT extends FileApi> {

  @SkylarkCallable(
    name = "source_jars",
    doc = "Returns the Jars containing Java source files for the target.",
    structField = true
  )
  public NestedSet<FileT> getSourceJars();

  @SkylarkCallable(
    name = "transitive_deps",
    doc = "Returns the transitive set of Jars required to build the target.",
    structField = true
  )
  public NestedSet<FileT> getTransitiveDeps();

  @SkylarkCallable(
    name = "transitive_runtime_deps",
    doc = "Returns the transitive set of Jars required on the target's runtime classpath.",
    structField = true
  )
  public NestedSet<FileT> getTransitiveRuntimeDeps();

  @SkylarkCallable(
    name = "transitive_source_jars",
    doc =
        "Returns the Jars containing Java source files for the target and all of its transitive "
            + "dependencies.",
    structField = true
  )
  public NestedSet<FileT> getTransitiveSourceJars();

  @SkylarkCallable(
    name = "outputs",
    doc = "Returns information about outputs of this Java target.",
    structField = true
  )
  public JavaRuleOutputJarsProviderApi<?> getOutputJars();

  @SkylarkCallable(
    name = "transitive_exports",
    structField = true,
    doc = "Returns transitive set of labels that are being exported from this rule."
  )
  public NestedSet<Label> getTransitiveExports();

  @SkylarkCallable(
    name = "annotation_processing",
    structField = true,
    allowReturnNones = true,
    doc = "Returns information about annotation processing for this Java target."
  )
  public JavaAnnotationProcessingApi<?> getGenJarsProvider();

  @SkylarkCallable(
    name = "compilation_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this Java target."
  )
  public JavaCompilationInfoProviderApi<?> getCompilationInfoProvider();
}
