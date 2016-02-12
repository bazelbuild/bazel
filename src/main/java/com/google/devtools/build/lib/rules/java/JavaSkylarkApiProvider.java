// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A class that exposes the Java providers to Skylark. It is intended to provide a
 * simple and stable interface for Skylark users.
 */
@SkylarkModule(
    name = "JavaSkylarkApiProvider", doc = "Provides access to information about Java rules")
public final class JavaSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "java";

  @SkylarkCallable(
      name = "source_jars",
      doc = "Returns the Jars containing Java source files for the target",
      structField = true)
  public NestedSet<Artifact> getSourceJars() {
    JavaSourceJarsProvider sourceJars = getInfo().getProvider(JavaSourceJarsProvider.class);
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJars.getSourceJars());
  }

  @SkylarkCallable(
      name = "transitive_deps",
      doc =  "Returns the transitive set of Jars required to build the target",
      structField = true)
  public NestedSet<Artifact> getTransitiveDeps() {
    JavaCompilationArgsProvider args = getInfo().getProvider(JavaCompilationArgsProvider.class);
    return args.getRecursiveJavaCompilationArgs().getCompileTimeJars();
  }

  @SkylarkCallable(
      name = "transitive_runtime_deps",
      doc =  "Returns the transitive set of Jars required on the target's runtime classpath",
      structField = true)
  public NestedSet<Artifact> getTransitiveRuntimeDeps() {
    JavaCompilationArgsProvider args = getInfo().getProvider(JavaCompilationArgsProvider.class);
    return args.getRecursiveJavaCompilationArgs().getRuntimeJars();
  }

  @SkylarkCallable(
      name = "transitive_source_jars",
      doc =
          "Returns the Jars containing Java source files for the target and all of its transitive "
              + "dependencies",
      structField = true)
  public NestedSet<Artifact> getTransitiveSourceJars() {
    JavaSourceJarsProvider sourceJars = getInfo().getProvider(JavaSourceJarsProvider.class);
    return sourceJars.getTransitiveSourceJars();
  }

  @SkylarkCallable(
    name = "outputs",
    doc = "Returns information about outputs of this Java target",
    structField = true
  )
  public JavaRuleOutputJarsProvider getOutputJars() {
    return getInfo().getProvider(JavaRuleOutputJarsProvider.class);
  }

  @SkylarkCallable(
    name = "annotation_processing",
    structField = true,
    allowReturnNones = true,
    doc = "Returns information about annotation processing for this Java target"
  )
  public JavaGenJarsProvider getGenJarsProvider() {
    return getInfo().getProvider(JavaGenJarsProvider.class);
  }

  @SkylarkCallable(
    name = "compilation_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this Java target"
  )
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return getInfo().getProvider(JavaCompilationInfoProvider.class);
  }

}
