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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/**
 * A class that exposes the Java providers to Skylark. It is intended to provide a simple and stable
 * interface for Skylark users.
 */
@SkylarkModule(
  name = "JavaSkylarkApiProvider",
  title = "java",
  category = SkylarkModuleCategory.PROVIDER,
  doc =
      "Provides access to information about Java rules. Every Java-related target provides "
          + "this struct, accessible as a 'java' field on a Target struct."
)
public final class JavaSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "java";
  /** The name of the field in Skylark proto aspects used to access this class. */
  public static final String PROTO_NAME = "proto_java";

  private final JavaRuleOutputJarsProvider ruleOutputJarsProvider;
  @Nullable private final JavaSourceJarsProvider sourceJarsProvider;
  @Nullable private final JavaGenJarsProvider genJarsProvider;
  @Nullable private final JavaCompilationInfoProvider compilationInfoProvider;
  @Nullable private final JavaCompilationArgsProvider compilationArgsProvider;
  @Nullable private final JavaExportsProvider exportsProvider;

  private JavaSkylarkApiProvider(
      JavaRuleOutputJarsProvider ruleOutputJarsProvider,
      @Nullable JavaSourceJarsProvider sourceJarsProvider,
      @Nullable JavaGenJarsProvider genJarsProvider,
      @Nullable JavaCompilationInfoProvider compilationInfoProvider,
      @Nullable JavaCompilationArgsProvider compilationArgsProvider,
      @Nullable JavaExportsProvider exportsProvider) {
    this.compilationInfoProvider = compilationInfoProvider;
    this.ruleOutputJarsProvider = ruleOutputJarsProvider;
    this.sourceJarsProvider = sourceJarsProvider;
    this.genJarsProvider = genJarsProvider;
    this.compilationArgsProvider = compilationArgsProvider;
    this.exportsProvider = exportsProvider;
  }

  @SkylarkCallable(
    name = "source_jars",
    doc = "Returns the Jars containing Java source files for the target.",
    structField = true
  )
  public NestedSet<Artifact> getSourceJars() {
    if (sourceJarsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJarsProvider.getSourceJars());
  }

  @SkylarkCallable(
    name = "transitive_deps",
    doc = "Returns the transitive set of Jars required to build the target.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveDeps() {
    if (compilationArgsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return compilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars();
  }

  @SkylarkCallable(
    name = "transitive_runtime_deps",
    doc = "Returns the transitive set of Jars required on the target's runtime classpath.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveRuntimeDeps() {
    if (compilationArgsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return compilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars();
  }

  @SkylarkCallable(
    name = "transitive_source_jars",
    doc =
        "Returns the Jars containing Java source files for the target and all of its transitive "
            + "dependencies",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveSourceJars() {
    if (sourceJarsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return sourceJarsProvider.getTransitiveSourceJars();
  }

  @SkylarkCallable(
    name = "outputs",
    doc = "Returns information about outputs of this Java target.",
    structField = true
  )
  public JavaRuleOutputJarsProvider getOutputJars() {
    return ruleOutputJarsProvider;
  }

  @SkylarkCallable(
    name = "transitive_exports",
    structField = true,
    doc = "Returns transitive set of labels that are being exported from this rule."
  )
  public NestedSet<Label> getTransitiveExports() {
    if (exportsProvider != null) {
      return exportsProvider.getTransitiveExports();
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  @SkylarkCallable(
    name = "annotation_processing",
    structField = true,
    allowReturnNones = true,
    doc = "Returns information about annotation processing for this Java target."
  )
  public JavaGenJarsProvider getGenJarsProvider() {
    return genJarsProvider;
  }

  @SkylarkCallable(
    name = "compilation_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this Java target."
  )
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return compilationInfoProvider;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link JavaSkylarkApiProvider} */
  public static class Builder {
    private JavaRuleOutputJarsProvider ruleOutputJarsProvider;
    private JavaSourceJarsProvider sourceJarsProvider;
    private JavaGenJarsProvider genJarsProvider;
    private JavaCompilationInfoProvider compilationInfoProvider;
    private JavaCompilationArgsProvider compilationArgsProvider;
    private JavaExportsProvider exportsProvider;

    public Builder setRuleOutputJarsProvider(JavaRuleOutputJarsProvider ruleOutputJarsProvider) {
      this.ruleOutputJarsProvider = ruleOutputJarsProvider;
      return this;
    }

    public Builder setSourceJarsProvider(JavaSourceJarsProvider sourceJarsProvider) {
      this.sourceJarsProvider = sourceJarsProvider;
      return this;
    }

    public Builder setGenJarsProvider1(JavaGenJarsProvider genJarsProvider) {
      this.genJarsProvider = genJarsProvider;
      return this;
    }

    public Builder setCompilationInfoProvider(JavaCompilationInfoProvider compilationInfoProvider) {
      this.compilationInfoProvider = compilationInfoProvider;
      return this;
    }

    public Builder setCompilationArgsProvider(JavaCompilationArgsProvider compilationArgsProvider) {
      this.compilationArgsProvider = compilationArgsProvider;
      return this;
    }

    public Builder setExportsProvider(JavaExportsProvider exportsProvider) {
      this.exportsProvider = exportsProvider;
      return this;
    }

    public JavaSkylarkApiProvider build() {
      checkNotNull(ruleOutputJarsProvider, "Must provide JavaRuleOutputJarsProvider");
      return new JavaSkylarkApiProvider(
          ruleOutputJarsProvider,
          sourceJarsProvider,
          genJarsProvider,
          compilationInfoProvider,
          compilationArgsProvider,
          exportsProvider);
    }
  }
}
