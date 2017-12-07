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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.skylark.SkylarkApiProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
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
          + "this struct, accessible as a <code>java</code> field on a "
          + "<a href=\"Target.html\">target</a>."
)
public final class JavaSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "java";
  /** The name of the field in Skylark proto aspects used to access this class. */
  public static final SkylarkProviderIdentifier PROTO_NAME =
      SkylarkProviderIdentifier.forLegacy("proto_java");

  @Nullable private final TransitiveInfoProviderMap transitiveInfoProviderMap;

  /**
   * Creates a Skylark API provider that reads information from its associated target's providers.
   */
  public static JavaSkylarkApiProvider fromRuleContext() {
    return new JavaSkylarkApiProvider(null);
  }

  /**
   * Creates a Skylark API provider that reads information from an explicit provider map.
   */
  public static JavaSkylarkApiProvider fromProviderMap(
      TransitiveInfoProviderMap transitiveInfoProviderMap) {
    return new JavaSkylarkApiProvider(transitiveInfoProviderMap);
  }

  private JavaSkylarkApiProvider(TransitiveInfoProviderMap transitiveInfoProviderMap) {
    this.transitiveInfoProviderMap = transitiveInfoProviderMap;
  }

  @Nullable
  private <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    if (transitiveInfoProviderMap != null) {
      return JavaInfo.getProvider(provider, transitiveInfoProviderMap);
    }
    return JavaInfo.getProvider(provider, getInfo());
  }

  @SkylarkCallable(
    name = "source_jars",
    doc = "Returns the Jars containing Java source files for the target.",
    structField = true
  )
  public NestedSet<Artifact> getSourceJars() {
    JavaSourceJarsProvider sourceJarsProvider = getProvider(JavaSourceJarsProvider.class);
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
    JavaCompilationArgsProvider compilationArgsProvider =
        getProvider(JavaCompilationArgsProvider.class);
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
    JavaCompilationArgsProvider compilationArgsProvider =
        getProvider(JavaCompilationArgsProvider.class);
    if (compilationArgsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return compilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars();
  }

  @SkylarkCallable(
    name = "transitive_source_jars",
    doc =
        "Returns the Jars containing Java source files for the target and all of its transitive "
            + "dependencies.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveSourceJars() {
    JavaSourceJarsProvider sourceJarsProvider = getProvider(JavaSourceJarsProvider.class);
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
    return getProvider(JavaRuleOutputJarsProvider.class);
  }

  @SkylarkCallable(
    name = "transitive_exports",
    structField = true,
    doc = "Returns transitive set of labels that are being exported from this rule."
  )
  public NestedSet<Label> getTransitiveExports() {
    JavaExportsProvider exportsProvider = getProvider(JavaExportsProvider.class);
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
    return getProvider(JavaGenJarsProvider.class);
  }

  @SkylarkCallable(
    name = "compilation_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this Java target."
  )
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return getProvider(JavaCompilationInfoProvider.class);
  }
}
