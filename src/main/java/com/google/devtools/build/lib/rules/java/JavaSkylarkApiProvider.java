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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaSkylarkApiProviderApi;
import javax.annotation.Nullable;

/**
 * A class that exposes the Java providers to Skylark. It is intended to provide a simple and stable
 * interface for Skylark users.
 */
@AutoCodec
public final class JavaSkylarkApiProvider extends SkylarkApiProvider
    implements JavaSkylarkApiProviderApi<Artifact> {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "java";
  /** The name of the field in Skylark proto aspects used to access this class. */
  public static final SkylarkProviderIdentifier SKYLARK_NAME =
      SkylarkProviderIdentifier.forLegacy(NAME);

  @Nullable private final TransitiveInfoProviderMap transitiveInfoProviderMap;

  /**
   * Creates a Skylark API provider that reads information from its associated target's providers.
   */
  public static JavaSkylarkApiProvider fromRuleContext() {
    return new JavaSkylarkApiProvider(null);
  }

  /** Creates a Skylark API provider that reads information from an explicit provider map. */
  @AutoCodec.Instantiator
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

  @Override
  public NestedSet<Artifact> getSourceJars() {
    JavaSourceJarsProvider sourceJarsProvider = getProvider(JavaSourceJarsProvider.class);
    if (sourceJarsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJarsProvider.getSourceJars());
  }

  @Override
  public NestedSet<Artifact> getTransitiveDeps() {
    JavaCompilationArgsProvider compilationArgsProvider =
        getProvider(JavaCompilationArgsProvider.class);
    if (compilationArgsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return compilationArgsProvider.getTransitiveCompileTimeJars();
  }

  @Override
  public NestedSet<Artifact> getTransitiveRuntimeDeps() {
    JavaCompilationArgsProvider compilationArgsProvider =
        getProvider(JavaCompilationArgsProvider.class);
    if (compilationArgsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return compilationArgsProvider.getRuntimeJars();
  }

  @Override
  public NestedSet<Artifact> getTransitiveSourceJars() {
    JavaSourceJarsProvider sourceJarsProvider = getProvider(JavaSourceJarsProvider.class);
    if (sourceJarsProvider == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return sourceJarsProvider.getTransitiveSourceJars();
  }

  @Override
  public JavaRuleOutputJarsProvider getOutputJars() {
    return getProvider(JavaRuleOutputJarsProvider.class);
  }

  @Override
  public NestedSet<Label> getTransitiveExports() {
    JavaExportsProvider exportsProvider = getProvider(JavaExportsProvider.class);
    if (exportsProvider != null) {
      return exportsProvider.getTransitiveExports();
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  @Override
  public JavaGenJarsProvider getGenJarsProvider() {
    return getProvider(JavaGenJarsProvider.class);
  }

  @Override
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return getProvider(JavaCompilationInfoProvider.class);
  }
}
