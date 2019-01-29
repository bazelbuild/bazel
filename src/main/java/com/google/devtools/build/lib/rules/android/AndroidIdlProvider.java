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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidIdlProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Configured targets implementing this provider can contribute Android IDL information to the
 * compilation.
 */
@Immutable
public final class AndroidIdlProvider extends NativeInfo
    implements AndroidIdlProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final NestedSet<String> transitiveIdlImportRoots;
  private final NestedSet<Artifact> transitiveIdlImports;
  private final NestedSet<Artifact> transitiveIdlJars;
  private final NestedSet<Artifact> transitiveIdlPreprocessed;

  public AndroidIdlProvider(
      NestedSet<String> transitiveIdlImportRoots,
      NestedSet<Artifact> transitiveIdlImports,
      NestedSet<Artifact> transitiveIdlJars,
      NestedSet<Artifact> transitiveIdlPreprocessed) {
    super(PROVIDER);
    this.transitiveIdlImportRoots = transitiveIdlImportRoots;
    this.transitiveIdlImports = transitiveIdlImports;
    this.transitiveIdlJars = transitiveIdlJars;
    this.transitiveIdlPreprocessed = transitiveIdlPreprocessed;
  }

  @Override
  public NestedSet<String> getTransitiveIdlImportRoots() {
    return transitiveIdlImportRoots;
  }

  @Override
  public NestedSet<Artifact> getTransitiveIdlImports() {
    return transitiveIdlImports;
  }

  @Override
  public NestedSet<Artifact> getTransitiveIdlJars() {
    return transitiveIdlJars;
  }

  @Override
  public NestedSet<Artifact> getTransitiveIdlPreprocessed() {
    return transitiveIdlPreprocessed;
  }

  /** The provider can construct the Android IDL provider. */
  public static class Provider extends BuiltinProvider<AndroidIdlProvider>
      implements AndroidIdlProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, AndroidIdlProvider.class);
    }

    @Override
    public AndroidIdlProvider createInfo(
        SkylarkNestedSet transitiveIdlImportRoots,
        SkylarkNestedSet transitiveIdlImports,
        SkylarkNestedSet transitiveIdlJars,
        SkylarkNestedSet transitiveIdlPreprocessed)
        throws EvalException {
      return new AndroidIdlProvider(
          NestedSetBuilder.<String>stableOrder()
              .addTransitive(transitiveIdlImportRoots.getSet(String.class))
              .build(),
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(transitiveIdlImports.getSet(Artifact.class))
              .build(),
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(transitiveIdlJars.getSet(Artifact.class))
              .build(),
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(transitiveIdlPreprocessed.getSet(Artifact.class))
              .build());
    }
  }
}
