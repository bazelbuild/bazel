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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LtoBackendArtifacts;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaNativeLibraryInfoApi;
import net.starlark.java.eval.EvalException;

/**
 * A target that provides native libraries in the transitive closure of its deps that are needed for
 * executing Java code.
 */
@Immutable
@AutoCodec
public final class JavaNativeLibraryInfo extends NativeInfo
    implements JavaNativeLibraryInfoApi<Artifact, LtoBackendArtifacts, LibraryToLink> {

  public static final Provider PROVIDER = new Provider();

  private final NestedSet<LibraryToLink> transitiveJavaNativeLibraries;

  public JavaNativeLibraryInfo(NestedSet<LibraryToLink> transitiveJavaNativeLibraries) {
    this.transitiveJavaNativeLibraries = transitiveJavaNativeLibraries;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /**
   * Collects native libraries in the transitive closure of its deps that are needed for executing
   * Java code.
   */
  NestedSet<LibraryToLink> getTransitiveJavaNativeLibraries() {
    return transitiveJavaNativeLibraries;
  }

  @Override
  public Depset /*<LibraryToLink>*/ getTransitiveJavaNativeLibrariesForStarlark() {
    return Depset.of(LibraryToLink.TYPE, transitiveJavaNativeLibraries);
  }

  public static JavaNativeLibraryInfo merge(Iterable<JavaNativeLibraryInfo> deps) {
    NestedSetBuilder<LibraryToLink> transitiveSourceJars = NestedSetBuilder.stableOrder();

    for (JavaNativeLibraryInfo wrapper : deps) {
      transitiveSourceJars.addTransitive(wrapper.getTransitiveJavaNativeLibraries());
    }
    return new JavaNativeLibraryInfo(transitiveSourceJars.build());
  }

  /** Provider class for {@link JavaNativeLibraryInfo} objects. */
  public static class Provider extends BuiltinProvider<JavaNativeLibraryInfo>
      implements JavaNativeLibraryInfoApi.Provider<Artifact, LtoBackendArtifacts, LibraryToLink> {

    private Provider() {
      super(NAME, JavaNativeLibraryInfo.class);
    }

    public String getName() {
      return NAME;
    }

    @Override
    public JavaNativeLibraryInfo create(Depset transitiveLibraries)
        throws EvalException {
      return new JavaNativeLibraryInfo(
          NestedSetBuilder.<LibraryToLink>stableOrder()
              .addTransitive(
                  Depset.cast(transitiveLibraries, LibraryToLink.class, "transitive_libraries"))
              .build());
    }
  }
}
