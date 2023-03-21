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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidNeverLinkLibrariesProviderApi;
import net.starlark.java.eval.EvalException;

/**
 * A target that can provide neverlink libraries for Android targets.
 *
 * <p>All targets implementing this interface must also implement {@link
 * com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider}.
 */
@Immutable
public final class AndroidNeverLinkLibrariesProvider extends NativeInfo
    implements AndroidNeverLinkLibrariesProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final NestedSet<Artifact> transitiveNeverLinkLibraries;

  public static AndroidNeverLinkLibrariesProvider create(
      NestedSet<Artifact> transitiveNeverLinkLibraries) {
    return new AndroidNeverLinkLibrariesProvider(transitiveNeverLinkLibraries);
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  AndroidNeverLinkLibrariesProvider(NestedSet<Artifact> transitiveNeverLinkLibraries) {
    if (transitiveNeverLinkLibraries == null) {
      throw new NullPointerException("Null transitiveNeverLinkLibraries");
    }
    this.transitiveNeverLinkLibraries = transitiveNeverLinkLibraries;
  }

  /** Returns the set of neverlink libraries in the transitive closure. */
  public NestedSet<Artifact> getTransitiveNeverLinkLibraries() {
    return transitiveNeverLinkLibraries;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveNeverLinkLibrariesForStarlark() {
    return Depset.of(Artifact.class, transitiveNeverLinkLibraries);
  }

  /** Provider class for {@link AndroidNeverLinkLibrariesProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidNeverLinkLibrariesProvider>
      implements AndroidNeverLinkLibrariesProviderApi.Provider<Artifact> {
    private Provider() {
      super(NAME, AndroidNeverLinkLibrariesProvider.class);
    }

    public String getName() {
      return NAME;
    }

    @Override
    public AndroidNeverLinkLibrariesProvider create(Depset transitiveNeverLinkLibraries)
        throws EvalException {
      return new AndroidNeverLinkLibrariesProvider(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(
                  Depset.cast(
                      transitiveNeverLinkLibraries,
                      Artifact.class,
                      "transitive_neverlink_libraries"))
              .build());
    }
  }
}
