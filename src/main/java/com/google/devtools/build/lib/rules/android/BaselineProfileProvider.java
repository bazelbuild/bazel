// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.android.AndroidStarlarkData.fromNoneable;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.BaselineProfileProviderApi;
import net.starlark.java.eval.EvalException;

/** A target that can provide baseline profile files to Android binaries. */
@Immutable
public final class BaselineProfileProvider extends NativeInfo
    implements BaselineProfileProviderApi<Artifact> {

  public static final String PROVIDER_NAME = "BaselineProfileProvider";
  public static final Provider PROVIDER = new Provider();

  private final NestedSet<Artifact> transitiveBaselineProfiles;
  private final Artifact artProfileZip;

  public BaselineProfileProvider(
      NestedSet<Artifact> transitiveBaselineProfiles, Artifact artProfileZip) {
    this.transitiveBaselineProfiles = transitiveBaselineProfiles;
    this.artProfileZip = artProfileZip;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveBaselineProfilesForStarlark() {
    return Depset.of(Artifact.class, transitiveBaselineProfiles);
  }

  public NestedSet<Artifact> getTransitiveBaselineProfiles() {
    return transitiveBaselineProfiles;
  }

  public Artifact getArtProfileZip() {
    return artProfileZip;
  }

  public static BaselineProfileProvider merge(Iterable<BaselineProfileProvider> providers) {
    NestedSetBuilder<Artifact> files = NestedSetBuilder.stableOrder();
    for (BaselineProfileProvider wrapper : providers) {
      files.addTransitive(wrapper.getTransitiveBaselineProfiles());
    }
    return new BaselineProfileProvider(files.build(), null);
  }

  /** Provider class for {@link BaselineProfileProvider} objects. */
  public static class Provider extends BuiltinProvider<BaselineProfileProvider>
      implements BaselineProfileProviderApi.Provider<Artifact> {
    private Provider() {
      super(PROVIDER_NAME, BaselineProfileProvider.class);
    }

    public String getName() {
      return PROVIDER_NAME;
    }

    @Override
    public BaselineProfileProvider create(Depset files, Object artProfileZip) throws EvalException {
      return new BaselineProfileProvider(
          Depset.cast(files, Artifact.class, "files"), fromNoneable(artProfileZip, Artifact.class));
    }
  }
}
