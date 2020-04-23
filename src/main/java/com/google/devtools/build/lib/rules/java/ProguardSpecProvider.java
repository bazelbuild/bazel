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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.java.ProguardSpecProviderApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that can provide proguard specifications to Android binaries. */
@Immutable
public final class ProguardSpecProvider extends NativeInfo
    implements ProguardSpecProviderApi<Artifact> {

  public static final String PROVIDER_NAME = "ProguardSpecProvider";
  public static final Provider PROVIDER = new Provider();

  private final NestedSet<Artifact> transitiveProguardSpecs;

  public ProguardSpecProvider(NestedSet<Artifact> transitiveProguardSpecs) {
    super(PROVIDER);
    this.transitiveProguardSpecs = transitiveProguardSpecs;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveProguardSpecsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveProguardSpecs);
  }

  public NestedSet<Artifact> getTransitiveProguardSpecs() {
    return transitiveProguardSpecs;
  }

  public static ProguardSpecProvider merge(Iterable<ProguardSpecProvider> providers) {
    NestedSetBuilder<Artifact> specs = NestedSetBuilder.stableOrder();
    for (ProguardSpecProvider wrapper : providers) {
      specs.addTransitive(wrapper.getTransitiveProguardSpecs());
    }
    return new ProguardSpecProvider(specs.build());
  }

  /** Provider class for {@link ProguardSpecProvider} objects. */
  public static class Provider extends BuiltinProvider<ProguardSpecProvider>
      implements ProguardSpecProviderApi.Provider<Artifact> {
    private Provider() {
      super(PROVIDER_NAME, ProguardSpecProvider.class);
    }

    public String getName() {
      return PROVIDER_NAME;
    }

    @Override
    public ProguardSpecProvider create(Depset specs) throws EvalException {
      return new ProguardSpecProvider(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(Depset.cast(specs, Artifact.class, "specs"))
              .build());
    }
  }
}
