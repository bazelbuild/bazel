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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.ProguardMappingProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that can provide a proguard obfuscation mapping to Android binaries or tests. */
@Immutable
public final class ProguardMappingProvider extends NativeInfo
    implements ProguardMappingProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final Artifact proguardMapping;

  public ProguardMappingProvider(Artifact proguardMapping) {
    super(PROVIDER);
    this.proguardMapping = proguardMapping;
  }

  @Override
  public Artifact getProguardMapping() {
    return proguardMapping;
  }

  /** The provider can construct the ProguardMappingProvider provider. */
  public static class Provider extends BuiltinProvider<ProguardMappingProvider>
      implements ProguardMappingProviderApi.Provider<Artifact> {

    private Provider() {
      super(NAME, ProguardMappingProvider.class);
    }

    @Override
    public ProguardMappingProvider createInfo(Artifact proguardMapping) throws EvalException {
      return new ProguardMappingProvider(proguardMapping);
    }
  }
}
