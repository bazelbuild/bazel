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
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidPreDexJarProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A provider of the final Jar to be dexed for targets that build APKs. */
@Immutable
public final class AndroidPreDexJarProvider extends NativeInfo
    implements AndroidPreDexJarProviderApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final Artifact preDexJar;

  /** Returns the jar to be dexed. */
  @Override
  public Artifact getPreDexJar() {
    return preDexJar;
  }

  public AndroidPreDexJarProvider(Artifact preDexJar) {
    super(PROVIDER);
    this.preDexJar = preDexJar;
  }

  /** Provider class for {@link AndroidPreDexJarProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidPreDexJarProvider>
      implements AndroidPreDexJarProviderApi.Provider<Artifact> {
    private Provider() {
      super(NAME, AndroidPreDexJarProvider.class);
    }

    @Override
    public AndroidPreDexJarProviderApi<Artifact> createInfo(Artifact preDexJar)
        throws EvalException {
      return new AndroidPreDexJarProvider(preDexJar);
    }
  }
}
