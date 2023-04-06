// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidOptimizedJarInfoApi;
import net.starlark.java.eval.EvalException;

/** A provider for Android Optimized artifacts */
@Immutable
public class AndroidOptimizedJarInfo extends NativeInfo
    implements AndroidOptimizedJarInfoApi<Artifact> {

  public static final String PROVIDER_NAME = "AndroidOptimizedJarInfo";
  public static final Provider PROVIDER = new Provider();

  private final Artifact optimizedJar;

  public AndroidOptimizedJarInfo(Artifact optimizedJar) {
    this.optimizedJar = optimizedJar;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public Artifact getOptimizedJar() {
    return optimizedJar;
  }

  /** Provider for {@link AndroidOptimizedJarInfo}. */
  public static class Provider extends BuiltinProvider<AndroidOptimizedJarInfo>
      implements AndroidOptimizedJarInfoApi.Provider<Artifact> {
    private Provider() {
      super(PROVIDER_NAME, AndroidOptimizedJarInfo.class);
    }

    public String getName() {
      return PROVIDER_NAME;
    }

    @Override
    public AndroidOptimizedJarInfo createInfo(Artifact optimizedJar) throws EvalException {
      return new AndroidOptimizedJarInfo(optimizedJar);
    }
  }
}
