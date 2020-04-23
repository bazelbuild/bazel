// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidNativeLibsInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Provider of transitively available ZIPs of native libs that should be directly copied into the
 * APK.
 */
@Immutable
public final class AndroidNativeLibsInfo extends NativeInfo
    implements AndroidNativeLibsInfoApi<Artifact> {

  private static final String SKYLARK_NAME = "AndroidNativeLibsInfo";

  public static final AndroidNativeLibsInfoProvider PROVIDER =
      new AndroidNativeLibsInfoProvider();

  private final NestedSet<Artifact> nativeLibs;

  public AndroidNativeLibsInfo(NestedSet<Artifact> nativeLibs) {
    super(PROVIDER);
    this.nativeLibs = nativeLibs;
  }

  @Override
  public Depset /*<Artifact>*/ getNativeLibsForStarlark() {
    return Depset.of(Artifact.TYPE, nativeLibs);
  }

  NestedSet<Artifact> getNativeLibs() {
    return nativeLibs;
  }

  /** Provider for {@link AndroidNativeLibsInfo}. */
  public static class AndroidNativeLibsInfoProvider extends BuiltinProvider<AndroidNativeLibsInfo>
      implements AndroidNativeLibsInfoApiProvider {

    private AndroidNativeLibsInfoProvider() {
      super(SKYLARK_NAME, AndroidNativeLibsInfo.class);
    }

    @Override
    public AndroidNativeLibsInfo createInfo(Depset nativeLibs) throws EvalException {
      return new AndroidNativeLibsInfo(Depset.cast(nativeLibs, Artifact.class, "native_libs"));
    }
  }
}
