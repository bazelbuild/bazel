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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidApplicationResourceInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidBinaryNativeLibsInfoApi;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;

/** A provider for native libs for android_binary. */
@Immutable
public class AndroidBinaryNativeLibsInfo extends NativeInfo
    implements AndroidBinaryNativeLibsInfoApi<Artifact> {

  /** Singleton instance of the provider type for {@link AndroidBinaryNativeLibsInfo}. */
  public static final AndroidBinaryNativeLibsInfoProvider PROVIDER =
      new AndroidBinaryNativeLibsInfoProvider();

  private final NativeLibs nativeLibs;
  private final NestedSet<Artifact> transitiveNativeLibs;

  AndroidBinaryNativeLibsInfo(NativeLibs nativeLibs, NestedSet<Artifact> transitiveNativeLibs) {
    this.nativeLibs = nativeLibs;
    this.transitiveNativeLibs = transitiveNativeLibs;
  }

  @Override
  public AndroidBinaryNativeLibsInfoProvider getProvider() {
    return PROVIDER;
  }

  @Nullable
  @Override
  public Dict<String, Depset> getNativeLibsStarlark() {
    if (nativeLibs == null) {
      return null;
    }
    return Dict.immutableCopyOf(
        Maps.transformValues(nativeLibs.getMap(), set -> Depset.of(Artifact.class, set)));
  }

  @Nullable
  @Override
  public Artifact getNativeLibsNameStarlark() {
    if (nativeLibs == null) {
      return null;
    }
    return nativeLibs.getName();
  }

  @Nullable
  @Override
  public Depset getTransitiveNativeLibsStarlark() {
    if (transitiveNativeLibs == null) {
      return null;
    }
    return Depset.of(Artifact.class, transitiveNativeLibs);
  }

  @Nullable
  public NativeLibs getNativeLibs() {
    return nativeLibs;
  }

  @Nullable
  public NestedSet<Artifact> getTransitiveNativeLibs() {
    return transitiveNativeLibs;
  }

  /** Provider for {@link AndroidBinaryNativeLibsInfo}. */
  public static class AndroidBinaryNativeLibsInfoProvider
      extends BuiltinProvider<AndroidBinaryNativeLibsInfo>
      implements AndroidBinaryNativeLibsInfoApi.Provider<Artifact> {

    private AndroidBinaryNativeLibsInfoProvider() {
      super(AndroidApplicationResourceInfoApi.NAME, AndroidBinaryNativeLibsInfo.class);
    }

    @Override
    public AndroidBinaryNativeLibsInfoApi<Artifact> createInfo(
        Dict<?, ?> nativeLibs, Object nativeLibsName, Object transitiveNativeLibs)
        throws EvalException {
      Dict<String, Depset> nativeLibsDict =
          Dict.cast(nativeLibs, String.class, Depset.class, "native_libs");
      ImmutableMap.Builder<String, NestedSet<Artifact>> nativeLibsMapBuilder =
          ImmutableMap.builder();
      for (Entry<String, Depset> entry : nativeLibsDict.entrySet()) {
        nativeLibsMapBuilder.put(
            entry.getKey(), Depset.cast(entry.getValue(), Artifact.class, "native_libs"));
      }
      return new AndroidBinaryNativeLibsInfo(
          NativeLibs.of(
              nativeLibsMapBuilder.buildOrThrow(),
              AndroidStarlarkData.fromNoneable(nativeLibsName, Artifact.class)),
          AndroidStarlarkData.fromNoneableDepset(transitiveNativeLibs, "transitive_native_libs"));
    }
  }
}
