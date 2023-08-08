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

package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidApplicationResourceInfoApi.AndroidApplicationResourceInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDeviceBrokerInfoApi.AndroidDeviceBrokerInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidInstrumentationInfoApi.AndroidInstrumentationInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidNativeLibsInfoApi.AndroidNativeLibsInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidResourcesInfoApi.AndroidResourcesInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.ApkInfoApi.ApkInfoApiProvider;
import com.google.devtools.build.lib.starlarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.core.ContextAndFlagGuardedValue;
import java.util.Map;
import net.starlark.java.eval.FlagGuardedValue;

/** {@link Bootstrap} for Starlark objects related to Android rules. */
public class AndroidBootstrap implements Bootstrap {
  private static final ImmutableSet<PackageIdentifier> allowedRepositories =
      ImmutableSet.of(
          PackageIdentifier.createUnchecked("_builtins", ""),
          PackageIdentifier.createUnchecked("rules_android", ""),
          PackageIdentifier.createUnchecked("", "tools/build_defs/android"));

  private final AndroidStarlarkCommonApi<?, ?, ?, ?, ?> androidCommon;
  private final ImmutableMap<String, Object> providers;

  public AndroidBootstrap(
      AndroidStarlarkCommonApi<?, ?, ?, ?, ?> androidCommon,
      ApkInfoApiProvider apkInfoProvider,
      AndroidInstrumentationInfoApiProvider<?> androidInstrumentationInfoProvider,
      AndroidDeviceBrokerInfoApiProvider androidDeviceBrokerInfoProvider,
      AndroidResourcesInfoApiProvider<?, ?, ?> androidResourcesInfoProvider,
      AndroidNativeLibsInfoApiProvider androidNativeLibsInfoProvider,
      AndroidApplicationResourceInfoApiProvider<?> androidApplicationResourceInfoApiProvider,
      AndroidSdkProviderApi.Provider<?, ?, ?> androidSdkProviderApi,
      AndroidManifestInfoApi.Provider<?> androidManifestInfo,
      AndroidAssetsInfoApi.Provider<?, ?> androidAssetsInfoProvider,
      AndroidLibraryAarInfoApi.Provider<?> androidLibraryAarInfoProvider,
      AndroidProguardInfoApi.Provider<?> androidProguardInfoProvider,
      AndroidIdlProviderApi.Provider<?> androidIdlProvider,
      AndroidIdeInfoProviderApi.Provider<?, ?> androidIdeInfoProvider,
      AndroidPreDexJarProviderApi.Provider<?> androidPreDexJarProviderApiProvider,
      AndroidCcLinkParamsProviderApi.Provider<?, ?> androidCcLinkParamsProviderApiProvider,
      DataBindingV2ProviderApi.Provider<?> dataBindingV2ProviderApiProvider,
      AndroidLibraryResourceClassJarProviderApi.Provider<?>
          androidLibraryResourceClassJarProviderApiProvider,
      AndroidFeatureFlagSetProviderApi.Provider androidFeatureFlagSetProviderApiProvider,
      ProguardMappingProviderApi.Provider<?> proguardMappingProviderApiProvider,
      AndroidBinaryDataInfoApi.Provider<?, ?, ?, ?> androidBinaryDataInfoProvider,
      AndroidBinaryNativeLibsInfoApi.Provider<?> androidBinaryInternalNativeLibsInfoApiProvider,
      BaselineProfileProviderApi.Provider<?> baselineProfileProvider,
      AndroidNeverLinkLibrariesProviderApi.Provider<?> androidNeverLinkLibrariesProvider,
      AndroidOptimizedJarInfoApi.Provider<?> androidOptimizedJarInfo,
      AndroidDexInfoApi.Provider<?> androidDexInfoApiProvider) {

    this.androidCommon = androidCommon;
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    builder.put(ApkInfoApi.NAME, apkInfoProvider);
    builder.put(AndroidInstrumentationInfoApi.NAME, androidInstrumentationInfoProvider);
    builder.put(AndroidDeviceBrokerInfoApi.NAME, androidDeviceBrokerInfoProvider);
    builder.put(AndroidResourcesInfoApi.NAME, androidResourcesInfoProvider);
    builder.put(AndroidNativeLibsInfoApi.NAME, androidNativeLibsInfoProvider);
    builder.put(AndroidApplicationResourceInfoApi.NAME, androidApplicationResourceInfoApiProvider);
    builder.put(
        AndroidBinaryNativeLibsInfoApi.NAME, androidBinaryInternalNativeLibsInfoApiProvider);
    builder.put(AndroidSdkProviderApi.NAME, androidSdkProviderApi);
    builder.put(AndroidManifestInfoApi.NAME, androidManifestInfo);
    builder.put(AndroidAssetsInfoApi.NAME, androidAssetsInfoProvider);
    builder.put(AndroidLibraryAarInfoApi.NAME, androidLibraryAarInfoProvider);
    builder.put(AndroidProguardInfoApi.NAME, androidProguardInfoProvider);
    builder.put(AndroidIdlProviderApi.NAME, androidIdlProvider);
    builder.put(AndroidIdeInfoProviderApi.NAME, androidIdeInfoProvider);
    builder.put(AndroidPreDexJarProviderApi.NAME, androidPreDexJarProviderApiProvider);
    builder.put(AndroidCcLinkParamsProviderApi.NAME, androidCcLinkParamsProviderApiProvider);
    builder.put(DataBindingV2ProviderApi.NAME, dataBindingV2ProviderApiProvider);
    builder.put(
        AndroidLibraryResourceClassJarProviderApi.NAME,
        androidLibraryResourceClassJarProviderApiProvider);
    builder.put(AndroidFeatureFlagSetProviderApi.NAME, androidFeatureFlagSetProviderApiProvider);
    builder.put(ProguardMappingProviderApi.NAME, proguardMappingProviderApiProvider);
    builder.put(AndroidBinaryDataInfoApi.NAME, androidBinaryDataInfoProvider);
    builder.put(BaselineProfileProviderApi.NAME, baselineProfileProvider);
    builder.put(AndroidNeverLinkLibrariesProviderApi.NAME, androidNeverLinkLibrariesProvider);
    builder.put(AndroidOptimizedJarInfoApi.NAME, androidOptimizedJarInfo);
    builder.put(AndroidDexInfoApi.NAME, androidDexInfoApiProvider);
    providers = builder.build();
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    // TODO: Make an incompatible change flag to hide android_common behind
    // --experimental_google_legacy_api.
    // Rationale: android_common module contains commonly used functions used outside of
    // the Android Starlark migration. Let's not break them without an incompatible
    // change process.
    builder.put(
        "android_common",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            androidCommon,
            allowedRepositories));

    for (Map.Entry<String, Object> provider : providers.entrySet()) {
      builder.put(
          provider.getKey(),
          FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
              BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API, provider.getValue()));
    }
  }
}
