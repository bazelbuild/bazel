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

package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDeviceBrokerInfoApi.AndroidDeviceBrokerInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidInstrumentationInfoApi.AndroidInstrumentationInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidNativeLibsInfoApi.AndroidNativeLibsInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidResourcesInfoApi.AndroidResourcesInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.ApkInfoApi.ApkInfoApiProvider;

/**
 * {@link Bootstrap} for skylark objects related to Android rules.
 */
public class AndroidBootstrap implements Bootstrap {

  private final AndroidSkylarkCommonApi<?, ?> androidCommon;
  private final ApkInfoApiProvider apkInfoProvider;
  private final AndroidInstrumentationInfoApiProvider<?> androidInstrumentationInfoProvider;
  private final AndroidDeviceBrokerInfoApiProvider androidDeviceBrokerInfoProvider;
  private final AndroidResourcesInfoApi.AndroidResourcesInfoApiProvider<?, ?, ?>
      androidResourcesInfoProvider;
  private final AndroidNativeLibsInfoApiProvider androidNativeLibsInfoProvider;

  public AndroidBootstrap(
      AndroidSkylarkCommonApi<?, ?> androidCommon,
      ApkInfoApiProvider apkInfoProvider,
      AndroidInstrumentationInfoApiProvider<?> androidInstrumentationInfoProvider,
      AndroidDeviceBrokerInfoApiProvider androidDeviceBrokerInfoProvider,
      AndroidResourcesInfoApiProvider<?, ?, ?> androidResourcesInfoProvider,
      AndroidNativeLibsInfoApiProvider androidNativeLibsInfoProvider) {
    this.androidCommon = androidCommon;
    this.apkInfoProvider = apkInfoProvider;
    this.androidInstrumentationInfoProvider = androidInstrumentationInfoProvider;
    this.androidDeviceBrokerInfoProvider = androidDeviceBrokerInfoProvider;
    this.androidResourcesInfoProvider = androidResourcesInfoProvider;
    this.androidNativeLibsInfoProvider = androidNativeLibsInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put("android_common", androidCommon);
    builder.put(ApkInfoApi.NAME, apkInfoProvider);
    builder.put(AndroidInstrumentationInfoApi.NAME, androidInstrumentationInfoProvider);
    builder.put(AndroidDeviceBrokerInfoApi.NAME, androidDeviceBrokerInfoProvider);
    builder.put(AndroidResourcesInfoApi.NAME, androidResourcesInfoProvider);
    builder.put(AndroidNativeLibsInfoApi.NAME, androidNativeLibsInfoProvider);
  }
}
