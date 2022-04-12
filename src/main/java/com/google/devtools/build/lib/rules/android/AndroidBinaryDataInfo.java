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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidBinaryDataInfoApi;
import net.starlark.java.eval.EvalException;

/**
 * Provides information on Android resource, asset, and manifest information specific to binaries.
 *
 * <p>This includes both android_binary targets and other top-level targets (such as
 * android_local_test)
 */
public class AndroidBinaryDataInfo extends NativeInfo
    implements AndroidBinaryDataInfoApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

  private final Artifact dataApk;
  private final Artifact resourceProguardConfig;

  private final AndroidResourcesInfo resourcesInfo;
  private final AndroidAssetsInfo assetsInfo;
  private final AndroidManifestInfo manifestInfo;

  public static AndroidBinaryDataInfo of(
      Artifact dataApk,
      Artifact resourceProguardConfig,
      AndroidResourcesInfo resourcesInfo,
      AndroidAssetsInfo assetsInfo,
      AndroidManifestInfo manifestInfo) {
    return new AndroidBinaryDataInfo(
        dataApk, resourceProguardConfig, resourcesInfo, assetsInfo, manifestInfo);
  }

  private AndroidBinaryDataInfo(
      Artifact dataApk,
      Artifact resourceProguardConfig,
      AndroidResourcesInfo resourcesInfo,
      AndroidAssetsInfo assetsInfo,
      AndroidManifestInfo manifestInfo) {
    this.dataApk = dataApk;
    this.resourceProguardConfig = resourceProguardConfig;
    this.resourcesInfo = resourcesInfo;
    this.assetsInfo = assetsInfo;
    this.manifestInfo = manifestInfo;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public Artifact getApk() {
    return dataApk;
  }

  @Override
  public Artifact getResourceProguardConfig() {
    return resourceProguardConfig;
  }

  public AndroidResourcesInfo getResourcesInfo() {
    return resourcesInfo;
  }

  public AndroidAssetsInfo getAssetsInfo() {
    return assetsInfo;
  }

  public AndroidManifestInfo getManifestInfo() {
    return manifestInfo;
  }

  public AndroidBinaryDataInfo withShrunkApk(Artifact shrunkApk) {
    return new AndroidBinaryDataInfo(
        shrunkApk, resourceProguardConfig, resourcesInfo, assetsInfo, manifestInfo);
  }

  /** Provider class for {@link AndroidBinaryDataInfo} objects. */
  public static class Provider extends BuiltinProvider<AndroidBinaryDataInfo>
      implements AndroidBinaryDataInfoApi.Provider<
          Artifact, AndroidResourcesInfo, AndroidAssetsInfo, AndroidManifestInfo> {

    private Provider() {
      super(NAME, AndroidBinaryDataInfo.class);
    }

    @Override
    public AndroidBinaryDataInfoApi<Artifact> create(
        Artifact resourceApk,
        Artifact resourceProguardConfig,
        AndroidResourcesInfo resourcesInfo,
        AndroidAssetsInfo assetsInfo,
        AndroidManifestInfo manifestInfo)
        throws EvalException {
      return new AndroidBinaryDataInfo(
          resourceApk, resourceProguardConfig, resourcesInfo, assetsInfo, manifestInfo);
    }
  }
}
