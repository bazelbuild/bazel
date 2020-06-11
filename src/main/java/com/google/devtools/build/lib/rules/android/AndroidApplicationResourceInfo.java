// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidApplicationResourceInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** A provider for Android resource APKs (".ap_") and related info. */
@Immutable
public class AndroidApplicationResourceInfo extends NativeInfo
    implements AndroidApplicationResourceInfoApi<Artifact> {

  /** Singleton instance of the provider type for {@link AndroidApplicationResourceInfo}. */
  public static final AndroidApplicationResourceInfoProvider PROVIDER =
      new AndroidApplicationResourceInfoProvider();

  private final Artifact resourceApk;
  private final Artifact resourceJavaSrcJar;
  private final Artifact resourceJavaClassJar;
  private final Artifact manifest;
  private final Artifact resourceProguardConfig;
  private final Artifact mainDexProguardConfig;

  AndroidApplicationResourceInfo(
      Artifact resourceApk,
      Artifact resourceJavaSrcJar,
      Artifact resourceJavaClassJar,
      Artifact manifest,
      Artifact resourceProguardConfig,
      Artifact mainDexProguardConfig) {
    super(PROVIDER);
    this.resourceApk = resourceApk;
    this.resourceJavaSrcJar = resourceJavaSrcJar;
    this.resourceJavaClassJar = resourceJavaClassJar;
    this.manifest = manifest;
    this.resourceProguardConfig = resourceProguardConfig;
    this.mainDexProguardConfig = mainDexProguardConfig;
  }

  @Override
  public Artifact getResourceApk() {
    return resourceApk;
  }

  @Override
  public Artifact getResourceJavaSrcJar() {
    return resourceJavaSrcJar;
  }

  @Override
  public Artifact getResourceJavaClassJar() {
    return resourceJavaClassJar;
  }

  @Override
  public Artifact getManifest() {
    return manifest;
  }

  @Override
  public Artifact getResourceProguardConfig() {
    return resourceProguardConfig;
  }

  @Override
  public Artifact getMainDexProguardConfig() {
    return mainDexProguardConfig;
  }

  /** Provider for {@link AndroidApplicationResourceInfo}. */
  public static class AndroidApplicationResourceInfoProvider
      extends BuiltinProvider<AndroidApplicationResourceInfo>
      implements AndroidApplicationResourceInfoApiProvider<Artifact> {

    private AndroidApplicationResourceInfoProvider() {
      super(AndroidApplicationResourceInfoApi.NAME, AndroidApplicationResourceInfo.class);
    }

    @Override
    public AndroidApplicationResourceInfoApi<Artifact> createInfo(
        Object resourceApk,
        Object resourceJavaSrcJar,
        Object resourceJavaClassJar,
        Artifact manifest,
        Object resourceProguardConfig,
        Object mainDexProguardConfig)
        throws EvalException {

      return new AndroidApplicationResourceInfo(
          fromNoneable(resourceApk, Artifact.class),
          fromNoneable(resourceJavaSrcJar, Artifact.class),
          fromNoneable(resourceJavaClassJar, Artifact.class),
          manifest,
          fromNoneable(resourceProguardConfig, Artifact.class),
          fromNoneable(mainDexProguardConfig, Artifact.class));
    }
  }
}
