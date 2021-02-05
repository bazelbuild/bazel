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

package com.google.devtools.build.skydoc.fakebuildapi.android;

import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSdkProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.TransitiveInfoCollectionApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Fake implementation of AndroidSdkProvider. */
public class FakeAndroidSdkProvider
    implements AndroidSdkProviderApi<
        FileApi, FilesToRunProviderApi<FileApi>, TransitiveInfoCollectionApi> {

  @Override
  public String getBuildToolsVersion() {
    return null;
  }

  @Override
  public FileApi getFrameworkAidl() {
    return null;
  }

  @Nullable
  @Override
  public TransitiveInfoCollectionApi getAidlLib() {
    return null;
  }

  @Override
  public FileApi getAndroidJar() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getSourceProperties() {
    return null;
  }

  @Override
  public FileApi getShrinkedAndroidJar() {
    return null;
  }

  @Override
  public FileApi getMainDexClasses() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getAdb() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getDx() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getMainDexListCreator() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getAidl() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getAapt() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getAapt2() {
    return null;
  }

  @Nullable
  @Override
  public FilesToRunProviderApi<FileApi> getApkBuilder() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getApkSigner() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getProguard() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<FileApi> getZipalign() {
    return null;
  }

  @Nullable
  @Override
  public FilesToRunProviderApi<FileApi> getLegacyMainDexListGenerator() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return null;
  }

  @Override
  public String toJson() throws EvalException {
    return null;
  }

  /** The provider can construct the fake Android SDK provider. */
  public static class FakeProvider
      implements AndroidSdkProviderApi.Provider<
          FileApi, FilesToRunProviderApi<FileApi>, TransitiveInfoCollectionApi> {

    @Override
    public AndroidSdkProviderApi<
            FileApi, FilesToRunProviderApi<FileApi>, TransitiveInfoCollectionApi>
        createInfo(
            String buildToolsVersion,
            FileApi frameworkAidl,
            Object aidlLib,
            FileApi androidJar,
            Object sourceProperties,
            FileApi shrinkedAndroidJar,
            FileApi mainDexClasses,
            FilesToRunProviderApi<FileApi> adb,
            FilesToRunProviderApi<FileApi> dx,
            FilesToRunProviderApi<FileApi> mainDexListCreator,
            FilesToRunProviderApi<FileApi> aidl,
            FilesToRunProviderApi<FileApi> aapt,
            FilesToRunProviderApi<FileApi> aapt2,
            Object apkBuilder,
            FilesToRunProviderApi<FileApi> apkSigner,
            FilesToRunProviderApi<FileApi> proguard,
            FilesToRunProviderApi<FileApi> zipalign,
            Object system,
            Object legacyMainDexListGenerator)
            throws EvalException {
      return new FakeAndroidSdkProvider();
    }
  }
}
