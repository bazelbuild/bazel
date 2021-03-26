// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidIdeInfoProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Fake implementation of AndroidIdeInfoProvider. */
public class FakeAndroidIdeInfoProvider
    implements AndroidIdeInfoProviderApi<FileApi, JavaOutputApi<FileApi>> {

  @Nullable
  @Override
  public String getJavaPackage() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getManifest() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getGeneratedManifest() {
    return null;
  }

  @Nullable
  @Override
  public String getIdlImportRoot() {
    return null;
  }

  @Override
  public ImmutableCollection<FileApi> getIdlSrcs() {
    return null;
  }

  @Override
  public ImmutableCollection<FileApi> getIdlGeneratedJavaFiles() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getIdlSourceJar() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getIdlClassJar() {
    return null;
  }

  @Override
  public boolean definesAndroidResources() {
    return false;
  }

  @Nullable
  @Override
  public JavaOutputApi<FileApi> getResourceJarJavaOutput() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getResourceApk() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getSignedApk() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getAar() {
    return null;
  }

  @Override
  public ImmutableCollection<FileApi> getApksUnderTest() {
    return null;
  }

  @Override
  public ImmutableMap<String, Depset> getNativeLibsStarlark() {
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

  /** Fake implementation of {@link AndroidIdeInfoProviderApi.Provider}. */
  public static class FakeProvider
      implements AndroidIdeInfoProviderApi.Provider<FileApi, JavaOutputApi<FileApi>> {

    @Override
    public AndroidIdeInfoProviderApi<FileApi, JavaOutputApi<FileApi>> createInfo(
        Object javaPackage,
        Object manifest,
        Object generatedManifest,
        Object idlImportRoot,
        Sequence<?> idlSrcs,
        Sequence<?> idlGeneratedJavaFiles,
        Object idlSourceJar,
        Object idlClassJar,
        boolean definesAndroidResources,
        Object resourceJar,
        Object resourceApk,
        Object signedApk,
        Object aar,
        Sequence<?> apksUnderTest,
        Dict<?, ?> nativeLibs)
        throws EvalException {
      return new FakeAndroidIdeInfoProvider();
    }
  }
}
