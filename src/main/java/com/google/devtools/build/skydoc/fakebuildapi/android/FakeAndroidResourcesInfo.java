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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidManifestInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidResourcesInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.ValidatedAndroidDataApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/** Fake implementation of {@link AndroidResourcesInfoApi}. */
public class FakeAndroidResourcesInfo
    implements AndroidResourcesInfoApi<
        FileApi,
        FakeAndroidResourcesInfo.FakeValidatedAndroidDataApi,
        FakeAndroidResourcesInfo.FakeAndroidManifestInfoApi> {

  public static final String PROVIDER_NAME = "FakeAndroidResourcesInfo";
  public static final FakeAndroidResourcesInfoProvider PROVIDER =
      new FakeAndroidResourcesInfoProvider();

  @Override
  public Label getLabel() {
    return null;
  }

  @Override
  public FakeAndroidManifestInfoApi getManifest() {
    return null;
  }

  @Override
  public FileApi getRTxt() {
    return null;
  }

  @Override
  public NestedSet<FakeValidatedAndroidDataApi> getTransitiveAndroidResources() {
    return null;
  }

  @Override
  public NestedSet<FakeValidatedAndroidDataApi> getDirectAndroidResources() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveResources() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveManifests() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveAapt2RTxt() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveSymbolsBin() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveCompiledSymbols() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveStaticLib() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveRTxt() {
    return null;
  }

  @Override
  public String toProto(Location loc) throws EvalException {
    return "";
  }

  @Override
  public String toJson(Location loc) throws EvalException {
    return "";
  }

  @Override
  public void repr(SkylarkPrinter printer) {}

  /** Fake implementation of {@link AndroidResourcesInfoApiProvider}. */
  public static class FakeAndroidResourcesInfoProvider
      implements AndroidResourcesInfoApi.AndroidResourcesInfoApiProvider<
          FileApi,
          FakeAndroidResourcesInfo.FakeValidatedAndroidDataApi,
          FakeAndroidResourcesInfo.FakeAndroidManifestInfoApi> {

    @Override
    public FakeAndroidResourcesInfo createInfo(
        Label label,
        FakeAndroidManifestInfoApi manifest,
        FileApi rTxt,
        SkylarkNestedSet transitiveAndroidResources,
        SkylarkNestedSet directAndroidResources,
        SkylarkNestedSet transitiveResources,
        SkylarkNestedSet transitiveManifests,
        SkylarkNestedSet transitiveAapt2RTxt,
        SkylarkNestedSet transitiveSymbolsBin,
        SkylarkNestedSet transitiveCompiledSymbols,
        SkylarkNestedSet transitiveStaticLib,
        SkylarkNestedSet transitiveRTxt)
        throws EvalException {
      return new FakeAndroidResourcesInfo();
    }

    @Override
    public void repr(SkylarkPrinter printer) {}
  }

  /** Fake implementation of {@link ValidatedAndroidDataApi}. */
  public static class FakeValidatedAndroidDataApi implements ValidatedAndroidDataApi<FileApi> {

    @Override
    public FileApi getRTxt() {
      return null;
    }

    @Override
    public FileApi getJavaSourceJar() {
      return null;
    }

    @Override
    public FileApi getApk() {
      return null;
    }

    @Override
    public FileApi getAapt2RTxt() {
      return null;
    }

    @Override
    public FileApi getAapt2SourceJar() {
      return null;
    }

    @Override
    public FileApi getStaticLibrary() {
      return null;
    }

    @Override
    public SkylarkList<FileApi> getResourcesList() {
      return null;
    }
  }

  /** Fake implementation of {@link AndroidManifestInfoApi}. */
  public static class FakeAndroidManifestInfoApi implements AndroidManifestInfoApi<FileApi> {

    @Override
    public FileApi getManifest() {
      return null;
    }

    @Override
    public boolean exportsManifest() {
      return false;
    }

    @Override
    public String getPackage() {
      return null;
    }

    @Override
    public String toProto(Location loc) throws EvalException {
      return "";
    }

    @Override
    public String toJson(Location loc) throws EvalException {
      return "";
    }

    @Override
    public void repr(SkylarkPrinter printer) {}
  }
}
