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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidAssetsInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.ParsedAndroidAssetsApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Fake implementation of AndroidAssetsInfo. */
public class FakeAndroidAssetsInfo
    implements AndroidAssetsInfoApi<FileApi, ParsedAndroidAssetsApi> {

  @Override
  public Label getLabel() {
    return null;
  }

  @Nullable
  @Override
  public FileApi getValidationResult() {
    return null;
  }

  @Override
  public Depset getDirectParsedAssetsForStarlark() {
    return null;
  }

  @Override
  public ImmutableList<FileApi> getLocalAssets() {
    return null;
  }

  @Override
  public String getLocalAssetDir() {
    return null;
  }

  @Override
  public Depset getTransitiveParsedAssetsForStarlark() {
    return null;
  }

  @Override
  public Depset getAssetsForStarlark() {
    return null;
  }

  @Override
  public Depset getSymbolsForStarlark() {
    return null;
  }

  @Override
  public Depset getCompiledSymbolsForStarlark() {
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

  /** Fake implementation of {@link AndroidAssetsInfoApi.Provider}. */
  public static class FakeProvider
      implements AndroidAssetsInfoApi.Provider<FileApi, ParsedAndroidAssetsApi> {

    @Override
    public AndroidAssetsInfoApi<FileApi, ParsedAndroidAssetsApi> createInfo(
        Label label,
        Object validationResult,
        Depset directParsedAssets,
        Depset transitiveParsedAssets,
        Depset transitiveAssets,
        Depset transitiveSymbols,
        Depset transitiveCompiledSymbols)
        throws EvalException {
      return new FakeAndroidAssetsInfo();
    }
  }
}
