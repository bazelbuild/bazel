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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidIdlProviderApi;
import net.starlark.java.eval.EvalException;

/** Fake implementation of AndroidIdlProvider. */
public class FakeAndroidIdlProvider implements AndroidIdlProviderApi<FileApi> {

  @Override
  public Depset getTransitiveIdlImportRootsForStarlark() {
    return null;
  }

  @Override
  public Depset getTransitiveIdlImportsForStarlark() {
    return null;
  }

  @Override
  public Depset getTransitiveIdlJarsForStarlark() {
    return null;
  }

  @Override
  public Depset getTransitiveIdlPreprocessedForStarlark() {
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

  /** Fake implementation of {@link AndroidIdlProviderApi.Provider}. */
  public static class FakeProvider implements AndroidIdlProviderApi.Provider<FileApi> {

    @Override
    public AndroidIdlProviderApi<FileApi> createInfo(
        Depset transitiveIdlImportRoots,
        Depset transitiveIdlImports,
        Depset transitiveIdlJars,
        Depset transitiveIdlPreprocessed)
        throws EvalException {
      return new FakeAndroidIdlProvider();
    }
  }
}
