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

package com.google.devtools.build.skydoc.fakebuildapi.java;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaNativeLibraryInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;

/** Fake implementation of {@link JavaNativeLibraryInfoApi}. */
public class FakeJavaNativeLibraryInfo
    implements JavaNativeLibraryInfoApi<FileApi, LibraryToLinkApi<FileApi>> {

  @Override
  public Depset getTransitiveJavaNativeLibrariesForStarlark() {
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

  /** Fake implementation of {@link JavaNativeLibraryInfoApi.Provider}. */
  public static class Provider
      implements JavaNativeLibraryInfoApi.Provider<FileApi, LibraryToLinkApi<FileApi>> {

    @Override
    public JavaNativeLibraryInfoApi<FileApi, LibraryToLinkApi<FileApi>> create(
        Depset transitiveLibraries) throws EvalException {
      return null;
    }
  }
}
