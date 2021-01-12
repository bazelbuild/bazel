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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LibraryToLinkApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LtoBackendArtifactsApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** A target that provides C++ {@link LibraryToLinkApi}s to be linked into Java targets. */
@StarlarkBuiltin(
    name = "JavaNativeLibraryInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Information about the C++ libraries to be linked into Java targets.",
    documented = true,
    category = DocCategory.PROVIDER)
public interface JavaNativeLibraryInfoApi<
        FileT extends FileApi,
        LtoBackendArtifactsT extends LtoBackendArtifactsApi<FileT>,
        LibraryToLinkT extends LibraryToLinkApi<FileT, LtoBackendArtifactsT>>
    extends StructApi {
  /** Name of this info object. */
  String NAME = "JavaNativeLibraryInfo";

  /** Returns the cc linking info */
  @StarlarkMethod(
      name = "transitive_libraries",
      structField = true,
      doc = "Returns the set of transitive LibraryToLink objects.",
      documented = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  Depset /*<LibraryToLinkT>*/ getTransitiveJavaNativeLibrariesForStarlark();

  /** The provider implementing this can construct the JavaNativeLibraryInfo provider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<
          FileT extends FileApi,
          LtoBackendArtifactsT extends LtoBackendArtifactsApi<FileT>,
          LibraryToLinkT extends LibraryToLinkApi<FileT, LtoBackendArtifactsT>>
      extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>JavaNativeLibraryInfo</code> constructor.",
        documented = true,
        enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
        parameters = {
          @Param(
              name = "transitive_libraries",
              doc = "The transitive set of LibraryToLink providers.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Depset.class, generic1 = LibraryToLinkApi.class)}),
        },
        selfCall = true)
    @StarlarkConstructor
    JavaNativeLibraryInfoApi<FileT, LtoBackendArtifactsT, LibraryToLinkT> create(
        Depset transitiveLibraries) throws EvalException;
  }
}
