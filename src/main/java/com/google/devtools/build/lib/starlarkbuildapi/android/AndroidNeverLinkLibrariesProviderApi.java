// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** Provides information about neverlink libraries for Android targets. */
@StarlarkBuiltin(
    name = "AndroidNeverLinkLibrariesProvider",
    category = DocCategory.PROVIDER,
    doc = "Information about neverlink libraries for Android targets.")
public interface AndroidNeverLinkLibrariesProviderApi<FileT extends FileApi> extends StructApi {

  /** The name of the provider for this info object. */
  static final String NAME = "AndroidNeverLinkLibrariesProvider";

  @StarlarkMethod(
      name = "transitive_neverlink_libraries",
      structField = true,
      doc = "",
      documented = false)
  Depset /*<FileT>*/ getTransitiveNeverLinkLibrariesForStarlark();

  /** Provider for {@link AndroidNeverLinkLibrariesProvider}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {
    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidNeverLinkLibrariesProvider</code> constructor.",
        parameters = {
          @Param(
              name = "transitive_neverlink_libraries",
              doc = "The transitive neverlink libraries",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Depset.class, generic1 = FileApi.class)}),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidNeverLinkLibrariesProviderApi<FileT> create(Depset transitiveNeverlinkLibraries)
        throws EvalException;
  }
}
