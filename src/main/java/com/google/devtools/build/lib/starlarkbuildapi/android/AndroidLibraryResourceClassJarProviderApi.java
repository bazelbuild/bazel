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

/** Provides resource class jars from android_library rules. */
@StarlarkBuiltin(
    name = "AndroidLibraryResourceClassJarProvider",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Resource class jars from android_library rules.",
    documented = false,
    category = DocCategory.PROVIDER)
public interface AndroidLibraryResourceClassJarProviderApi<FileT extends FileApi>
    extends StructApi {
  /** The name of the provider for this info object. */
  String NAME = "AndroidLibraryResourceClassJarProvider";

  @StarlarkMethod(name = "jars", structField = true, doc = "", documented = false)
  Depset /*<FileT>*/ getResourceClassJarsForStarlark();

  /** The provider implementing this can construct the AndroidLibraryResourceClassJarProvider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidLibraryResourceClassJarProvider</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "jars",
              doc = "Resource class jars.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Depset.class, generic1 = FileApi.class)}),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidLibraryResourceClassJarProviderApi<FileT> create(Depset jars) throws EvalException;
  }
}
