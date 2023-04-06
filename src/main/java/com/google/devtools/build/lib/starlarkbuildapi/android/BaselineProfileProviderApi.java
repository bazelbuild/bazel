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
import net.starlark.java.eval.NoneType;

/** Provides information about baseline profile for Android binaries. */
@StarlarkBuiltin(
    name = "BaselineProfileProvider",
    doc = "Baseline profile file used for Android binaries.",
    category = DocCategory.PROVIDER)
public interface BaselineProfileProviderApi<FileT extends FileApi> extends StructApi {

  String NAME = "BaselineProfileProvider";

  @StarlarkMethod(name = "files", structField = true, doc = "", documented = false)
  Depset /*<FileT>*/ getTransitiveBaselineProfilesForStarlark();

  /** The provider implementing this can construct the BaselineProfileProvider. */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>BaselineProfileProvider</code> constructor.",
        parameters = {
          @Param(
              name = "files",
              doc = "Transitive baseline profile files.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Depset.class, generic1 = FileApi.class)}),
          @Param(
              name = "art_profile_zip",
              doc =
                  "The final ART profile zip to be packaged in the APK. Optional, only used for"
                      + " migration purposes.",
              positional = true,
              named = false,
              defaultValue = "None",
              allowedTypes = {@ParamType(type = FileApi.class), @ParamType(type = NoneType.class)})
        },
        selfCall = true)
    @StarlarkConstructor
    BaselineProfileProviderApi<FileT> create(Depset files, Object artProfileZip)
        throws EvalException;
  }
}
