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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/**
 * Interface for C++ debug related objects, specifically when fission is used.
 *
 * <p>It is not expected for this to be used externally at this time. This API is experimental and
 * subject to change, and its usage should be restricted to internal packages.
 *
 * <p>See javadoc for {@link com.google.devtools.build.lib.rules.cpp.CcModule}.
 */
@StarlarkBuiltin(name = "CcNativeLibraryInfo", category = DocCategory.PROVIDER, documented = false)
public interface CcNativeLibraryInfoApi extends StructApi {
  String NAME = "CcNativeLibraryInfo";

  /** Provider implementing this will be able to construct CcNativeLibraryInfo objects */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  interface Provider extends ProviderApi {
    @StarlarkMethod(
        name = NAME,
        documented = false,
        selfCall = true,
        parameters = {
          @Param(
              name = "libraries_to_link",
              positional = false,
              named = true,
              allowedTypes = {@ParamType(type = Depset.class)})
        })
    @StarlarkConstructor
    CcNativeLibraryInfoApi createCcNativeLibraryInfo(Object librariesToLinkObject)
        throws EvalException;
  }
}
