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

/**
 * Provider of transitively available ZIPs of native libs that should be directly copied into the
 * APK.
 */
@StarlarkBuiltin(
    name = "AndroidNativeLibsInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidNativeLibsInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidNativeLibsInfo";

  @StarlarkMethod(
      name = "native_libs",
      doc = "Returns the native libraries produced by the rule.",
      documented = false,
      structField = true)
  Depset /*<FileT>*/ getNativeLibsForStarlark();

  /** Provider for {@link AndroidNativeLibsInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface AndroidNativeLibsInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "AndroidNativeLibsInfo",
        doc = "The <code>AndroidNativeLibsInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "native_libs",
              allowedTypes = {@ParamType(type = Depset.class, generic1 = FileApi.class)},
              named = true,
              doc = "The native libraries produced by the rule."),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidNativeLibsInfoApi<?> createInfo(Depset nativeLibs) throws EvalException;
  }
}
