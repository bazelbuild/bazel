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

import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;

/** A provider for native libs for android_binary. */
@StarlarkBuiltin(
    name = "AndroidBinaryNativeLibsInfoApi",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidBinaryNativeLibsInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidBinaryNativeLibsInfo";

  @Nullable
  @StarlarkMethod(
      name = "native_libs",
      documented = false,
      allowReturnNones = true,
      structField = true)
  Dict<String, Depset> getNativeLibsStarlark();

  @Nullable
  @StarlarkMethod(
      name = "native_libs_name",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileApi getNativeLibsNameStarlark();

  @Nullable
  @StarlarkMethod(
      name = "transitive_native_libs",
      documented = false,
      allowReturnNones = true,
      structField = true)
  Depset getTransitiveNativeLibsStarlark();

  /** Provider for {@link AndroidBinaryNativeLibsInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidBinaryNativeLibsInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "native_libs",
              allowedTypes = {
                @ParamType(type = Dict.class),
              },
              named = true,
              doc = ""),
          @Param(
              name = "native_libs_name",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
          @Param(
              name = "transitive_native_libs",
              allowedTypes = {
                @ParamType(type = Depset.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidBinaryNativeLibsInfoApi<FileT> createInfo(
        Dict<?, ?> nativeLibs, Object nativeLibsName, Object transitiveNativeLibs)
        throws EvalException;
  }
}
