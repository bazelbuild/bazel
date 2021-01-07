// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.NoneType;

/**
 * Supplies the pregenerate_oat_files_for_tests attribute of type boolean provided by android_device
 * rule.
 */
@StarlarkBuiltin(
    name = "AndroidDex2OatInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidDex2OatInfoApi<
        FileT extends FileApi, FilesToRunProviderT extends FilesToRunProviderApi<FileT>>
    extends StructApi {
  /** The name of the provider for this info object. */
  String NAME = "AndroidDex2OatInfo";

  /** Provider for {@link AndroidDex2OatInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface Provider<
          FileT extends FileApi, FilesToRunProviderT extends FilesToRunProviderApi<FileT>>
      extends ProviderApi {
    @StarlarkMethod(
        name = "AndroidDex2OatInfo",
        doc = "The <code>AndroidDex2OatInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "enabled", positional = false, named = true),
          @Param(
              name = "execute_dex2oat_on_host",
              positional = false,
              named = true,
              defaultValue = "False"),
          @Param(
              name = "sandbox_for_pregenerating_oat_files_for_tests",
              positional = false,
              named = true,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = FilesToRunProviderApi.class),
              },
              defaultValue = "None"),
          @Param(
              name = "framework",
              positional = false,
              named = true,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = FileApi.class),
              },
              defaultValue = "None"),
          @Param(
              name = "dalvik_cache",
              positional = false,
              named = true,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = FileApi.class),
              },
              defaultValue = "None"),
          @Param(
              name = "device_props",
              positional = false,
              named = true,
              allowedTypes = {
                @ParamType(type = NoneType.class),
                @ParamType(type = FileApi.class),
              },
              defaultValue = "None"),
        },
        selfCall = true)
    AndroidDex2OatInfoApi<FileT, FilesToRunProviderT> androidDex2OatInfo(
        Boolean enabled,
        Boolean executeDex2OatOnHost,
        /*noneable*/ Object sandboxForPregeneratingOatFilesForTests,
        /*noneable*/ Object framework,
        /*noneable*/ Object dalvikCache,
        /*noneable*/ Object deviceProps);
  }
}
