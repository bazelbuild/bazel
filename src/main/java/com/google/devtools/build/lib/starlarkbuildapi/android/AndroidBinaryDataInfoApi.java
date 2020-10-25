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
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/**
 * Provides information on Android resource, asset, and manifest information specific to binaries.
 */
@StarlarkBuiltin(
    name = "AndroidBinaryData",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Information about Android resource, asset, and manifest information specific to "
            + "binaries",
    documented = false,
    category = DocCategory.PROVIDER)
public interface AndroidBinaryDataInfoApi<FileT extends FileApi> extends StructApi {

  /** The name of the provider for this info object. */
  String NAME = "AndroidBinaryData";

  @StarlarkMethod(
      name = "resource_apk",
      structField = true,
      doc = "The resource apk.",
      documented = false)
  FileT getApk();

  @StarlarkMethod(
      name = "resource_proguard_config",
      structField = true,
      doc = "Proguard config generated for the resources.",
      documented = false)
  FileT getResourceProguardConfig();

  /** The provider implementing this can construct the AndroidBinaryDataInfoApi provider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<
          FileT extends FileApi,
          AndroidResourcesInfoApiT,
          AndroidAssetsInfoApiT,
          AndroidManifestInfoApiT>
      extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidBinaryDataInfoApi</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "resource_apk", doc = "resource_apk", positional = false, named = true),
          @Param(
              name = "resource_proguard_config",
              doc = "resource_proguard_config",
              positional = false,
              named = true),
          @Param(name = "resources_info", doc = "resources_info", positional = false, named = true),
          @Param(name = "assets_info", doc = "assets_info", positional = false, named = true),
          @Param(name = "manifest_info", doc = "manifest_info", positional = false, named = true),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidBinaryDataInfoApi<FileT> create(
        FileT resourceApk,
        FileT resourceProguardConfig,
        AndroidResourcesInfoApiT resourcesInfo,
        AndroidAssetsInfoApiT assetsInfo,
        AndroidManifestInfoApiT manifestInfo)
        throws EvalException;
  }
}
