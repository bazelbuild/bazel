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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.EvalException;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** A provider of information about this target's manifest. */
@StarlarkBuiltin(
    name = "AndroidManifestInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Information about the Android manifest provided by a rule.",
    documented = false,
    category = StarlarkDocumentationCategory.PROVIDER)
public interface AndroidManifestInfoApi<FileT extends FileApi> extends StructApi {

  /** The name of the provider for this info object. */
  String NAME = "AndroidManifestInfo";

  @StarlarkMethod(
      name = "manifest",
      doc = "This target's manifest, merged with manifests from dependencies",
      documented = false,
      structField = true)
  FileT getManifest();

  @StarlarkMethod(
      name = "package",
      doc = "This target's package",
      documented = false,
      structField = true)
  String getPackage();

  @StarlarkMethod(
      name = "exports_manifest",
      doc = "If this manifest should be exported to targets that depend on it",
      documented = false,
      structField = true)
  boolean exportsManifest();

  /** Provider for {@link AndroidManifestInfoApi} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = "AndroidManifestInfo",
        doc = "The <code>AndroidManifestInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "manifest", positional = true, named = true, type = FileApi.class),
          @Param(name = "package", positional = true, named = true, type = String.class),
          @Param(
              name = "exports_manifest",
              positional = true,
              named = true,
              defaultValue = "False",
              type = Boolean.class),
        },
        selfCall = true)
    AndroidManifestInfoApi<FileT> androidManifestInfo(
        FileT manifest, String packageString, Boolean exportsManifest) throws EvalException;
  }
}
