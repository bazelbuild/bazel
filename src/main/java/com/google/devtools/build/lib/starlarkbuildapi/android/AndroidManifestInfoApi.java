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
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** A provider of information about this target's manifest. */
@StarlarkBuiltin(
    name = "AndroidManifestInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Information about the Android manifest provided by a rule.",
    documented = false,
    category = DocCategory.PROVIDER)
public interface AndroidManifestInfoApi<FileT extends FileApi> extends StructApi {

  /** The name of the provider for this info object. */
  String NAME = "AndroidManifestInfo";

  @StarlarkMethod(
      name = "manifest",
      doc = "This target's manifest, merged with manifests from dependencies",
      documented = false,
      structField = true)
  FileT getManifest();

  @Nullable
  @StarlarkMethod(
      name = "package",
      doc = "This target's package",
      documented = false,
      structField = true,
      allowReturnNones = true)
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
          @Param(name = "manifest", positional = true, named = true),
          @Param(name = "package", positional = true, named = true),
          @Param(
              name = "exports_manifest",
              positional = true,
              named = true,
              defaultValue = "False"),
        },
        selfCall = true)
    AndroidManifestInfoApi<FileT> androidManifestInfo(
        FileT manifest, String packageString, Boolean exportsManifest) throws EvalException;
  }
}
