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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Returns information about executables produced by a target and the files needed to run it. */
@StarlarkBuiltin(
    name = "FilesToRunProvider",
    doc =
"""
Contains information about executables produced by a target and the files needed to run it. This \
provider can not be created directly, it is an implicit output of executable targets accessible \
via <a href="../providers/DefaultInfo.html#files_to_run"><code>DefaultInfo.files_to_run</code></a>.
""",
    category = DocCategory.PROVIDER)
public interface FilesToRunProviderApi<FileT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(
      name = "executable",
      doc = "The main executable or None if it does not exist.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getExecutable();

  @StarlarkMethod(
      name = "runfiles_manifest",
      doc = "The runfiles manifest or None if it does not exist.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getRunfilesManifest();

  @StarlarkMethod(
      name = "repo_mapping_manifest",
      doc = "The repo mapping manifest or None if it does not exist.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getRepoMappingManifest();
}
