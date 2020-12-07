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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a structured representation of the linking outputs of a C++ rule. */
@StarlarkBuiltin(
    name = "CcLinkingOutputs",
    category = DocCategory.BUILTIN,
    documented = true,
    doc = "Helper class containing CC compilation outputs.")
public interface CcLinkingOutputsApi<
        FileT extends FileApi, LtoBackendArtifactsT extends LtoBackendArtifactsApi<FileT>>
    extends StarlarkValue {
  @StarlarkMethod(
      name = "library_to_link",
      structField = true,
      allowReturnNones = true,
      doc =
          "<a href='LibraryToLink.html'><code>LibraryToLink</code></a> for including these outputs "
              + "in further linking.",
      documented = true)
  @Nullable
  LibraryToLinkApi<FileT, LtoBackendArtifactsT> getLibraryToLink();

  @StarlarkMethod(
      name = "executable",
      structField = true,
      allowReturnNones = true,
      doc = "<a href='File.html'><code>File</code></a> object representing the linked executable.",
      documented = true)
  @Nullable
  FileT getExecutable();
}
