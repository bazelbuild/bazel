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

package com.google.devtools.build.lib.starlarkbuildapi.proto;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** The interface for {@code .proto} source files in Starlark. */
@StarlarkBuiltin(
    name = "ProtoSource",
    category = DocCategory.BUILTIN,
    doc = "Represents a single <code>.proto</code> source file.")
public interface ProtoSourceApi<FileT extends FileApi> extends StarlarkValue {
  @StarlarkMethod(
      name = "file",
      structField = true,
      doc = "The <code>.proto</code> source file.")
  FileT getSourceFile();

  @StarlarkMethod(
      name = "import_path",
      structField = true,
      doc = "The <code>import path</code> of this <code>.proto</code> source file.")
  String getImportPathForStarlark();
}
