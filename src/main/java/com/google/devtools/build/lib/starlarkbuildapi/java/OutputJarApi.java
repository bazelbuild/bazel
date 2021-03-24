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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** A tuple of a java classes jar and its associated source and interface archives. */
@StarlarkBuiltin(
    name = "java_output",
    category = DocCategory.BUILTIN,
    doc = "Java classes jar, together with their associated source and interface archives.")
public interface OutputJarApi<FileT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(name = "class_jar", doc = "A classes jar file.", structField = true)
  FileT getClassJar();

  @StarlarkMethod(
      name = "ijar",
      doc = "A interface jar file.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getIJar();

  @StarlarkMethod(
      name = "manifest_proto",
      doc =
          "A manifest proto file. The protobuf file containing the manifest generated from "
              + "JavaBuilder.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getManifestProto();

  @StarlarkMethod(
      name = "source_jar",
      doc =
          "A sources archive file. Deprecated. Kept for migration reasons. "
              + "Please use source_jars instead.",
      allowReturnNones = true,
      structField = true)
  @Deprecated
  @Nullable
  FileT getSrcJar();

  @StarlarkMethod(
      name = "source_jars",
      doc = "A list of sources archive files.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  Sequence<FileT> getSrcJarsStarlark();
}
