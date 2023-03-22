// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** Supplies a optimized jar from Android Rules. */
@StarlarkBuiltin(
    name = "AndroidOptimizedJarInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidOptimizedJarInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidOptimizedJarInfo";

  @StarlarkMethod(
      name = "optimized_jar",
      doc = "The optimized jar.",
      documented = false,
      structField = true)
  FileT getOptimizedJar();

  /** Provider for {@link AndroidOptimizedJarInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidOptimizedJarInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "optimized_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true,
              doc = "The optimized jar."),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidOptimizedJarInfoApi<FileT> createInfo(FileT optimizedJar) throws EvalException;
  }
}
