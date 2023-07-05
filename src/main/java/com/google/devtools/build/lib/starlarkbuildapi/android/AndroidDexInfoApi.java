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
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;

/** An Info that can provides dex artifacts. */
@StarlarkBuiltin(
    name = "AndroidDexInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidDexInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidDexInfo";

  @StarlarkMethod(
      name = "deploy_jar",
      doc = "The deploy jar.",
      documented = false,
      structField = true)
  FileT getDeployJar();

  @StarlarkMethod(
      name = "final_classes_dex_zip",
      doc = "The zip file containing the final dex classes.",
      documented = false,
      structField = true)
  FileT getFinalClassesDexZip();

  @Nullable
  @StarlarkMethod(
      name = "java_resource_jar",
      doc = "The final Java resource jar.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getJavaResourceJar();

  /** Provider for {@link AndroidDexInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidDexInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "deploy_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true,
              doc = "The \"_deploy\" jar suitable for deployment."),
          @Param(
              name = "final_classes_dex_zip",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true,
              doc = "The zip file containing the final dex classes."),
          @Param(
              name = "java_resource_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The final Java resource jar."),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidDexInfoApi<FileT> createInfo(
        FileT deployJar, FileT finalClassesDexZip, Object javaResourceJar) throws EvalException;
  }
}
