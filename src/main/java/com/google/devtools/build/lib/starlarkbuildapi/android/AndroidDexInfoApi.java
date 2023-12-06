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

  @Nullable
  @StarlarkMethod(
      name = "final_classes_dex_zip",
      doc = "The zip file containing the final dex classes.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getFinalClassesDexZip();

  @Nullable
  @StarlarkMethod(
      name = "filtered_deploy_jar",
      doc = "The filtered deploy jar.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getFilteredDeployJar();

  @Nullable
  @StarlarkMethod(
      name = "final_proguard_output_map",
      doc = "The final proguard output map.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getFinalProguardOutputMap();

  @Nullable
  @StarlarkMethod(
      name = "java_resource_jar",
      doc = "The final Java resource jar.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getJavaResourceJar();

  @Nullable
  @StarlarkMethod(
      name = "shuffled_java_resource_jar",
      doc = "The output java resource jar after shuffling the proguarded jar.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getShuffledJavaResourceJar();

  @Nullable
  @StarlarkMethod(
      name = "rex_output_package_map",
      doc = "The output rex package map.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getRexOutputPackageMap();

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
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The zip file containing the final dex classes."),
          @Param(
              name = "filtered_deploy_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The filtered deploy jar.",
              defaultValue = "None"),
          @Param(
              name = "final_proguard_output_map",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The final proguard output map.",
              defaultValue = "None"),
          @Param(
              name = "java_resource_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The final Java resource jar.",
              defaultValue = "None"),
          @Param(
              name = "shuffled_java_resource_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The output java resource jar after shuffling the proguarded jar.",
              defaultValue = "None"),
          @Param(
              name = "rex_output_package_map",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The output rex package map.",
              defaultValue = "None"),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidDexInfoApi<FileT> createInfo(
        FileT deployJar,
        Object finalClassesDexZip,
        Object filteredDeployJar,
        Object finalProguardOutputMap,
        Object javaResourceJar,
        Object shuffledJavaResourceJar,
        Object rexOutputPackageMap)
        throws EvalException;
  }
}
