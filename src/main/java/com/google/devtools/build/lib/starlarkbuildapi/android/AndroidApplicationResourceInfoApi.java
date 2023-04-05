// Copyright 2019 The Bazel Authors. All rights reserved.
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

/** Supplies a resource apk file (".ap_") and related info. */
@StarlarkBuiltin(
    name = "AndroidApplicationResourceInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidApplicationResourceInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidApplicationResourceInfo";

  /** Returns the ap_ artifact to be merged into the final apk. */
  @StarlarkMethod(
      name = "resource_apk",
      doc = "The resource apk file ('.ap_')",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getResourceApk();

  /** The jar containing the R java source files. */
  @StarlarkMethod(
      name = "resource_java_src_jar",
      doc = "The jar containing the R java source files.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getResourceJavaSrcJar();

  /** The jar containing the R java class files. */
  @StarlarkMethod(
      name = "resource_java_class_jar",
      doc = "The jar containing the R java class files.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getResourceJavaClassJar();

  /** The final proessed manifest. */
  @StarlarkMethod(
      name = "manifest",
      doc = "The final processed manifest.",
      documented = false,
      structField = true)
  FileT getManifest();

  /** The proguard config for Android resources. */
  @StarlarkMethod(
      name = "resource_proguard_config",
      doc = "The resource proguard config file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getResourceProguardConfig();

  /** The main dex proguard config file. */
  @StarlarkMethod(
      name = "main_dex_proguard_config",
      doc = "The main dex proguard config file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getMainDexProguardConfig();

  /** The R.txt file. */
  @StarlarkMethod(
      name = "r_txt",
      doc = "The R.txt file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getRTxt();

  /** The merged resource files zip. */
  @StarlarkMethod(
      name = "resources_zip",
      doc = "The merged resource files zip.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getResourcesZip();

  /** The databinding layout info file */
  @StarlarkMethod(
      name = "databinding_info",
      doc = "The databinding layout info file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getDatabindingLayoutInfoZip();

  /** The build stamp jar file */
  @StarlarkMethod(
      name = "build_stamp_jar",
      doc = "The build stamp jar file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileT getBuildStampJar();

  /** Whether to compile Java Srcs within the android_binary rule */
  @StarlarkMethod(
      name = "should_compile_java_srcs",
      doc = "Whether to compile Java Srcs within the android_binary rule.",
      documented = false,
      allowReturnNones = false,
      structField = true)
  boolean shouldCompileJavaSrcs();

  /** Provider for {@link AndroidApplicationResourceInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface AndroidApplicationResourceInfoApiProvider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidApplicationResourceInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "resource_apk",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = ""),
          @Param(
              name = "resource_java_src_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = ""),
          @Param(
              name = "resource_java_class_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = ""),
          @Param(name = "manifest", named = true, doc = ""),
          @Param(
              name = "resource_proguard_config",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = ""),
          @Param(
              name = "main_dex_proguard_config",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = ""),
          @Param(
              name = "r_txt",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
          @Param(
              name = "resources_zip",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
          @Param(
              name = "databinding_info",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
          @Param(
              name = "build_stamp_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "",
              defaultValue = "None"),
          @Param(name = "should_compile_java_srcs", named = true, doc = "", defaultValue = "True"),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidApplicationResourceInfoApi<FileT> createInfo(
        Object resourceApk,
        Object resourceJavaSrcJar,
        Object resourceJavaClassJar,
        FileT manifest,
        Object resourceProguardConfig,
        Object mainDexProguardConfig,
        Object rTxt,
        Object resourcesZip,
        Object databindingLayoutInfoZip,
        Object buildStampJar,
        boolean shouldCompileJava)
        throws EvalException;
  }
}
