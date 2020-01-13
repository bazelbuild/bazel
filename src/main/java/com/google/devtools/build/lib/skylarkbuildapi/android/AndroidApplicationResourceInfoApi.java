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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;

/** Supplies a resource apk file (".ap_") and related info. */
@SkylarkModule(
    name = "AndroidApplicationResourceInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidApplicationResourceInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidApplicationResourceInfo";

  /** Returns the ap_ artifact to be merged into the final apk. */
  @SkylarkCallable(
      name = "resource_apk",
      doc = "The resource apk file ('.ap_')",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileT getResourceApk();

  /** The jar containing the R java source files. */
  @SkylarkCallable(
      name = "resource_java_src_jar",
      doc = "The jar containing the R java source files.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileT getResourceJavaSrcJar();

  /** The jar containing the R java class files. */
  @SkylarkCallable(
      name = "resource_java_class_jar",
      doc = "The jar containing the R java class files.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileT getResourceJavaClassJar();

  /** The final proessed manifest. */
  @SkylarkCallable(
      name = "manifest",
      doc = "The final processed manifest.",
      documented = false,
      structField = true)
  FileT getManifest();

  /** The proguard config for Android resources. */
  @SkylarkCallable(
      name = "resource_proguard_config",
      doc = "The resource proguard config file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileT getResourceProguardConfig();

  /** The main dex proguard config file. */
  @SkylarkCallable(
      name = "main_dex_proguard_config",
      doc = "The main dex proguard config file.",
      documented = false,
      allowReturnNones = true,
      structField = true)
  FileT getMainDexProguardConfig();

  /** Provider for {@link AndroidApplicationResourceInfoApi}. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface AndroidApplicationResourceInfoApiProvider<FileT extends FileApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>AndroidApplicationResourceInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "resource_apk",
              type = FileApi.class,
              noneable = true,
              named = true,
              doc = ""),
          @Param(
              name = "resource_java_src_jar",
              type = FileApi.class,
              noneable = true,
              named = true,
              doc = ""),
          @Param(
              name = "resource_java_class_jar",
              type = FileApi.class,
              noneable = true,
              named = true,
              doc = ""),
          @Param(name = "manifest", type = FileApi.class, named = true, doc = ""),
          @Param(
              name = "resource_proguard_config",
              type = FileApi.class,
              noneable = true,
              named = true,
              doc = ""),
          @Param(
              name = "main_dex_proguard_config",
              type = FileApi.class,
              noneable = true,
              named = true,
              doc = ""),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidApplicationResourceInfoApi.class)
    AndroidApplicationResourceInfoApi<FileT> createInfo(
        Object resourceApk,
        Object resourceJavaSrcJar,
        Object resourceJavaClassJar,
        FileT manifest,
        Object resourceProguardConfig,
        Object mainDexProguardConfig)
        throws EvalException;
  }
}
