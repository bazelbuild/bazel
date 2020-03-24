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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;

/**
 * Configured targets implementing this provider can contribute Android Sdk information to the
 * compilation.
 */
@SkylarkModule(
    name = "AndroidSdkInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidSdkProviderApi<
        FileT extends FileApi,
        FilesToRunProviderT extends FilesToRunProviderApi<FileT>,
        TransT extends TransitiveInfoCollectionApi>
    extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidSdkInfo";

  /** The value of build_tools_version. May be null or empty. */
  @SkylarkCallable(name = "build_tools_version", structField = true, doc = "", documented = false)
  String getBuildToolsVersion();

  @SkylarkCallable(
      name = "framework_aidl",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  FileT getFrameworkAidl();

  @SkylarkCallable(
      name = "aidl_lib",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  TransT getAidlLib();

  @SkylarkCallable(name = "android_jar", structField = true, doc = "", documented = false)
  FileT getAndroidJar();

  @SkylarkCallable(
      name = "source_properties",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getSourceProperties();

  @SkylarkCallable(name = "shrinked_android_jar", structField = true, doc = "", documented = false)
  FileT getShrinkedAndroidJar();

  @SkylarkCallable(name = "main_dex_classes", structField = true, doc = "", documented = false)
  FileT getMainDexClasses();

  @SkylarkCallable(name = "adb", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAdb();

  @SkylarkCallable(name = "dx", structField = true, doc = "", documented = false)
  FilesToRunProviderT getDx();

  @SkylarkCallable(name = "main_dex_list_creator", structField = true, doc = "", documented = false)
  FilesToRunProviderT getMainDexListCreator();

  @SkylarkCallable(name = "aidl", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAidl();

  @SkylarkCallable(name = "aapt", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAapt();

  @SkylarkCallable(name = "aapt2", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAapt2();

  @SkylarkCallable(
      name = "apk_builder",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderT getApkBuilder();

  @SkylarkCallable(name = "apk_signer", structField = true, doc = "", documented = false)
  FilesToRunProviderT getApkSigner();

  @SkylarkCallable(name = "proguard", structField = true, doc = "", documented = false)
  FilesToRunProviderT getProguard();

  @SkylarkCallable(name = "zip_align", structField = true, doc = "", documented = false)
  FilesToRunProviderT getZipalign();

  /** The provider implementing this can construct the AndroidSdkInfo provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<
          FileT extends FileApi,
          FilesToRunProviderT extends FilesToRunProviderApi<FileT>,
          TransT extends TransitiveInfoCollectionApi>
      extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>AndroidSdkInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "build_tools_version",
              doc = "A string of the build tools version.",
              positional = true,
              named = false,
              type = String.class),
          @Param(
              name = "framework_aidl",
              doc = "An artifact of the AIDL framework.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "aidl_lib",
              doc = "A transitive info collection of the AIDL lib.",
              positional = true,
              named = false,
              type = TransitiveInfoCollectionApi.class,
              noneable = true),
          @Param(
              name = "android_jar",
              doc = "An artifact of the Android Jar.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "sourceProperties",
              doc = "An artifact of the AIDL lib.",
              positional = true,
              named = false,
              type = FileApi.class,
              noneable = true),
          @Param(
              name = "shrinked_android_jar",
              doc = "An artifact of the shrunk Android Jar.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "main_dex_classes",
              doc = "An artifact of the main dex classes.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "adb",
              doc = "A files to run provider of ADB.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "dx",
              doc = "A files to run provider of Dx.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "main_dex_list_creator",
              doc = "A files to run provider of the main dex list creator.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "aidl",
              doc = "A files to run provider of AIDL.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "aapt",
              doc = "A files to run provider of AAPT.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "aapt2",
              doc = "A files to run provider of AAPT2.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "apk_builder",
              doc = "A files to run provider of the Apk builder.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class,
              noneable = true),
          @Param(
              name = "apk_signer",
              doc = "A files to run provider of the Apk signer.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "proguard",
              doc = "A files to run provider of Proguard.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "zipalign",
              doc = "A files to run provider of Zipalign.",
              positional = true,
              named = false,
              type = FilesToRunProviderApi.class),
          @Param(
              name = "system",
              doc = "",
              noneable = true,
              defaultValue = "None",
              positional = true,
              named = false,
              type = StarlarkValue.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidSdkProviderApi.class)
    AndroidSdkProviderApi<FileT, FilesToRunProviderT, TransT> createInfo(
        String buildToolsVersion,
        FileT frameworkAidl,
        /*noneable*/ Object aidlLib,
        FileT androidJar,
        /*noneable*/ Object sourceProperties,
        FileT shrinkedAndroidJar,
        FileT mainDexClasses,
        FilesToRunProviderT adb,
        FilesToRunProviderT dx,
        FilesToRunProviderT mainDexListCreator,
        FilesToRunProviderT aidl,
        FilesToRunProviderT aapt,
        FilesToRunProviderT aapt2,
        /*noneable*/ Object apkBuilder,
        FilesToRunProviderT apkSigner,
        FilesToRunProviderT proguard,
        FilesToRunProviderT zipalign,
        /*noneable*/ Object system)
        throws EvalException;
  }
}
