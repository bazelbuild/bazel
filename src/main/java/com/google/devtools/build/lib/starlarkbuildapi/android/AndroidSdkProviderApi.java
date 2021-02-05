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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.TransitiveInfoCollectionApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;

/**
 * Configured targets implementing this provider can contribute Android Sdk information to the
 * compilation.
 */
@StarlarkBuiltin(
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
  @StarlarkMethod(name = "build_tools_version", structField = true, doc = "", documented = false)
  String getBuildToolsVersion();

  @StarlarkMethod(
      name = "framework_aidl",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getFrameworkAidl();

  @StarlarkMethod(
      name = "aidl_lib",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  TransT getAidlLib();

  @StarlarkMethod(name = "android_jar", structField = true, doc = "", documented = false)
  FileT getAndroidJar();

  @StarlarkMethod(
      name = "source_properties",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getSourceProperties();

  @StarlarkMethod(name = "shrinked_android_jar", structField = true, doc = "", documented = false)
  FileT getShrinkedAndroidJar();

  @StarlarkMethod(name = "main_dex_classes", structField = true, doc = "", documented = false)
  FileT getMainDexClasses();

  @StarlarkMethod(name = "adb", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAdb();

  @StarlarkMethod(name = "dx", structField = true, doc = "", documented = false)
  FilesToRunProviderT getDx();

  @StarlarkMethod(name = "main_dex_list_creator", structField = true, doc = "", documented = false)
  FilesToRunProviderT getMainDexListCreator();

  @StarlarkMethod(name = "aidl", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAidl();

  @StarlarkMethod(name = "aapt", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAapt();

  @StarlarkMethod(name = "aapt2", structField = true, doc = "", documented = false)
  FilesToRunProviderT getAapt2();

  @StarlarkMethod(
      name = "apk_builder",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderT getApkBuilder();

  @StarlarkMethod(name = "apk_signer", structField = true, doc = "", documented = false)
  FilesToRunProviderT getApkSigner();

  @StarlarkMethod(name = "proguard", structField = true, doc = "", documented = false)
  FilesToRunProviderT getProguard();

  @StarlarkMethod(name = "zip_align", structField = true, doc = "", documented = false)
  FilesToRunProviderT getZipalign();

  @StarlarkMethod(
      name = "legacy_main_dex_list_generator",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FilesToRunProviderT getLegacyMainDexListGenerator();

  /** The provider implementing this can construct the AndroidSdkInfo provider. */
  @StarlarkBuiltin(
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

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidSdkInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "build_tools_version",
              doc = "A string of the build tools version.",
              positional = true,
              named = false),
          @Param(
              name = "framework_aidl",
              doc = "An artifact of the AIDL framework.",
              positional = true,
              named = false),
          @Param(
              name = "aidl_lib",
              doc = "A transitive info collection of the AIDL lib.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = TransitiveInfoCollectionApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "android_jar",
              doc = "An artifact of the Android Jar.",
              positional = true,
              named = false),
          @Param(
              name = "sourceProperties",
              doc = "An artifact of the AIDL lib.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "shrinked_android_jar",
              doc = "An artifact of the shrunk Android Jar.",
              positional = true,
              named = false),
          @Param(
              name = "main_dex_classes",
              doc = "An artifact of the main dex classes.",
              positional = true,
              named = false),
          @Param(
              name = "adb",
              doc = "A files to run provider of ADB.",
              positional = true,
              named = false),
          @Param(
              name = "dx",
              doc = "A files to run provider of Dx.",
              positional = true,
              named = false),
          @Param(
              name = "main_dex_list_creator",
              doc = "A files to run provider of the main dex list creator.",
              positional = true,
              named = false),
          @Param(
              name = "aidl",
              doc = "A files to run provider of AIDL.",
              positional = true,
              named = false),
          @Param(
              name = "aapt",
              doc = "A files to run provider of AAPT.",
              positional = true,
              named = false),
          @Param(
              name = "aapt2",
              doc = "A files to run provider of AAPT2.",
              positional = true,
              named = false),
          @Param(
              name = "apk_builder",
              doc = "A files to run provider of the Apk builder.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FilesToRunProviderApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "apk_signer",
              doc = "A files to run provider of the Apk signer.",
              positional = true,
              named = false),
          @Param(
              name = "proguard",
              doc = "A files to run provider of Proguard.",
              positional = true,
              named = false),
          @Param(
              name = "zipalign",
              doc = "A files to run provider of Zipalign.",
              positional = true,
              named = false),
          @Param(
              name = "system",
              doc = "",
              defaultValue = "None",
              positional = true,
              named = false),
          @Param(
              name = "legacy_main_dex_list_generator",
              defaultValue = "None",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FilesToRunProviderApi.class),
                @ParamType(type = NoneType.class),
              }),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidSdkProviderApi<FileT, FilesToRunProviderT, TransT> createInfo(
        String buildToolsVersion,
        FileT frameworkAidl,
        Object aidlLib,
        FileT androidJar,
        Object sourceProperties,
        FileT shrinkedAndroidJar,
        FileT mainDexClasses,
        FilesToRunProviderT adb,
        FilesToRunProviderT dx,
        FilesToRunProviderT mainDexListCreator,
        FilesToRunProviderT aidl,
        FilesToRunProviderT aapt,
        FilesToRunProviderT aapt2,
        Object apkBuilder,
        FilesToRunProviderT apkSigner,
        FilesToRunProviderT proguard,
        FilesToRunProviderT zipalign,
        Object system,
        Object legacyMainDexListGenerator)
        throws EvalException;
  }
}
