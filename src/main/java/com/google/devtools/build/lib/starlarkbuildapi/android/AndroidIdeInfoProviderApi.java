// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;

/**
 * Configured targets implementing this provider can contribute Android-specific info to IDE to the
 * compilation.
 */
@StarlarkBuiltin(
    name = "AndroidIdeInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidIdeInfoProviderApi<
        FileT extends FileApi, JavaOutputT extends JavaOutputApi<FileT>>
    extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidIdeInfo";

  /** Returns the Java package. */
  @StarlarkMethod(
      name = "java_package",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  String getJavaPackage();

  /** Returns the direct AndroidManifest. */
  @StarlarkMethod(
      name = "manifest",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getManifest();

  /** Returns the direct generated AndroidManifest. */
  @StarlarkMethod(
      name = "generated_manifest",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getGeneratedManifest();

  @StarlarkMethod(
      name = "idl_import_root",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  String getIdlImportRoot();

  /** A list of sources from the "idl_srcs" attribute. */
  @StarlarkMethod(name = "idl_srcs", structField = true, doc = "", documented = false)
  ImmutableCollection<FileT> getIdlSrcs();

  /** A list of java files generated from the "idl_srcs" attribute. */
  @StarlarkMethod(
      name = "idl_generated_java_files",
      structField = true,
      doc = "",
      documented = false)
  ImmutableCollection<FileT> getIdlGeneratedJavaFiles();

  @StarlarkMethod(
      name = "idl_source_jar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getIdlSourceJar();

  @StarlarkMethod(
      name = "idl_class_jar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getIdlClassJar();

  /**
   * Returns true if the target defined Android resources. Exposes {@link
   * LocalResourceContainer#definesAndroidResources(AttributeMap)}
   */
  @StarlarkMethod(
      name = "defines_android_resources",
      structField = true,
      doc = "",
      documented = false)
  boolean definesAndroidResources();

  @StarlarkMethod(
      name = "resource_jar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  JavaOutputT getResourceJarJavaOutput();

  @StarlarkMethod(
      name = "resource_apk",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getResourceApk();

  /** Returns the direct debug key signed apk, if there is one. */
  @StarlarkMethod(
      name = "signed_apk",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getSignedApk();

  @StarlarkMethod(
      name = "aar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getAar();

  /** A list of the APKs related to the app under test, if any. */
  @StarlarkMethod(name = "apks_under_test", structField = true, doc = "", documented = false)
  ImmutableCollection<FileT> getApksUnderTest();

  /** A map, keyed on architecture, of the native libs for the app, if any. */
  @StarlarkMethod(name = "native_libs", structField = true, doc = "", documented = false)
  ImmutableMap<String, Depset> getNativeLibsStarlark();

  /** The provider implementing this can construct the AndroidIdeInfo provider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi, OutputJarT extends JavaOutputApi<FileT>>
      extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidIdeInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "java_package",
              doc = "A string of the Java package.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = String.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "manifest",
              doc = "An artifact of the Android manifest.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "generated_manifest",
              doc = "An artifact of the generated Android manifest.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "idl_import_root",
              doc = "A string of the idl import root.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = String.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "idl_srcs",
              doc = "A list of artifacts of the idl srcs.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)}),
          @Param(
              name = "idl_generated_java_files",
              doc = "A list of artifacts of the idl generated java files.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)}),
          @Param(
              name = "idl_source_jar",
              doc = "An artifact of the source Jar with the idl generated java files.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "idl_class_jar",
              doc = "An artifact of the class Jar with the compiled idl generated java files.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "defines_android_resources",
              doc = "A boolean if target specifies Android resources.",
              positional = true,
              named = false),
          @Param(
              name = "resource_jar",
              doc = "An artifact of the Jar containing Android resources.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = JavaOutputApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "resource_apk",
              doc = "An artifact of the Apk containing Android resources.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "signed_apk",
              doc = "An artifact of the signed Apk.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "aar",
              doc = "An artifact of the Android archive.",
              positional = true,
              named = false,
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "apks_under_test",
              doc = "A list of artifacts of the apks under test",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)}),
          @Param(
              name = "native_libs",
              doc =
                  "A dictionary of string to a list of artifacts mapping architectures to native "
                      + "libs.",
              positional = true,
              named = false,
              allowedTypes = {@ParamType(type = Dict.class, generic1 = String.class)})
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidIdeInfoProviderApi<FileT, OutputJarT> createInfo(
        /*noneable*/ Object javaPackage,
        /*noneable*/ Object manifest,
        /*noneable*/ Object generatedManifest,
        /*noneable*/ Object idlImportRoot,
        Sequence<?> idlSrcs, // <FileT>
        Sequence<?> idlGeneratedJavaFiles, // <FileT>
        /*noneable*/ Object idlSourceJar,
        /*noneable*/ Object idlClassJar,
        boolean definesAndroidResources,
        /*noneable*/ Object resourceJar,
        /*noneable*/ Object resourceApk,
        /*noneable*/ Object signedApk,
        /*noneable*/ Object aar,
        Sequence<?> apksUnderTest, // <FileT>
        Dict<?, ?> nativeLibs /* <String, Depset> */)
        throws EvalException;
  }
}
