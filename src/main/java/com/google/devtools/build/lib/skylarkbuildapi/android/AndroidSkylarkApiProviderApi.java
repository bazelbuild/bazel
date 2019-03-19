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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.OutputJarApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import javax.annotation.Nullable;

/**
 * A class that exposes the Android providers to Skylark. It is intended to provide a simple and
 * stable interface for Skylark users.
 */
@SkylarkModule(
    name = "AndroidSkylarkApiProvider",
    title = "android",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Provides access to information about Android rules. Every Android-related target "
            + "provides this struct, accessible as a <code>android</code> field on a "
            + "<a href=\"Target.html\">target</a>.",
    documented = false)
public interface AndroidSkylarkApiProviderApi<FileT extends FileApi> {

  @SkylarkCallable(
      name = "apk",
      structField = true,
      allowReturnNones = true,
      doc = "Returns an APK produced by this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getApk();

  @SkylarkCallable(
      name = "java_package",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a java package for this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  String getJavaPackage();

  @SkylarkCallable(
      name = "manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getManifest();

  @SkylarkCallable(
      name = "merged_manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target after all processing, e.g.: merging, etc.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getMergedManifest();

  @SkylarkCallable(
      name = "native_libs",
      structField = true,
      doc =
          "Returns the native libraries as a dictionary of the libraries' architecture as a string "
              + "to a set of the native library files, or the empty dictionary if there are no "
              + "native libraries.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  ImmutableMap<String, NestedSet<FileT>> getNativeLibs();

  @SkylarkCallable(
      name = "resource_apk",
      structField = true,
      doc = "Returns the resources container for the target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
      allowReturnNones = true)
  @Nullable
  FileT getResourceApk();

  @SkylarkCallable(
      name = "apks_under_test",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a collection of APKs that this target tests.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  ImmutableCollection<FileT> getApksUnderTest();

  @SkylarkCallable(
      name = "defines_resources",
      structField = true,
      doc = "Returns <code>True</code> if the target defines any Android resources directly.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  boolean definesAndroidResources();

  @SkylarkCallable(
      name = "idl",
      structField = true,
      doc = "Returns information about IDL files associated with this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  IdlInfoApi<FileT> getIdlInfo();

  @SkylarkCallable(
      name = "resources",
      structField = true,
      doc = "Returns resources defined by this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getResources();

  @SkylarkCallable(
      name = "resource_jar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a jar file for classes generated from resources.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  OutputJarApi<FileT> getResourceJar();

  @SkylarkCallable(
      name = "aar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns the aar output of this target.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getAar();

  /** Helper class to provide information about IDLs related to this rule. */
  @SkylarkModule(
      name = "AndroidSkylarkIdlInfo",
      category = SkylarkModuleCategory.NONE,
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed."
              + "Provides access to information about Android rules.",
      documented = false)
  interface IdlInfoApi<FileT extends FileApi> {
    @SkylarkCallable(
        name = "import_root",
        structField = true,
        allowReturnNones = true,
        doc = "Returns the root of IDL packages if not the java root.",
        documented = false,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
    @Nullable
    String getImportRoot();

    @SkylarkCallable(
        name = "sources",
        structField = true,
        doc = "Returns a list of IDL files.",
        documented = false,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
    ImmutableCollection<FileT> getSources();

    @SkylarkCallable(
        name = "generated_java_files",
        structField = true,
        doc = "Returns a list Java files generated from IDL sources.",
        documented = false,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
    ImmutableCollection<FileT> getIdlGeneratedJavaFiles();

    @SkylarkCallable(
        name = "output",
        structField = true,
        allowReturnNones = true,
        doc = "Returns a jar file for classes generated from IDL sources.",
        documented = false,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
    @Nullable
    OutputJarApi<FileT> getIdlOutput();
  }
}
