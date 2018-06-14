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
        "Provides access to information about Android rules. Every Android-related target provides "
            + "this struct, accessible as a <code>android</code> field on a "
            + "<a href=\"Target.html\">target</a>.")
public interface AndroidSkylarkApiProviderApi<FileT extends FileApi> {

  @SkylarkCallable(
      name = "apk",
      structField = true,
      allowReturnNones = true,
      doc = "Returns an APK produced by this target.")
  FileT getApk();

  @SkylarkCallable(
      name = "java_package",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a java package for this target.")
  String getJavaPackage();

  @SkylarkCallable(
      name = "manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target.")
  FileT getManifest();

  @SkylarkCallable(
      name = "merged_manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target after all processing, e.g.: merging, etc.")
  FileT getMergedManifest();

  @SkylarkCallable(
      name = "native_libs",
      structField = true,
      doc =
          "Returns the native libraries as a dictionary of the libraries' architecture as a string "
              + "to a set of the native library files, or the empty dictionary if there are no "
              + "native libraries.")
  ImmutableMap<String, NestedSet<FileT>> getNativeLibs();

  @SkylarkCallable(
      name = "resource_apk",
      structField = true,
      doc = "Returns the resources container for the target.")
  FileT getResourceApk();

  @SkylarkCallable(
      name = "apks_under_test",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a collection of APKs that this target tests.")
  ImmutableCollection<FileT> getApksUnderTest();

  @SkylarkCallable(
      name = "defines_resources",
      structField = true,
      doc = "Returns <code>True</code> if the target defines any Android resources directly.")
  boolean definesAndroidResources();

  @SkylarkCallable(
      name = "idl",
      structField = true,
      doc = "Returns information about IDL files associated with this target.")
  IdlInfoApi<FileT> getIdlInfo();

  @SkylarkCallable(
      name = "resources",
      structField = true,
      doc = "Returns resources defined by this target.")
  NestedSet<FileT> getResources();

  @SkylarkCallable(
      name = "resource_jar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a jar file for classes generated from resources.")
  @Nullable
  OutputJarApi<FileT> getResourceJar();

  @SkylarkCallable(
      name = "aar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns the aar output of this target.")
  FileT getAar();

  /** Helper class to provide information about IDLs related to this rule. */
  @SkylarkModule(
      name = "AndroidSkylarkIdlInfo",
      category = SkylarkModuleCategory.NONE,
      doc = "Provides access to information about Android rules.")
  interface IdlInfoApi<FileT extends FileApi> {
    @SkylarkCallable(
        name = "import_root",
        structField = true,
        allowReturnNones = true,
        doc = "Returns the root of IDL packages if not the java root.")
    String getImportRoot();

    @SkylarkCallable(name = "sources", structField = true, doc = "Returns a list of IDL files.")
    ImmutableCollection<FileT> getSources();

    @SkylarkCallable(
        name = "generated_java_files",
        structField = true,
        doc = "Returns a list Java files generated from IDL sources.")
    ImmutableCollection<FileT> getIdlGeneratedJavaFiles();

    @SkylarkCallable(
        name = "output",
        structField = true,
        allowReturnNones = true,
        doc = "Returns a jar file for classes generated from IDL sources.")
    @Nullable
    OutputJarApi<FileT> getIdlOutput();
  }
}
