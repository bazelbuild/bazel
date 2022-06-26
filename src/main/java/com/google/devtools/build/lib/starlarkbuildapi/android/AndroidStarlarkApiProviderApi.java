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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/**
 * A class that exposes the Android providers to Starlark. It is intended to provide a simple and
 * stable interface for Starlark users.
 */
@StarlarkBuiltin(
    name = "AndroidStarlarkApiProvider",
    category = DocCategory.PROVIDER,
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Provides access to information about Android rules. Every Android-related target "
            + "provides this struct, accessible as a <code>android</code> field on a "
            + "<a href=\"Target.html\">target</a>.",
    documented = false)
public interface AndroidStarlarkApiProviderApi<FileT extends FileApi> extends StarlarkValue {

  @StarlarkMethod(
      name = "apk",
      structField = true,
      allowReturnNones = true,
      doc = "Returns an APK produced by this target.",
      documented = false)
  @Nullable
  FileT getApk();

  @StarlarkMethod(
      name = "java_package",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a java package for this target.",
      documented = false)
  @Nullable
  String getJavaPackage();

  @StarlarkMethod(
      name = "manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target.",
      documented = false)
  @Nullable
  FileT getManifest();

  @StarlarkMethod(
      name = "merged_manifest",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a manifest file for this target after all processing, e.g.: merging, etc.",
      documented = false)
  @Nullable
  FileT getMergedManifest();

  @StarlarkMethod(
      name = "native_libs",
      structField = true,
      doc =
          "Returns the native libraries as a dictionary of the libraries' architecture as a string "
              + "to a set of the native library files, or the empty dictionary if there are no "
              + "native libraries.",
      documented = false)
  ImmutableMap<String, Depset> getNativeLibs();

  @StarlarkMethod(
      name = "resource_apk",
      structField = true,
      doc = "Returns the resources container for the target.",
      documented = false,
      allowReturnNones = true)
  @Nullable
  FileT getResourceApk();

  @StarlarkMethod(
      name = "apks_under_test",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a collection of APKs that this target tests.",
      documented = false)
  @Nullable
  ImmutableCollection<FileT> getApksUnderTest();

  @StarlarkMethod(
      name = "defines_resources",
      structField = true,
      doc = "Returns <code>True</code> if the target defines any Android resources directly.",
      documented = false)
  boolean definesAndroidResources();

  @StarlarkMethod(
      name = "idl",
      structField = true,
      doc = "Returns information about IDL files associated with this target.",
      documented = false)
  IdlInfoApi<FileT> getIdlInfo();

  @StarlarkMethod(
      name = "resources",
      structField = true,
      doc = "Returns resources defined by this target.",
      documented = false)
  Depset /*<FileT>*/ getResources();

  @StarlarkMethod(
      name = "resource_jar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a jar file for classes generated from resources.",
      documented = false)
  @Nullable
  JavaOutputApi<FileT> getResourceJar();

  @StarlarkMethod(
      name = "aar",
      structField = true,
      allowReturnNones = true,
      doc = "Returns the aar output of this target.",
      documented = false)
  @Nullable
  FileT getAar();

  /** Helper class to provide information about IDLs related to this rule. */
  @StarlarkBuiltin(
      name = "AndroidStarlarkIdlInfo",
      category = DocCategory.NONE,
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed."
              + "Provides access to information about Android rules.",
      documented = false)
  interface IdlInfoApi<FileT extends FileApi> extends StarlarkValue {
    @StarlarkMethod(
        name = "import_root",
        structField = true,
        allowReturnNones = true,
        doc = "Returns the root of IDL packages if not the java root.",
        documented = false)
    @Nullable
    String getImportRoot();

    @StarlarkMethod(
        name = "sources",
        structField = true,
        doc = "Returns a list of IDL files.",
        documented = false)
    ImmutableCollection<FileT> getSources();

    @StarlarkMethod(
        name = "generated_java_files",
        structField = true,
        doc = "Returns a list Java files generated from IDL sources.",
        documented = false)
    ImmutableCollection<FileT> getIdlGeneratedJavaFiles();

    @StarlarkMethod(
        name = "output",
        structField = true,
        allowReturnNones = true,
        doc = "Returns a jar file for classes generated from IDL sources.",
        documented = false)
    @Nullable
    JavaOutputApi<FileT> getIdlOutput();
  }
}
