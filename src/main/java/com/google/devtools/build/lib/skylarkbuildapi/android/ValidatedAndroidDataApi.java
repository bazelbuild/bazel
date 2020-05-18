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
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Validated Android data which can be merged together with assets from dependencies. */
@StarlarkBuiltin(
    name = "ValidatedAndroidDataApi",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Validated Android data which can be merged together with assets from dependencies.",
    documented = false,
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ValidatedAndroidDataApi<
        FileT extends FileApi,
        AndroidResourcesInfoT extends
            AndroidResourcesInfoApi<
                    FileT,
                    ? extends ValidatedAndroidDataApi<FileT, AndroidResourcesInfoT>,
                    ? extends AndroidManifestInfoApi<FileT>>>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "to_provider",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  AndroidResourcesInfoT toProvider();

  @StarlarkMethod(
      name = "r_txt",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getRTxt();

  @StarlarkMethod(
      name = "java_class_jar",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getJavaClassJar();

  @StarlarkMethod(
      name = "java_source_jar",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getJavaSourceJar();

  @StarlarkMethod(
      name = "apk",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getApk();

  @StarlarkMethod(
      name = "aapt2_r_txt",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getAapt2RTxt();

  @StarlarkMethod(
      name = "aapt2_java_source_jar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getAapt2SourceJar();

  @StarlarkMethod(
      name = "static_library",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getStaticLibrary();

  @StarlarkMethod(
      name = "resources",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  Sequence<FileT> getResourcesList();
}
