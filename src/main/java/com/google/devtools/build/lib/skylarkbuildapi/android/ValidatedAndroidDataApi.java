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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import javax.annotation.Nullable;

/** Validated Android data which can be merged together with assets from dependencies. */
@SkylarkModule(
    name = "ValidatedAndroidDataApi",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Validated Android data which can be merged together with assets from dependencies.",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface ValidatedAndroidDataApi<FileT extends FileApi> {

  @SkylarkCallable(
      name = "r_txt",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getRTxt();

  @SkylarkCallable(
      name = "java_source_jar",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getJavaSourceJar();

  @SkylarkCallable(
      name = "apk",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getApk();

  @SkylarkCallable(
      name = "aapt2_r_txt",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getAapt2RTxt();

  @SkylarkCallable(
      name = "aapt2_java_source_jar",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getAapt2SourceJar();

  @SkylarkCallable(
      name = "static_library",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  @Nullable
  FileT getStaticLibrary();

  @SkylarkCallable(
      name = "resources",
      structField = true,
      doc = "",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  SkylarkList<FileT> getResourcesList();
}
