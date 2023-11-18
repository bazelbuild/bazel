// Copyright 2023 The Bazel Authors. All rights reserved.
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

/** Provider of proguard + resource shrinking outputs, used as part of the Starlark migration. */
@StarlarkBuiltin(
    name = "AndroidOptimizationInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidOptimizationInfoApi<FileT extends FileApi> extends StructApi {
  /** The name of the provider for this info object. */
  String NAME = "AndroidOptimizationInfo";

  @StarlarkMethod(
      name = "optimized_jar",
      structField = true,
      doc = "Returns the optimized jar.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getOptimizedJar();

  @StarlarkMethod(
      name = "mapping",
      structField = true,
      doc = "Returns the proguard mapping.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getMapping();

  @StarlarkMethod(
      name = "seeds",
      structField = true,
      doc = "Returns the proguard seeds.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getSeeds();

  @StarlarkMethod(
      name = "library_jar",
      structField = true,
      doc = "Returns the proguard library jar.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getLibraryJar();

  @StarlarkMethod(
      name = "config",
      structField = true,
      doc = "Returns the proguard config.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getConfig();

  @StarlarkMethod(
      name = "usage",
      structField = true,
      doc = "Returns the proguard usage.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getUsage();

  @StarlarkMethod(
      name = "proto_mapping",
      structField = true,
      doc = "Returns the proguard proto mapping.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getProtoMapping();

  @StarlarkMethod(
      name = "rewritten_startup_profile",
      structField = true,
      doc = "Returns the rewritten startup profile.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getRewrittenStartupProfile();

  @StarlarkMethod(
      name = "rewritten_merged_baseline_profile",
      structField = true,
      doc = "Returns the rewritten merged baseline profile.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getRewrittenMergedBaselineProfile();

  @StarlarkMethod(
      name = "optimized_resource_apk",
      structField = true,
      doc = "Returns the optimized resource apk.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getOptimizedResourceApk();

  @StarlarkMethod(
      name = "shrunk_resource_apk",
      structField = true,
      doc = "Returns the shrunk resource apk.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getShrunkResourceApk();

  @StarlarkMethod(
      name = "shrunk_resource_zip",
      structField = true,
      doc = "Returns the shrunk resource zip.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getShrunkResourceZip();

  @StarlarkMethod(
      name = "resource_shrinker_log",
      structField = true,
      doc = "Returns the resource shrinker log.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getResourceShrinkerLog();

  @StarlarkMethod(
      name = "resource_optimization_config",
      structField = true,
      doc = "Returns the resource optimization config.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getResourceOptimizationConfig();

  @StarlarkMethod(
      name = "resource_path_shortening_map",
      structField = true,
      doc = "Returns the resource path shortening map.",
      allowReturnNones = true,
      documented = false)
  @Nullable
  FileT getResourcePathShorteningMap();

  /** The provider implementing this can construct the AndroidOptimizationInfoApi provider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>AndroidOptimizationInfoApi</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "optimized_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The optimized jar.",
              defaultValue = "None"),
          @Param(
              name = "mapping",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard mapping.",
              defaultValue = "None"),
          @Param(
              name = "seeds",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard seeds.",
              defaultValue = "None"),
          @Param(
              name = "library_jar",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard library jar.",
              defaultValue = "None"),
          @Param(
              name = "config",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard config.",
              defaultValue = "None"),
          @Param(
              name = "usage",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard usage.",
              defaultValue = "None"),
          @Param(
              name = "proto_mapping",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The proguard proto mapping.",
              defaultValue = "None"),
          @Param(
              name = "rewritten_startup_profile",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The rewritten startup profile.",
              defaultValue = "None"),
          @Param(
              name = "rewriten_merged_baseline_profile",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The rewritten merged baseline profile.",
              defaultValue = "None"),
          @Param(
              name = "optimized_resource_apk",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The optimized resource apk.",
              defaultValue = "None"),
          @Param(
              name = "shrunk_resource_apk",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The shrunk resource apk.",
              defaultValue = "None"),
          @Param(
              name = "shrunk_resource_zip",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The shrunk resource zip.",
              defaultValue = "None"),
          @Param(
              name = "resource_shrinker_log",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The resource shrinker log.",
              defaultValue = "None"),
          @Param(
              name = "resource_optimization_config",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The resource optimization config.",
              defaultValue = "None"),
          @Param(
              name = "resource_path_shortening_map",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              doc = "The resource path shortening map.",
              defaultValue = "None"),
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidOptimizationInfoApi<FileT> createInfo(
        Object optimizedJar,
        Object mapping,
        Object seeds,
        Object libraryJar,
        Object config,
        Object usage,
        Object protoMapping,
        Object rewrittenStartupProfile,
        Object rewrittenMergedBaselineProfile,
        Object optimizedResourceApk,
        Object shrunkResourceApk,
        Object shrunkResourceZip,
        Object resourceShrinkerLog,
        Object resourceOptimizationConfig,
        Object resourcePathShorteningMap)
        throws EvalException;
  }
}
