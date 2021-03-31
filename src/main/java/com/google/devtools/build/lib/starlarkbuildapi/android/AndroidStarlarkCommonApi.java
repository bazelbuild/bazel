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

import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Common utilities for Starlark rules related to Android. */
@StarlarkBuiltin(
    name = "android_common",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Common utilities and functionality related to Android rules.",
    documented = false)
public interface AndroidStarlarkCommonApi<
        FileT extends FileApi, JavaInfoT extends JavaInfoApi<?, ?>>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "create_device_broker_info",
      documented = false,
      parameters = {@Param(name = "type")})
  AndroidDeviceBrokerInfoApi createDeviceBrokerInfo(String deviceBrokerType);

  @StarlarkMethod(
      name = "resource_source_directory",
      allowReturnNones = true,
      doc =
          "Returns a source directory for Android resource file. "
              + "The source directory is a prefix of resource's relative path up to "
              + "a directory that designates resource kind (cf. "
              + "http://developer.android.com/guide/topics/resources/providing-resources.html).",
      documented = false,
      parameters = {
        @Param(
            name = "resource",
            doc = "The android resource file.",
            positional = true,
            named = false)
      })
  @Nullable
  String getSourceDirectoryRelativePathFromResource(FileT resource);

  @StarlarkMethod(
      name = "multi_cpu_configuration",
      doc =
          "A configuration for rule attributes that compiles native code according to "
              + "the --fat_apk_cpu and --android_crosstool_top flags.",
      documented = false,
      structField = true)
  AndroidSplitTransititionApi getAndroidSplitTransition();

  @StarlarkMethod(
      name = "enable_implicit_sourceless_deps_exports_compatibility",
      doc = "Takes a JavaInfo and converts it to an implicit exportable JavaInfo.",
      documented = false,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
      parameters = {
        @Param(
            name = "dep",
            doc =
                "A JavaInfo that will be used as an implicit export for sourceless deps exports"
                    + " compatibility.",
            positional = true,
            named = false)
      })
  JavaInfoT enableImplicitSourcelessDepsExportsCompatibility(JavaInfoT javaInfo);
}
