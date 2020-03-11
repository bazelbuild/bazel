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
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Common utilities for Skylark rules related to Android. */
@SkylarkModule(
    name = "android_common",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Common utilities and functionality related to Android rules.",
    documented = false)
public interface AndroidSkylarkCommonApi<FileT extends FileApi, JavaInfoT extends JavaInfoApi<?>>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "create_device_broker_info",
      documented = false,
      parameters = {@Param(name = "type", type = String.class)})
  AndroidDeviceBrokerInfoApi createDeviceBrokerInfo(String deviceBrokerType);

  @SkylarkCallable(
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
            named = false,
            type = FileApi.class)
      })
  String getSourceDirectoryRelativePathFromResource(FileT resource);

  @SkylarkCallable(
      name = "multi_cpu_configuration",
      doc =
          "A configuration for rule attributes that compiles native code according to "
              + "the --fat_apk_cpu and --android_crosstool_top flags.",
      documented = false,
      structField = true)
  AndroidSplitTransititionApi getAndroidSplitTransition();

  @SkylarkCallable(
      name = "enable_implicit_sourceless_deps_exports_compatibility",
      doc = "Takes a JavaInfo and converts it to an implicit exportable JavaInfo.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
      parameters = {
        @Param(
            name = "dep",
            doc =
                "A JavaInfo that will be used as an implicit export for sourceless deps exports"
                    + " compatibility.",
            positional = true,
            named = false,
            type = JavaInfoApi.class)
      })
  JavaInfoT enableImplicitSourcelessDepsExportsCompatibility(JavaInfoT javaInfo);
}
