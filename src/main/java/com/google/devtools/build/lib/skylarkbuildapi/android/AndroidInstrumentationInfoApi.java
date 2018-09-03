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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * A provider for targets that create Android instrumentations. Consumed by Android testing rules.
 */
@SkylarkModule(
    name = "AndroidInstrumentationInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Android instrumentation and target APKs to run in a test",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidInstrumentationInfoApi<FileT extends FileApi> extends StructApi {

  /**
   * Name of this info object.
   */
  public static String NAME = "AndroidInstrumentationInfo";

  @SkylarkCallable(
      name = "target_apk",
      doc = "Returns the target APK of the instrumentation test.",
      documented = false,
      structField = true)
  FileT getTargetApk();

  @SkylarkCallable(
      name = "instrumentation_apk",
      doc = "Returns the instrumentation APK that should be executed.",
      documented = false,
      structField = true)
  FileT getInstrumentationApk();

  /** Provider for {@link AndroidInstrumentationInfoApi}. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  public interface AndroidInstrumentationInfoApiProvider<FileT extends FileApi>
      extends ProviderApi {

    @SkylarkCallable(
        name = "AndroidInstrumentationInfo",
        doc = "The <code>AndroidInstrumentationInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "target_apk",
              type = FileApi.class,
              named = true,
              doc = "The target APK of the instrumentation test."),
          @Param(
              name = "instrumentation_apk",
              type = FileApi.class,
              named = true,
              doc = "The instrumentation APK that should be executed.")
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidInstrumentationInfoApi.class, receiverNameForDoc = NAME)
    public AndroidInstrumentationInfoApi<FileT> createInfo(
        FileT targetApk, FileT instrumentationApk) throws EvalException;
  }
}
