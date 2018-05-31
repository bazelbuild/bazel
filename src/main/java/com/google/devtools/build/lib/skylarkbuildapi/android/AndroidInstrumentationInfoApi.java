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
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A provider for targets that create Android instrumentations. Consumed by Android testing rules.
 */
@SkylarkModule(
    name = "AndroidInstrumentationInfo",
    doc = "Android instrumentation and target APKs to run in a test",
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidInstrumentationInfoApi<FileT extends FileApi> extends StructApi {

  @SkylarkCallable(
      name = "target_apk",
      doc = "Returns the target APK of the instrumentation test.",
      structField = true)
  FileT getTargetApk();

  @SkylarkCallable(
      name = "instrumentation_apk",
      doc = "Returns the instrumentation APK that should be executed.",
      structField = true)
  FileT getInstrumentationApk();
}
