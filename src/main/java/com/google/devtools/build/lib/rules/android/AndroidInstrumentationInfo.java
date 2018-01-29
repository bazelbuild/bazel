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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A provider for targets that create Android instrumentations. Consumed by Android testing rules.
 */
 @SkylarkModule(
    name = "AndroidInstrumentationInfo",
    doc = "Android instrumentation and target APKs to run in a test",
    category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class AndroidInstrumentationInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "AndroidInstrumentationInfo";
  public static final NativeProvider<AndroidInstrumentationInfo> PROVIDER =
      new NativeProvider<AndroidInstrumentationInfo>(
          AndroidInstrumentationInfo.class, SKYLARK_NAME) {};

  private final Artifact targetApk;
  private final Artifact instrumentationApk;

  AndroidInstrumentationInfo(Artifact targetApk, Artifact instrumentationApk) {
    super(PROVIDER);
    this.targetApk = targetApk;
    this.instrumentationApk = instrumentationApk;
  }

  @SkylarkCallable(
      name = "target_apk",
      doc = "Returns the target APK of the instrumentation test.",
      structField = true
  )
  public Artifact getTargetApk() {
    return targetApk;
  }

  @SkylarkCallable(
      name = "instrumentation_apk",
      doc = "Returns the instrumentation APK that should be executed.",
      structField = true
  )
  public Artifact getInstrumentationApk() {
    return instrumentationApk;
  }
}
