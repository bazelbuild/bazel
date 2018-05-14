// Copyright 2015 The Bazel Authors. All rights reserved.
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
import javax.annotation.Nullable;

/** A provider for targets that produce an apk file. */
@SkylarkModule(
    name = "ApkInfo",
    doc = "APKs provided by a rule",
    category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ApkInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "ApkInfo";
  public static final NativeProvider<ApkInfo> PROVIDER =
      new NativeProvider<ApkInfo>(ApkInfo.class, SKYLARK_NAME) {};

  private final Artifact apk;
  private final Artifact unsignedApk;
  @Nullable
  private final Artifact coverageMetadata;
  private final Artifact mergedManifest;
  private final Artifact keystore;

  ApkInfo(
      Artifact apk,
      Artifact unsignedApk,
      @Nullable Artifact coverageMetadata,
      Artifact mergedManifest,
      Artifact keystore) {
    super(PROVIDER);
    this.apk = apk;
    this.unsignedApk = unsignedApk;
    this.coverageMetadata = coverageMetadata;
    this.mergedManifest = mergedManifest;
    this.keystore = keystore;
  }

  /** Returns the APK file built in the transitive closure. */
  @SkylarkCallable(
      name = "signed_apk",
      doc = "Returns a signed APK built from the target.",
      structField = true

  )
  public Artifact getApk() {
    return apk;
  }

  /** Returns the unsigned APK file built in the transitive closure. */
  public Artifact getUnsignedApk() {
    return unsignedApk;
  }

  /** Returns the coverage metadata artifacts generated in the transitive closure. */
  @Nullable
  public Artifact getCoverageMetadata() {
    return coverageMetadata;
  }

  /** Returns the merged manifest. */
  public Artifact getMergedManifest() {
    return mergedManifest;
  }

  /* The keystore that was used to sign the apk returned from {@see getApk() */
  public Artifact getKeystore() {
    return keystore;
  }
}
