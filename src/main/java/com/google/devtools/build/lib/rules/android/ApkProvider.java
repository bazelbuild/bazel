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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

/** A provider for targets that produce an apk file. */
@AutoValue
@Immutable
public abstract class ApkProvider implements TransitiveInfoProvider {

  public static ApkProvider create(
      Artifact apk,
      Artifact unsignedApk,
      @Nullable Artifact coverageMetadata,
      Artifact mergedManifest,
      Artifact keystore) {
    return new AutoValue_ApkProvider(apk, unsignedApk, coverageMetadata, mergedManifest, keystore);
  }

  /** Returns the APK file built in the transitive closure. */
  public abstract Artifact getApk();

  /** Returns the unsigned APK file built in the transitive closure. */
  public abstract Artifact getUnsignedApk();

  /** Returns the coverage metadata artifacts generated in the transitive closure. */
  @Nullable
  public abstract Artifact getCoverageMetadata();

  /** Returns the merged manifest. */
  public abstract Artifact getMergedManifest();

  /* The keystore that was used to sign the apk returned from {@see getApk() */
  public abstract Artifact getKeystore();

  ApkProvider() {}
}
