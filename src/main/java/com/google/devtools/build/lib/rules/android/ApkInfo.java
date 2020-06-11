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
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.ApkInfoApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import javax.annotation.Nullable;

/** A provider for targets that produce an apk file. */
@Immutable
public class ApkInfo extends NativeInfo implements ApkInfoApi<Artifact> {

  private static final String STARLARK_NAME = "ApkInfo";

  /**
   * Provider instance for {@link ApkInfo}.
   */
  public static final ApkInfoProvider PROVIDER = new ApkInfoProvider();

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

  @Override
  public Artifact getApk() {
    return apk;
  }

  /** Returns the unsigned APK file built in the transitive closure. */
  @Override
  public Artifact getUnsignedApk() {
    return unsignedApk;
  }

  /** Returns the coverage metadata artifact generated in the transitive closure. */
  @Nullable
  @Override
  public Artifact getCoverageMetadata() {
    return coverageMetadata;
  }

  /** Returns the merged manifest. */
  public Artifact getMergedManifest() {
    return mergedManifest;
  }

  /* The keystore that was used to sign the apk returned from {@see getApk() */
  @Override
  public Artifact getKeystore() {
    return keystore;
  }

  /** Provider for {@link ApkInfo}. */
  public static class ApkInfoProvider extends BuiltinProvider<ApkInfo>
      implements ApkInfoApiProvider {

    private ApkInfoProvider() {
      super(STARLARK_NAME, ApkInfo.class);
    }

    @Override
    public ApkInfoApi<?> createInfo(Dict<String, Object> kwargs) throws EvalException {
      return throwUnsupportedConstructorException();
    }
  }
}
