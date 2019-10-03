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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidManifestInfoApi;

/** A provider of information about this target's manifest. */
public class AndroidManifestInfo extends NativeInfo implements AndroidManifestInfoApi<Artifact> {

  /** Provider singleton constant. */
  public static final BuiltinProvider<AndroidManifestInfo> PROVIDER = new Provider();

  /** Provider for {@link AndroidManifestInfo} objects. */
  private static class Provider extends BuiltinProvider<AndroidManifestInfo>
      implements AndroidManifestInfoApi.Provider<Artifact> {
    private Provider() {
      super(NAME, AndroidManifestInfo.class);
    }

    @Override
    public AndroidManifestInfo androidManifestInfo(
        Artifact manifest, String packageString, Boolean exportsManifest) {
      return of(manifest, packageString, exportsManifest);
    }
  }

  private final Artifact manifest;
  private final String pkg;
  private final boolean exportsManifest;

  static AndroidManifestInfo of(Artifact manifest, String pkg, boolean exportsManifest) {
    return new AndroidManifestInfo(manifest, pkg, exportsManifest);
  }

  private AndroidManifestInfo(Artifact manifest, String pkg, boolean exportsManifest) {
    super(PROVIDER);
    this.manifest = manifest;
    this.pkg = pkg;
    this.exportsManifest = exportsManifest;
  }

  @Override
  public Artifact getManifest() {
    return manifest;
  }

  @Override
  public String getPackage() {
    return pkg;
  }

  @Override
  public boolean exportsManifest() {
    return exportsManifest;
  }

  public StampedAndroidManifest asStampedManifest() {
    return new StampedAndroidManifest(manifest, pkg, exportsManifest);
  }
}
