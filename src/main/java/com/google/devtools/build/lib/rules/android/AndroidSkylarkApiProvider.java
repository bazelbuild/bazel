// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A class that exposes the Android providers to Skylark. It is intended to provide a
 * simple and stable interface for Skylark users.
 */
@SkylarkModule(
  name = "AndroidSkylarkApiProvider",
  doc = "Provides access to information about Android rules"
)
public class AndroidSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "android";

  @SkylarkCallable(
    name = "apk",
    structField = true,
    allowReturnNones = true,
    doc = "Returns an APK produced by this target."
  )
  public Artifact getApk() {
    return getInfo().getProvider(AndroidIdeInfoProvider.class).getSignedApk();
  }

  @SkylarkCallable(
    name = "java_package",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a java package for this target."
  )
  public String getJavaPackage() {
    return getInfo().getProvider(AndroidIdeInfoProvider.class).getJavaPackage();
  }

  @SkylarkCallable(
    name = "manifest",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a manifest file for this target."
  )
  public Artifact getManifest() {
    return getInfo().getProvider(AndroidIdeInfoProvider.class).getManifest();
  }
}
