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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;

/** A provider of information about this target's manifest. */
@SkylarkModule(
    name = "AndroidManifestInfo",
    doc =
        "Information about the Android manifest provided by a rule. Note that, as of now, the"
            + " information exposed in this object is not directly consumed by Android rules -"
            + " instead, use an AndroidResourcesInfo.",
    category = SkylarkModuleCategory.PROVIDER)
public class AndroidManifestInfo extends NativeInfo {
  private static final String SKYLARK_NAME = "AndroidManifestInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /* numMandatoryPositionals = */ 2, // Manifest file and package
              /* numOptionalPositionals = */ 1, // exports_manifest
              /* numMandatoryNamedOnly = */ 0,
              /* starArg = */ false,
              /* kwArg = */ false,
              /* names = */ "manifest",
              "package",
              "exports_manifest"),
          /* defaultValues = */ ImmutableList.of(false), // is_dummy
          /* types = */ ImmutableList.of(
              SkylarkType.of(Artifact.class), // manifest
              SkylarkType.STRING, // package
              SkylarkType.BOOL)); // exports_manifest

  public static final NativeProvider<AndroidManifestInfo> PROVIDER =
      new NativeProvider<AndroidManifestInfo>(AndroidManifestInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        public AndroidManifestInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          // Skylark support code puts positional inputs in the correct order and validates types.
          return of((Artifact) args[0], (String) args[1], (boolean) args[2]);
        }
      };

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

  @SkylarkCallable(
      name = "manifest",
      doc = "This target's manifest, merged with manifests from dependencies",
      structField = true)
  public Artifact getManifest() {
    return manifest;
  }

  @SkylarkCallable(name = "package", doc = "This target's package", structField = true)
  public String getPackage() {
    return pkg;
  }

  @SkylarkCallable(
      name = "exports_manifest",
      doc = "If this manifest should be exported to targets that depend on it",
      structField = true)
  public boolean exportsManifest() {
    return exportsManifest;
  }

  public StampedAndroidManifest asStampedManifest() {
    return new StampedAndroidManifest(manifest, pkg, exportsManifest);
  }
}
