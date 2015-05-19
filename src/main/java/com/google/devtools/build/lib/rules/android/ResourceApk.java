// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;

import javax.annotation.Nullable;

/**
 * The ResourceApk represents the packaged resources that serve as the basis for the signed and the
 * unsigned APKs.
 */
@Immutable
public class ResourceApk {
  // TODO(bazel-team): The only field that is legitimately nullable is javaSrcJar. The rest is
  // marked as such due to .fromTransitiveResources(). It seems like there should be a better way
  // to do this.
  @Nullable private final Artifact resourceApk;  // The .ap_ file
  @Nullable private final Artifact resourceJavaSrcJar;  // Source jar containing R.java and friends
  private final NestedSet<ResourceContainer> transitiveResources;
  @Nullable private final ResourceContainer primaryResource;
  @Nullable private final Artifact manifest;  // The non-binary XML version of AndroidManifest.xml
  @Nullable private final Artifact resourceProguardConfig;
  private final boolean legacy;

  public ResourceApk(
      @Nullable Artifact resourceApk,
      @Nullable Artifact resourceJavaSrcJar,
      NestedSet<ResourceContainer> transitiveResources,
      @Nullable ResourceContainer primaryResource,
      @Nullable Artifact manifest,
      @Nullable Artifact resourceProguardConfig,
      boolean legacy) {
    this.resourceApk = resourceApk;
    this.resourceJavaSrcJar = resourceJavaSrcJar;
    this.transitiveResources = transitiveResources;
    this.primaryResource = primaryResource;
    this.manifest = manifest;
    this.resourceProguardConfig = resourceProguardConfig;
    this.legacy = legacy;
  }

  public Artifact getArtifact() {
    return resourceApk;
  }

  public ResourceContainer getPrimaryResource() {
    return primaryResource;
  }

  public Artifact getManifest() {
    return manifest;
  }

  public Artifact getResourceJavaSrcJar() {
    return resourceJavaSrcJar;
  }

  public boolean isLegacy() {
    return legacy;
  }

  public NestedSet<ResourceContainer> getTransitiveResources() {
    return transitiveResources;
  }

  public static ResourceApk fromTransitiveResources(
      NestedSet<ResourceContainer> transitiveResources) {
    return new ResourceApk(null, null, transitiveResources, null, null, null, false);
  }

  public Artifact getResourceProguardConfig() {
    return resourceProguardConfig;
  }
}
