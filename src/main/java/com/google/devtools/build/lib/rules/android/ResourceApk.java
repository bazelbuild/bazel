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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;

import javax.annotation.Nullable;

/**
 * The ResourceApk represents the packaged resources that serve as the basis for the signed and the
 * unsigned APKs.
 */
@Immutable
public final class ResourceApk {
  // TODO(bazel-team): The only fields that are legitimately nullable are javaSrcJar and
  // mainDexProguardConfig. The rest are marked as such due to .fromTransitiveResources().
  // It seems like there should be a better way to do this.
  @Nullable private final Artifact resourceApk;  // The .ap_ file
  @Nullable private final Artifact resourceJavaSrcJar;  // Source jar containing R.java and friends
  @Nullable private final Artifact resourceJavaClassJar;  // Class jar containing R.class files
  private final ResourceDependencies resourceDeps;
  @Nullable private final ResourceContainer primaryResource;
  @Nullable private final Artifact manifest;  // The non-binary XML version of AndroidManifest.xml
  @Nullable private final Artifact resourceProguardConfig;
  @Nullable private final Artifact mainDexProguardConfig;
  private final boolean legacy;

  public ResourceApk(
      @Nullable Artifact resourceApk,
      @Nullable Artifact resourceJavaSrcJar,
      @Nullable Artifact resourceJavaClassJar,
      ResourceDependencies resourceDeps,
      @Nullable ResourceContainer primaryResource,
      @Nullable Artifact manifest,
      @Nullable Artifact resourceProguardConfig,
      @Nullable Artifact mainDexProguardConfig,
      boolean legacy) {
    this.resourceApk = resourceApk;
    this.resourceJavaSrcJar = resourceJavaSrcJar;
    this.resourceJavaClassJar = resourceJavaClassJar;
    this.resourceDeps = resourceDeps;
    this.primaryResource = primaryResource;
    this.manifest = manifest;
    this.resourceProguardConfig = resourceProguardConfig;
    this.mainDexProguardConfig = mainDexProguardConfig;
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

  public Artifact getResourceJavaClassJar() {
    return resourceJavaClassJar;
  }

  public boolean isLegacy() {
    return legacy;
  }

  public static ResourceApk fromTransitiveResources(
      ResourceDependencies resourceDeps) {
    return new ResourceApk(null, null, null, resourceDeps, null, null, null, null, false);
  }

  public Artifact getResourceProguardConfig() {
    return resourceProguardConfig;
  }

  public Artifact getMainDexProguardConfig() {
    return mainDexProguardConfig;
  }

  public ResourceDependencies getResourceDependencies() {
    return resourceDeps;
  }

  /**
   * Creates an provider from the resources in the ResourceApk.
   * 
   * <p>If the ResourceApk was created from transitive resources, the provider will effectively
   * contain the "forwarded" resources: The merged transitive and merged direct dependencies of this
   * library.
   * 
   * <p>If the ResourceApk was generated from a "resources" attribute, it will contain the
   * "resources" container in the direct dependencies and the rest as transitive.
   * 
   * <p>If the ResourceApk was generated from local resources, that will be the direct dependencies and
   * the rest will be transitive.
   */
  public AndroidResourcesProvider toResourceProvider(Label label) {
    if (primaryResource == null) {
      return resourceDeps.toProvider(label);
    }
    return resourceDeps.toProvider(label, primaryResource);
  }
}
