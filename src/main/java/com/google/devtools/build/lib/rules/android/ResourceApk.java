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
public class ResourceApk {
  // TODO(bazel-team): The only field that is legitimately nullable is javaSrcJar. The rest is
  // marked as such due to .fromTransitiveResources(). It seems like there should be a better way
  // to do this.
  @Nullable private final Artifact resourceApk;  // The .ap_ file
  @Nullable private final Artifact resourceJavaSrcJar;  // Source jar containing R.java and friends
  private final ResourceDependencies resourceDeps;
  @Nullable private final ResourceContainer primaryResource;
  @Nullable private final Artifact manifest;  // The non-binary XML version of AndroidManifest.xml
  @Nullable private final Artifact resourceProguardConfig;
  private final boolean legacy;

  public ResourceApk(
      @Nullable Artifact resourceApk,
      @Nullable Artifact resourceJavaSrcJar,
      ResourceDependencies resourceDeps,
      @Nullable ResourceContainer primaryResource,
      @Nullable Artifact manifest,
      @Nullable Artifact resourceProguardConfig,
      boolean legacy) {
    this.resourceApk = resourceApk;
    this.resourceJavaSrcJar = resourceJavaSrcJar;
    this.resourceDeps = resourceDeps;
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

  public static ResourceApk fromTransitiveResources(
      ResourceDependencies resourceDeps) {
    return new ResourceApk(null, null, resourceDeps, null, null, null, false);
  }

  public Artifact getResourceProguardConfig() {
    return resourceProguardConfig;
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
