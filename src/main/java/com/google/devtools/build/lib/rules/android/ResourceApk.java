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
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
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
  @Nullable private final Artifact resourceApk; // The .ap_ file
  @Nullable private final Artifact resourceJavaSrcJar; // Source jar containing R.java and friends
  @Nullable private final Artifact resourceJavaClassJar; // Class jar containing R.class files
  private final ResourceDependencies resourceDeps;
  private final AssetDependencies assetDeps;
  @Nullable private final ResourceContainer primaryResources;
  @Nullable private final AndroidAssets primaryAssets;

  @Nullable private final Artifact manifest; // The non-binary XML version of AndroidManifest.xml
  @Nullable private final Artifact resourceProguardConfig;
  @Nullable private final Artifact mainDexProguardConfig;

  static ResourceApk of(ResourceContainer resourceContainer, ResourceDependencies resourceDeps) {
    return of(resourceContainer, resourceDeps, null, null);
  }

  static ResourceApk of(
      ResourceContainer resourceContainer,
      ResourceDependencies resourceDeps,
      @Nullable Artifact resourceProguardConfig,
      @Nullable Artifact mainDexProguardConfig) {
    return new ResourceApk(
        resourceContainer.getApk(),
        resourceContainer.getJavaSourceJar(),
        resourceContainer.getJavaClassJar(),
        resourceDeps,
        AssetDependencies.empty(),
        resourceContainer,
        resourceContainer.getAndroidAssets(),
        resourceContainer.getManifest(),
        resourceProguardConfig,
        mainDexProguardConfig);
  }

  private ResourceApk(
      @Nullable Artifact resourceApk,
      @Nullable Artifact resourceJavaSrcJar,
      @Nullable Artifact resourceJavaClassJar,
      ResourceDependencies resourceDeps,
      AssetDependencies assetDeps,
      @Nullable ResourceContainer primaryResources,
      @Nullable AndroidAssets primaryAssets,
      @Nullable Artifact manifest,
      @Nullable Artifact resourceProguardConfig,
      @Nullable Artifact mainDexProguardConfig) {
    this.resourceApk = resourceApk;
    this.resourceJavaSrcJar = resourceJavaSrcJar;
    this.resourceJavaClassJar = resourceJavaClassJar;
    this.resourceDeps = resourceDeps;
    this.assetDeps = assetDeps;
    this.primaryResources = primaryResources;
    this.primaryAssets = primaryAssets;
    this.manifest = manifest;
    this.resourceProguardConfig = resourceProguardConfig;
    this.mainDexProguardConfig = mainDexProguardConfig;
  }

  ResourceApk withApk(Artifact apk) {
    return new ResourceApk(
        apk,
        resourceJavaSrcJar,
        resourceJavaClassJar,
        resourceDeps,
        assetDeps,
        primaryResources,
        primaryAssets,
        manifest,
        resourceProguardConfig,
        mainDexProguardConfig);
  }

  public Artifact getArtifact() {
    return resourceApk;
  }

  public ResourceContainer getPrimaryResources() {
    return primaryResources;
  }

  /**
   * TODO(b/77574966): Use MergedAndroidAssets rather than the base class once we have completely
   * decoupled assets and resources.
   */
  public AndroidAssets getPrimaryAssets() {
    return primaryAssets;
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

  public static ResourceApk fromTransitiveResources(
      ResourceDependencies resourceDeps, AssetDependencies assetDeps) {
    return new ResourceApk(null, null, null, resourceDeps, assetDeps, null, null, null, null, null);
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

  public AssetDependencies getAssetDependencies() {
    return assetDeps;
  }

  /**
   * Creates an provider from the resources in the ResourceApk.
   *
   * <p>If the ResourceApk was created from transitive resources, the provider will effectively
   * contain the "forwarded" resources: The merged transitive and merged direct dependencies of this
   * library.
   *
   * <p>If the ResourceApk was generated from local resources, that will be the direct dependencies
   * and the rest will be transitive.
   */
  private AndroidResourcesInfo toResourceInfo(Label label) {
    if (primaryResources == null) {
      return resourceDeps.toInfo(label);
    }
    return resourceDeps.toInfo(primaryResources);
  }

  public void addToConfiguredTargetBuilder(RuleConfiguredTargetBuilder builder, Label label) {
    AndroidResourcesInfo resourceInfo = toResourceInfo(label);
    builder.addNativeDeclaredProvider(resourceInfo);

    // TODO(b/77574966): Remove this cast once we get rid of ResourceContainer and can guarantee
    // that only properly merged resources are passed into this object.
    if (primaryAssets instanceof MergedAndroidAssets) {
      MergedAndroidAssets merged = (MergedAndroidAssets) primaryAssets;
      builder.addNativeDeclaredProvider(merged.toProvider());

      // Asset merging output isn't consumed by anything. Require it to be run by top-level targets
      // so we can validate there are no asset merging conflicts.
      builder.addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, merged.getMergedAssets());

    } else if (primaryAssets == null) {
      builder.addNativeDeclaredProvider(assetDeps.toInfo(label));
    }

    builder.addSkylarkTransitiveInfo(
        AndroidSkylarkApiProvider.NAME, new AndroidSkylarkApiProvider(resourceInfo));
  }
}
