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
import javax.annotation.Nullable;

/**
 * Processed Android data (assets, resources, and manifest) returned from resource processing.
 *
 * <p>In libraries, data is parsed, merged, and then validated. For top-level targets like
 * android_binary, however, most of these steps happen in a single monolithic action. The only thing
 * the monolithic action doesn't do is generate an R.class file for the resources. When combined
 * with such a file, this object should contain the same data as the results of the individual
 * actions.
 *
 * <p>In general, the individual actions should be used, as they are decoupled and designed to allow
 * parallelized processing of a dependency tree of android_library targets. The monolithic action
 * should only be used as part of building the data into a final APK that can become part of a
 * produce android_binary or other top-level APK.
 */
public class ProcessedAndroidData {
  private final ParsedAndroidResources resources;
  private final MergedAndroidAssets assets;
  private final ProcessedAndroidManifest manifest;
  private final Artifact rTxt;
  private final Artifact sourceJar;
  private final Artifact apk;
  @Nullable private final Artifact dataBindingInfoZip;
  private final ResourceDependencies resourceDeps;
  private final Artifact resourceProguardConfig;
  @Nullable private final Artifact mainDexProguardConfig;

  static ProcessedAndroidData of(
      ParsedAndroidResources resources,
      MergedAndroidAssets assets,
      ProcessedAndroidManifest manifest,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDeps,
      Artifact resourceProguardConfig,
      @Nullable Artifact mainDexProguardConfig) {
    return new ProcessedAndroidData(
        resources,
        assets,
        manifest,
        rTxt,
        sourceJar,
        apk,
        dataBindingInfoZip,
        resourceDeps,
        resourceProguardConfig,
        mainDexProguardConfig);
  }

  private ProcessedAndroidData(
      ParsedAndroidResources resources,
      MergedAndroidAssets assets,
      ProcessedAndroidManifest manifest,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      @Nullable Artifact dataBindingInfoZip,
      ResourceDependencies resourceDeps,
      Artifact resourceProguardConfig,
      @Nullable Artifact mainDexProguardConfig) {
    this.resources = resources;
    this.assets = assets;
    this.manifest = manifest;
    this.rTxt = rTxt;
    this.sourceJar = sourceJar;
    this.apk = apk;
    this.dataBindingInfoZip = dataBindingInfoZip;
    this.resourceDeps = resourceDeps;
    this.resourceProguardConfig = resourceProguardConfig;
    this.mainDexProguardConfig = mainDexProguardConfig;
  }

  /**
   * Gets the fully processed data from this class.
   *
   * <p>Registers an action to run R class generation, the last step needed in resource processing.
   * Returns the fully processed data, including validated resources, wrapped in a ResourceApk.
   */
  public ResourceApk generateRClass(AndroidDataContext dataContext) throws InterruptedException {
    return new RClassGeneratorActionBuilder()
        .withDependencies(resourceDeps)
        .setClassJarOut(
            dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
        .build(dataContext, this);
  }

  /**
   * Returns fully processed resources. The R class generator action will not be registered.
   *
   * @param rClassJar an artifact containing the resource class jar for these resources. An action
   *     to generate it must be registered elsewhere.
   */
  ResourceApk withValidatedResources(Artifact rClassJar) {
    // When assets and resources are processed together, they are both merged into the same zip
    Artifact mergedResources = assets.getMergedAssets();

    // Since parts of both merging and validation were already done in combined resource processing,
    // we need to build containers for both here.
    MergedAndroidResources merged =
        MergedAndroidResources.of(
            resources,
            mergedResources,
            rClassJar,
            /*aapt2RTxt=*/ null,
            dataBindingInfoZip,
            resourceDeps,
            manifest);

    // Combined resource processing does not produce aapt2 artifacts; they're nulled out
    ValidatedAndroidResources validated =
        ValidatedAndroidResources.of(
            merged,
            rTxt,
            sourceJar,
            apk,
            /*aapt2ValidationArtifact=*/ (Artifact) null,
            /*aapt2SourceJar*/ (Artifact) null,
            /*staticLibrary*/ (Artifact) null,
            /*useRTxtFromMergedResources=*/ true);
    return ResourceApk.of(validated, assets, resourceProguardConfig, mainDexProguardConfig);
  }

  public MergedAndroidAssets getAssets() {
    return assets;
  }

  public ProcessedAndroidManifest getManifest() {
    return manifest;
  }

  public Artifact getRTxt() {
    return rTxt;
  }
}
