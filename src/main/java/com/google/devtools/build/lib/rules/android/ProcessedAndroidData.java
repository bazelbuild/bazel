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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.java.ProguardHelper;
import com.google.devtools.build.lib.syntax.Type;
import java.util.List;
import java.util.Map;
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

  /** Processes Android data (assets, resources, and manifest) for android_binary targets. */
  public static ProcessedAndroidData processBinaryDataFrom(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      boolean conditionalKeepRules,
      Map<String, String> manifestValues,
      AndroidAaptVersion aaptVersion,
      AndroidResources resources,
      AndroidAssets assets,
      ResourceDependencies resourceDeps,
      AssetDependencies assetDeps,
      ResourceFilterFactory resourceFilterFactory,
      List<String> noCompressExtensions,
      boolean crunchPng,
      boolean dataBindingEnabled)
      throws RuleErrorException, InterruptedException {
    if (conditionalKeepRules && aaptVersion != AndroidAaptVersion.AAPT2) {
      throw ruleContext.throwWithRuleError(
          "resource cycle shrinking can only be enabled for builds with aapt2");
    }

    AndroidResourcesProcessorBuilder builder =
        builderForNonIncrementalTopLevelTarget(ruleContext, manifest, manifestValues, aaptVersion)
            .setUseCompiledResourcesForMerge(
                aaptVersion == AndroidAaptVersion.AAPT2
                    && AndroidCommon.getAndroidConfig(ruleContext).skipParsingAction())
            .setManifestOut(
                ruleContext.getImplicitOutputArtifact(
                    AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST))
            .setMergedResourcesOut(
                ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP))
            .setMainDexProguardOut(AndroidBinary.createMainDexProguardSpec(ruleContext))
            .conditionalKeepRules(conditionalKeepRules)
            .setDataBindingInfoZip(
                dataBindingEnabled ? DataBinding.getLayoutInfoFile(ruleContext) : null)
            .setFeatureOf(
                ruleContext.attributes().isAttributeValueExplicitlySpecified("feature_of")
                    ? ruleContext
                        .getPrerequisite("feature_of", Mode.TARGET, ApkInfo.PROVIDER)
                        .getApk()
                    : null)
            .setFeatureAfter(
                ruleContext.attributes().isAttributeValueExplicitlySpecified("feature_after")
                    ? ruleContext
                        .getPrerequisite("feature_after", Mode.TARGET, ApkInfo.PROVIDER)
                        .getApk()
                    : null);
    return buildActionForBinary(
        ruleContext,
        builder,
        manifest,
        resources,
        assets,
        resourceDeps,
        assetDeps,
        resourceFilterFactory,
        noCompressExtensions,
        crunchPng);
  }

  public static ProcessedAndroidData processIncrementalBinaryDataFrom(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      Artifact apkOut,
      String proguardPrefix,
      Map<String, String> manifestValues)
      throws RuleErrorException {

    AndroidResourcesProcessorBuilder builder =
        builderForTopLevelTarget(ruleContext, manifest, proguardPrefix, manifestValues)
            .setApkOut(apkOut);

    return buildActionForBinary(
        ruleContext,
        builder,
        manifest,
        AndroidResources.from(ruleContext, "resource_files"),
        AndroidAssets.from(ruleContext),
        ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
        AssetDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
        ResourceFilterFactory.fromRuleContextAndAttrs(ruleContext),
        ruleContext.getExpander().withDataLocations().tokenized("nocompress_extensions"),
        ruleContext.attributes().get("crunch_png", Type.BOOLEAN));
  }

  private static ProcessedAndroidData buildActionForBinary(
      RuleErrorConsumer errorConsumer,
      AndroidResourcesProcessorBuilder builder,
      StampedAndroidManifest manifest,
      AndroidResources resources,
      AndroidAssets assets,
      ResourceDependencies resourceDeps,
      AssetDependencies assetDeps,
      ResourceFilterFactory resourceFilterFactory,
      List<String> noCompressExtensions,
      boolean crunchPng)
      throws RuleErrorException {

    ResourceFilter resourceFilter =
        resourceFilterFactory.getResourceFilter(errorConsumer, resourceDeps, resources);

    // Filter unwanted resources out
    resources = resources.filterLocalResources(errorConsumer, resourceFilter);
    resourceDeps = resourceDeps.filter(errorConsumer, resourceFilter);

    return builder
        .setResourceFilterFactory(resourceFilterFactory)
        .setUncompressedExtensions(noCompressExtensions)
        .setCrunchPng(crunchPng)
        .withResourceDependencies(resourceDeps)
        .withAssetDependencies(assetDeps)
        .build(resources, assets, manifest);
  }

  /** Processes Android data (assets, resources, and manifest) for android_local_test targets. */
  public static ProcessedAndroidData processLocalTestDataFrom(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      Map<String, String> manifestValues,
      AndroidAaptVersion aaptVersion,
      AndroidResources resources,
      AndroidAssets assets,
      ResourceDependencies resourceDeps,
      AssetDependencies assetDeps)
      throws InterruptedException {

    return builderForNonIncrementalTopLevelTarget(
            ruleContext, manifest, manifestValues, aaptVersion)
        .setUseCompiledResourcesForMerge(
            aaptVersion == AndroidAaptVersion.AAPT2
                && AndroidCommon.getAndroidConfig(ruleContext).skipParsingAction())
        .setManifestOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST))
        .setMergedResourcesOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP))
        .setCrunchPng(false)
        .withResourceDependencies(resourceDeps)
        .withAssetDependencies(assetDeps)
        .build(resources, assets, manifest);
  }

  /** Processes Android data (assets, resources, and manifest) for android_test targets. */
  public static ProcessedAndroidData processTestDataFrom(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      String packageUnderTest,
      boolean hasLocalResourceFiles,
      AndroidAaptVersion aaptVersion,
      AndroidResources resources,
      ResourceDependencies resourceDeps,
      AndroidAssets assets,
      AssetDependencies assetDeps)
      throws InterruptedException {

    AndroidResourcesProcessorBuilder builder =
        builderForNonIncrementalTopLevelTarget(
                ruleContext, manifest, ImmutableMap.of(), aaptVersion)
            .setMainDexProguardOut(AndroidBinary.createMainDexProguardSpec(ruleContext))
            .setPackageUnderTest(packageUnderTest)
            .setIsTestWithResources(hasLocalResourceFiles)
            .withResourceDependencies(resourceDeps)
            .withAssetDependencies(assetDeps);

    return builder.build(resources, assets, manifest);
  }

  /**
   * Common {@link AndroidResourcesProcessorBuilder} builder for non-incremental top-level targets.
   *
   * <p>The builder will be populated with commonly-used settings and outputs.
   */
  private static AndroidResourcesProcessorBuilder builderForNonIncrementalTopLevelTarget(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      Map<String, String> manifestValues,
      AndroidAaptVersion aaptVersion)
      throws InterruptedException {

    return builderForTopLevelTarget(ruleContext, manifest, "", manifestValues)
        .targetAaptVersion(aaptVersion)

        // Outputs
        .setApkOut(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK))
        .setRTxtOut(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT))
        .setSourceJarOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR))
        .setSymbols(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_MERGED_SYMBOLS));
  }

  /**
   * Common {@link AndroidResourcesProcessorBuilder} builder for top-level targets.
   *
   * <p>The builder will be populated with commonly-used settings and outputs.
   */
  private static AndroidResourcesProcessorBuilder builderForTopLevelTarget(
      RuleContext ruleContext,
      StampedAndroidManifest manifest,
      String proguardPrefix,
      Map<String, String> manifestValues) {
    return new AndroidResourcesProcessorBuilder(ruleContext)
        // Settings
        .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
        .setJavaPackage(manifest.getPackage())
        .setApplicationId(manifestValues.get("applicationId"))
        .setVersionCode(manifestValues.get("versionCode"))
        .setVersionName(manifestValues.get("versionName"))
        .setThrowOnResourceConflict(
            AndroidCommon.getAndroidConfig(ruleContext).throwOnResourceConflict())

        // Output
        .setProguardOut(ProguardHelper.getProguardConfigArtifact(ruleContext, proguardPrefix));
  }

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
        resources, assets, manifest, rTxt, sourceJar, apk, dataBindingInfoZip, resourceDeps,
        resourceProguardConfig, mainDexProguardConfig);
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
  public ResourceApk generateRClass(RuleContext ruleContext, AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    return new RClassGeneratorActionBuilder(ruleContext)
        .targetAaptVersion(aaptVersion)
        .withDependencies(resourceDeps)
        .setClassJarOut(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
        .build(this);
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
            resources, mergedResources, rClassJar, dataBindingInfoZip, resourceDeps, manifest);

    // Combined resource processing does not produce aapt2 artifacts; they're nulled out
    ValidatedAndroidResources validated =
        ValidatedAndroidResources.of(merged, rTxt, sourceJar, apk, null, null, null);
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
