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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.android.AndroidDataConverter.JoinerType;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Collections;
import java.util.List;

/** Builder for creating resource processing action. */
public class AndroidResourcesProcessorBuilder {

  @AutoCodec @VisibleForSerialization
  static final AndroidDataConverter<ValidatedAndroidResources> AAPT2_RESOURCE_DEP_TO_ARG_NO_PARSE =
      AndroidDataConverter.<ValidatedAndroidResources>builder(JoinerType.COLON_COMMA)
          .withRoots(ValidatedAndroidResources::getResourceRoots)
          .withEmpty()
          .withArtifact(ValidatedAndroidResources::getManifest)
          .maybeWithArtifact(ValidatedAndroidResources::getAapt2RTxt)
          .maybeWithArtifact(ValidatedAndroidResources::getCompiledSymbols)
          .build();

  private ResourceDependencies resourceDependencies = ResourceDependencies.empty();
  private AssetDependencies assetDependencies = AssetDependencies.empty();

  private Artifact proguardOut;
  private Artifact mainDexProguardOut;
  private boolean conditionalKeepRules;
  private Artifact rTxtOut;
  private Artifact sourceJarOut;
  private boolean debug = false;
  private ResourceFilterFactory resourceFilterFactory = ResourceFilterFactory.empty();
  private List<String> uncompressedExtensions = Collections.emptyList();
  private Artifact apkOut;
  private String customJavaPackage;
  private String versionCode;
  private String applicationId;
  private String versionName;
  private Artifact symbols;
  private Artifact dataBindingInfoZip;

  private Artifact manifestOut;
  private Artifact mergedResourcesOut;
  private boolean isLibrary;
  private boolean crunchPng = true;
  private boolean throwOnResourceConflict;
  private String packageUnderTest;
  private boolean isTestWithResources = false;

  /**
   * The output zip for resource-processed data binding expressions (i.e. a zip of .xml files).
   *
   * <p>If null, data binding processing is skipped (and data binding expressions aren't allowed in
   * layout resources).
   */
  public AndroidResourcesProcessorBuilder setDataBindingInfoZip(Artifact zip) {
    this.dataBindingInfoZip = zip;
    return this;
  }

  public AndroidResourcesProcessorBuilder withResourceDependencies(
      ResourceDependencies resourceDeps) {
    this.resourceDependencies = resourceDeps;
    return this;
  }

  public AndroidResourcesProcessorBuilder withAssetDependencies(AssetDependencies assetDeps) {
    this.assetDependencies = assetDeps;
    return this;
  }

  public AndroidResourcesProcessorBuilder setUncompressedExtensions(
      List<String> uncompressedExtensions) {
    this.uncompressedExtensions = uncompressedExtensions;
    return this;
  }

  public AndroidResourcesProcessorBuilder setCrunchPng(boolean crunchPng) {
    this.crunchPng = crunchPng;
    return this;
  }

  public AndroidResourcesProcessorBuilder setResourceFilterFactory(
      ResourceFilterFactory resourceFilterFactory) {
    this.resourceFilterFactory = resourceFilterFactory;
    return this;
  }

  public AndroidResourcesProcessorBuilder setDebug(boolean debug) {
    this.debug = debug;
    return this;
  }

  public AndroidResourcesProcessorBuilder setProguardOut(Artifact proguardCfg) {
    this.proguardOut = proguardCfg;
    return this;
  }

  public AndroidResourcesProcessorBuilder conditionalKeepRules(boolean conditionalKeepRules) {
    this.conditionalKeepRules = conditionalKeepRules;
    return this;
  }

  public AndroidResourcesProcessorBuilder setMainDexProguardOut(Artifact mainDexProguardCfg) {
    this.mainDexProguardOut = mainDexProguardCfg;
    return this;
  }

  public AndroidResourcesProcessorBuilder setRTxtOut(Artifact rTxtOut) {
    this.rTxtOut = rTxtOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setSymbols(Artifact symbols) {
    this.symbols = symbols;
    return this;
  }

  public AndroidResourcesProcessorBuilder setApkOut(Artifact apkOut) {
    this.apkOut = apkOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setSourceJarOut(Artifact sourceJarOut) {
    this.sourceJarOut = sourceJarOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setManifestOut(Artifact manifestOut) {
    this.manifestOut = manifestOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setMergedResourcesOut(Artifact mergedResourcesOut) {
    this.mergedResourcesOut = mergedResourcesOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setLibrary(boolean isLibrary) {
    this.isLibrary = isLibrary;
    return this;
  }

  public AndroidResourcesProcessorBuilder setThrowOnResourceConflict(
      boolean throwOnResourceConflict) {
    this.throwOnResourceConflict = throwOnResourceConflict;
    return this;
  }

  /**
   * Creates and registers an action that processes only transitive data.
   *
   * <p>Local resources and assets will be completely ignored by this action.
   *
   * @return a {@link ResourceApk} containing the processed resource, asset, and manifest
   *     information.
   */
  public ResourceApk buildWithoutLocalResources(
      AndroidDataContext dataContext,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext) {

    build(
        dataContext, AndroidResources.empty(), AndroidAssets.empty(), manifest, dataBindingContext);

    return ResourceApk.fromTransitiveResources(
        resourceDependencies,
        assetDependencies,
        manifest.withProcessedManifest(manifestOut == null ? manifest.getManifest() : manifestOut),
        rTxtOut,
        dataBindingContext);
  }

  public ProcessedAndroidData build(
      AndroidDataContext dataContext,
      AndroidResources primaryResources,
      AndroidAssets primaryAssets,
      StampedAndroidManifest primaryManifest,
      DataBindingContext dataBindingContext) {

    // Wrap the new manifest, if any
    ProcessedAndroidManifest processedManifest =
        new ProcessedAndroidManifest(
            manifestOut == null ? primaryManifest.getManifest() : manifestOut,
            primaryManifest.getPackage(),
            primaryManifest.isExported());

    // In databinding v2, this strips out the databinding and generates the layout info file.
    AndroidResources databindingProcessedResources = dataBindingContext.processResources(
        dataContext, primaryResources, processedManifest.getPackage());

    createAapt2ApkAction(
        dataContext, databindingProcessedResources, primaryAssets, primaryManifest);

    // Wrap the parsed resources
    ParsedAndroidResources parsedResources =
        ParsedAndroidResources.of(
            databindingProcessedResources,
            symbols,
            /* compiledSymbols = */ null,
            dataContext.getLabel(),
            processedManifest,
            dataBindingContext);

    // Wrap the parsed and merged assets
    ParsedAndroidAssets parsedAssets =
        ParsedAndroidAssets.of(
            primaryAssets, symbols, /* compiledSymbols = */ null, dataContext.getLabel());
    MergedAndroidAssets mergedAssets =
        MergedAndroidAssets.of(parsedAssets, mergedResourcesOut, assetDependencies);

    return ProcessedAndroidData.of(
        parsedResources,
        mergedAssets,
        processedManifest,
        rTxtOut,
        sourceJarOut,
        apkOut,
        dataBindingInfoZip,
        resourceDependencies,
        proguardOut,
        mainDexProguardOut);
  }

  public AndroidResourcesProcessorBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public AndroidResourcesProcessorBuilder setVersionCode(String versionCode) {
    this.versionCode = versionCode;
    return this;
  }

  public AndroidResourcesProcessorBuilder setApplicationId(String applicationId) {
    if (applicationId != null && !applicationId.isEmpty()) {
      this.applicationId = applicationId;
    }
    return this;
  }

  public AndroidResourcesProcessorBuilder setVersionName(String versionName) {
    this.versionName = versionName;
    return this;
  }

  public AndroidResourcesProcessorBuilder setPackageUnderTest(String packageUnderTest) {
    this.packageUnderTest = packageUnderTest;
    return this;
  }

  public AndroidResourcesProcessorBuilder setIsTestWithResources(boolean isTestWithResources) {
    this.isTestWithResources = isTestWithResources;
    return this;
  }

  private void createAapt2ApkAction(
      AndroidDataContext dataContext,
      AndroidResources primaryResources,
      AndroidAssets primaryAssets,
      StampedAndroidManifest primaryManifest) {
    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "AAPT2_PACKAGE").addAapt();

    if (resourceDependencies != null) {
      builder
          .addTransitiveFlag(
              "--data",
              resourceDependencies.getTransitiveResourceContainers(),
              AAPT2_RESOURCE_DEP_TO_ARG_NO_PARSE)
          .addTransitiveFlag(
              "--directData",
              resourceDependencies.getDirectResourceContainers(),
              AAPT2_RESOURCE_DEP_TO_ARG_NO_PARSE)
          .addTransitiveInputValues(resourceDependencies.getTransitiveResources())
          .addTransitiveInputValues(resourceDependencies.getTransitiveManifests())
          .addTransitiveInputValues(resourceDependencies.getTransitiveAapt2RTxt())
          .addTransitiveInputValues(resourceDependencies.getTransitiveCompiledSymbols());
    }

    if (assetDependencies != null && !assetDependencies.getTransitiveAssets().isEmpty()) {
      builder
          .addTransitiveFlag(
              "--directAssets",
              assetDependencies.getDirectParsedAssets(),
              AndroidDataConverter.COMPILED_ASSET_CONVERTER)
          .addTransitiveFlag(
              "--assets",
              assetDependencies.getTransitiveParsedAssets(),
              AndroidDataConverter.COMPILED_ASSET_CONVERTER)
          .addTransitiveInputValues(assetDependencies.getTransitiveAssets())
          .addTransitiveInputValues(assetDependencies.getTransitiveCompiledSymbols());
    }

    builder.maybeAddFlag("--conditionalKeepRules", conditionalKeepRules);

    configureCommonFlags(dataContext, primaryResources, primaryAssets, primaryManifest, builder)
        .buildAndRegister("Processing Android resources", "AndroidAapt2");
  }

  private BusyBoxActionBuilder configureCommonFlags(
      AndroidDataContext dataContext,
      AndroidResources primaryResources,
      AndroidAssets primaryAssets,
      StampedAndroidManifest primaryManifest,
      BusyBoxActionBuilder builder) {

    return builder
        .addInput(
            "--primaryData",
            String.format(
                "%s:%s:%s",
                AndroidDataConverter.rootsToString(primaryResources.getResourceRoots()),
                AndroidDataConverter.rootsToString(primaryAssets.getAssetRoots()),
                primaryManifest.getManifest().getExecPathString()),
            Iterables.concat(
                primaryResources.getResources(),
                primaryAssets.getAssets(),
                ImmutableList.of(primaryManifest.getManifest())))
        .maybeAddFlag("--buildToolsVersion", dataContext.getSdk().getBuildToolsVersion())
        .addAndroidJar()
        .maybeAddFlag("--packageType", isLibrary)
        .maybeAddFlag("LIBRARY", isLibrary)
        .maybeAddOutput("--rOutput", rTxtOut)
        .maybeAddOutput("--symbolsOut", symbols)
        .maybeAddOutput("--srcJarOutput", sourceJarOut)
        .maybeAddOutput("--proguardOutput", proguardOut)
        .maybeAddOutput("--mainDexProguardOutput", mainDexProguardOut)
        .maybeAddOutput("--manifestOutput", manifestOut)
        .maybeAddOutput("--resourcesOutput", mergedResourcesOut)
        .maybeAddOutput("--packagePath", apkOut)

        // Always pass density and resource configuration filter strings to execution, even when
        // filtering in analysis. Filtering in analysis cannot remove resources from Filesets, and,
        // in addition, aapt needs access to resource filters to generate pseudolocalized resources
        // and because its resource filtering is somewhat stricter for locales, and resource
        // processing needs access to densities to add them to the manifest.
        .maybeAddFlag("--resourceConfigs", resourceFilterFactory.getConfigurationFilterString())
        .maybeAddFlag("--densities", resourceFilterFactory.getDensityString())
        .maybeAddVectoredFlag("--uncompressedExtensions", uncompressedExtensions)
        .maybeAddFlag("--useAaptCruncher=no", !crunchPng)
        .maybeAddFlag("--debug", debug)
        .maybeAddFlag("--versionCode", versionCode)
        .maybeAddFlag("--versionName", versionName)
        .maybeAddFlag("--applicationId", applicationId)
        .maybeAddOutput("--dataBindingInfoOut", dataBindingInfoZip)
        .maybeAddFlag("--packageForR", customJavaPackage)
        .maybeAddFlag("--throwOnResourceConflict", throwOnResourceConflict)
        .maybeAddFlag("--packageUnderTest", packageUnderTest)
        .maybeAddFlag("--isTestWithResources", isTestWithResources);
  }
}
