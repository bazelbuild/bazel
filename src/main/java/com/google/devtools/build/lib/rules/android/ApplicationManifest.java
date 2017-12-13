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

import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidManifestMerger;
import com.google.devtools.build.lib.rules.android.ResourceContainer.Builder.JavaPackageSource;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {

  public static ApplicationManifest fromResourcesRule(RuleContext ruleContext)
      throws RuleErrorException {
    final AndroidResourcesProvider resources = AndroidCommon.getAndroidResources(ruleContext);
    if (resources == null) {
      ruleContext.attributeError("manifest", "a resources or manifest attribute is mandatory.");
      return null;
    }
    return fromExplicitManifest(
        ruleContext, Iterables.getOnlyElement(resources.getDirectAndroidResources()).getManifest());
  }

  public ApplicationManifest createSplitManifest(
      RuleContext ruleContext, String splitName, boolean hasCode) {
    // aapt insists that manifests be called AndroidManifest.xml, even though they have to be
    // explicitly designated as manifests on the command line
    Artifact result =
        AndroidBinary.getDxArtifact(ruleContext, "split_" + splitName + "/AndroidManifest.xml");
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$build_split_manifest", Mode.HOST))
            .setProgressMessage("Creating manifest for split %s", splitName)
            .setMnemonic("AndroidBuildSplitManifest")
            .addInput(manifest)
            .addOutput(result);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPath("--main_manifest", manifest)
            .addExecPath("--split_manifest", result)
            .add("--split", splitName);
    if (hasCode) {
      commandLine.add("--hascode");
    } else {
      commandLine.add("--nohascode");
    }

    String overridePackage = manifestValues.get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));
    return new ApplicationManifest(ruleContext, result, targetAaptVersion);
  }

  public ApplicationManifest addMobileInstallStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_MANIFEST);
    Artifact stubData =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_DATA);

    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
            .setProgressMessage("Injecting mobile install stub application")
            .setMnemonic("InjectMobileInstallStubApplication")
            .addInput(manifest)
            .addOutput(stubManifest)
            .addOutput(stubData);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .add("--mode=mobile_install")
            .addExecPath("--input_manifest", manifest)
            .addExecPath("--output_manifest", stubManifest)
            .addExecPath("--output_datafile", stubData);

    String overridePackage = manifestValues.get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));

    return new ApplicationManifest(ruleContext, stubManifest, targetAaptVersion);
  }

  public ApplicationManifest addInstantRunStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.INSTANT_RUN_STUB_APPLICATION_MANIFEST);

    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
            .setProgressMessage("Injecting instant run stub application")
            .setMnemonic("InjectInstantRunStubApplication")
            .addInput(manifest)
            .addOutput(stubManifest)
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("--mode=instant_run")
                    .addExecPath("--input_manifest", manifest)
                    .addExecPath("--output_manifest", stubManifest)
                    .build());

    ruleContext.registerAction(builder.build(ruleContext));

    return new ApplicationManifest(ruleContext, stubManifest, targetAaptVersion);
  }

  public static ApplicationManifest fromRule(RuleContext ruleContext) throws RuleErrorException {
    return fromExplicitManifest(
        ruleContext, ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET));
  }

  public static ApplicationManifest fromExplicitManifest(RuleContext ruleContext, Artifact manifest)
      throws RuleErrorException {
    return new ApplicationManifest(
        ruleContext, manifest, AndroidAaptVersion.chooseTargetAaptVersion(ruleContext));
  }

  /**
   * Generates an empty manifest for a rule that does not directly specify resources.
   *
   * <p><strong>Note:</strong> This generated manifest can then be used as the primary manifest when
   * merging with dependencies.
   *
   * @return the generated ApplicationManifest
   */
  public static ApplicationManifest generatedManifest(RuleContext ruleContext)
      throws RuleErrorException {
    Artifact generatedManifest =
        ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_generated",
            PathFragment.create("AndroidManifest.xml"),
            ruleContext.getBinOrGenfilesDirectory());

    String manifestPackage = AndroidCommon.getJavaPackage(ruleContext);
    String contents =
        Joiner.on("\n")
            .join(
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"",
                "          package=\"" + manifestPackage + "\">",
                "   <application>",
                "   </application>",
                "</manifest>");
    ruleContext
        .getAnalysisEnvironment()
        .registerAction(
            FileWriteAction.create(
                ruleContext, generatedManifest, contents, /*makeExecutable=*/ false));
    return fromExplicitManifest(ruleContext, generatedManifest);
  }

  private static ImmutableMap<String, String> getManifestValues(RuleContext context) {
    Map<String, String> manifestValues = new TreeMap<>();
    // applicationId is set from manifest_values or android_resources.rename_manifest_package
    // with descending priority.
    AndroidResourcesProvider resourcesProvider = AndroidCommon.getAndroidResources(context);
    if (resourcesProvider != null) {
      ResourceContainer resourceContainer =
          Iterables.getOnlyElement(resourcesProvider.getDirectAndroidResources());
      if (resourceContainer.getRenameManifestPackage() != null) {
        manifestValues.put("applicationId", resourceContainer.getRenameManifestPackage());
      }
    }
    if (context.attributes().isAttributeValueExplicitlySpecified("manifest_values")) {
      manifestValues.putAll(context.attributes().get("manifest_values", Type.STRING_DICT));
    }

    for (String variable : manifestValues.keySet()) {
      manifestValues.put(
          variable, context.getExpander().expand("manifest_values", manifestValues.get(variable)));
    }
    return ImmutableMap.copyOf(manifestValues);
  }

  private final Artifact manifest;
  private final ImmutableMap<String, String> manifestValues;
  private final AndroidAaptVersion targetAaptVersion;

  private ApplicationManifest(
      RuleContext ruleContext, Artifact manifest, AndroidAaptVersion targetAaptVersion) {
    this.manifest = manifest;
    this.manifestValues = getManifestValues(ruleContext);
    this.targetAaptVersion = targetAaptVersion;
  }

  public ApplicationManifest mergeWith(RuleContext ruleContext, ResourceDependencies resourceDeps) {
    boolean legacy = useLegacyMerging(ruleContext);
    return mergeWith(ruleContext, resourceDeps, legacy);
  }

  public ApplicationManifest mergeWith(
      RuleContext ruleContext, ResourceDependencies resourceDeps, boolean legacy) {
    Map<Artifact, Label> mergeeManifests =
        getMergeeManifests(resourceDeps.getResourceContainers());

    if (legacy) {
      if (!mergeeManifests.isEmpty()) {
        Artifact outputManifest =
            ruleContext.getUniqueDirectoryArtifact(
                ruleContext.getRule().getName() + "_merged",
                "AndroidManifest.xml",
                ruleContext.getBinOrGenfilesDirectory());
        AndroidManifestMergeHelper.createMergeManifestAction(
            ruleContext,
            getManifest(),
            mergeeManifests.keySet(),
            ImmutableList.of("all"),
            outputManifest);
        return new ApplicationManifest(ruleContext, outputManifest, targetAaptVersion);
      }
    } else {
      if (!mergeeManifests.isEmpty() || !manifestValues.isEmpty()) {
        Artifact outputManifest =
            ruleContext.getUniqueDirectoryArtifact(
                ruleContext.getRule().getName() + "_merged",
                "AndroidManifest.xml",
                ruleContext.getBinOrGenfilesDirectory());
        Artifact mergeLog =
            ruleContext.getUniqueDirectoryArtifact(
                ruleContext.getRule().getName() + "_merged",
                "manifest_merger_log.txt",
                ruleContext.getBinOrGenfilesDirectory());
        new ManifestMergerActionBuilder(ruleContext)
            .setManifest(getManifest())
            .setMergeeManifests(mergeeManifests)
            .setLibrary(false)
            .setManifestValues(manifestValues)
            .setCustomPackage(AndroidCommon.getJavaPackage(ruleContext))
            .setManifestOutput(outputManifest)
            .setLogOut(mergeLog)
            .build(ruleContext);
        return new ApplicationManifest(ruleContext, outputManifest, targetAaptVersion);
      }
    }
    return this;
  }

  private boolean useLegacyMerging(RuleContext ruleContext) {
    boolean legacy = true;
    if (ruleContext.isLegalFragment(AndroidConfiguration.class)
        && ruleContext.getRule().isAttrDefined("manifest_merger", STRING)) {
      AndroidManifestMerger merger =
          AndroidManifestMerger.fromString(ruleContext.attributes().get("manifest_merger", STRING));
      if (merger == null) {
        merger = ruleContext.getFragment(AndroidConfiguration.class).getManifestMerger();
      }
      if (merger == AndroidManifestMerger.LEGACY) {
        ruleContext.ruleWarning(
            "manifest_merger 'legacy' is deprecated. Please update to 'android'.\n"
                + "See https://developer.android.com/studio/build/manifest-merge.html for more "
                + "information about the manifest merger.");
      }
      legacy = merger == AndroidManifestMerger.LEGACY;
    }
    return legacy;
  }

  private static Map<Artifact, Label> getMergeeManifests(
      Iterable<ResourceContainer> resourceContainers) {
    ImmutableSortedMap.Builder<Artifact, Label> builder =
        ImmutableSortedMap.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
    for (ResourceContainer r : resourceContainers) {
      if (r.isManifestExported()) {
        builder.put(r.getManifest(), r.getLabel());
      }
    }
    return builder.build();
  }

  public ApplicationManifest renamePackage(RuleContext ruleContext, String customPackage) {
    if (isNullOrEmpty(customPackage)) {
      return this;
    }
    Artifact outputManifest =
        ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_renamed",
            "AndroidManifest.xml",
            ruleContext.getBinOrGenfilesDirectory());
    new ManifestMergerActionBuilder(ruleContext)
        .setManifest(getManifest())
        .setLibrary(true)
        .setCustomPackage(customPackage)
        .setManifestOutput(outputManifest)
        .build(ruleContext);
    return new ApplicationManifest(ruleContext, outputManifest, targetAaptVersion);
  }

  public ResourceApk packTestWithDataAndResources(
      RuleContext ruleContext,
      Artifact resourceApk,
      ResourceDependencies resourceDeps,
      @Nullable Artifact rTxt,
      boolean incremental,
      Artifact proguardCfg,
      @Nullable String packageUnderTest,
      boolean hasLocalResourceFiles)
      throws InterruptedException, RuleErrorException {
    LocalResourceContainer data =
        LocalResourceContainer.forAssetsAndResources(
            ruleContext, "assets", AndroidCommon.getAssetDir(ruleContext), "local_resource_files");

    ResourceContainer resourceContainer =
        checkForInlinedResources(
            ResourceContainer.builderFromRule(ruleContext)
                .setAssetsAndResourcesFrom(data)
                .setManifest(getManifest())
                .setApk(resourceApk)
                .setRTxt(rTxt)
                .build(),
            resourceDeps
                .getResourceContainers(), // TODO(bazel-team): Figure out if we really need to check
            // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule
            // with resources would check for inline resources, we can rely on the previous rule to
            // have checked its dependencies.
            ruleContext);

    AndroidResourcesProcessorBuilder builder =
        new AndroidResourcesProcessorBuilder(ruleContext)
            .setLibrary(false)
            .setApkOut(resourceContainer.getApk())
            .setUncompressedExtensions(ImmutableList.of())
            .setCrunchPng(true)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setProguardOut(proguardCfg)
            .setApplicationId(manifestValues.get("applicationId"))
            .setVersionCode(manifestValues.get("versionCode"))
            .setVersionName(manifestValues.get("versionName"))
            .setThrowOnResourceConflict(
                ruleContext
                    .getConfiguration()
                    .getFragment(AndroidConfiguration.class)
                    .throwOnResourceConflict())
            .setPackageUnderTest(packageUnderTest)
            .setIsTestWithResources(hasLocalResourceFiles);
    if (!incremental) {
      builder
          .targetAaptVersion(targetAaptVersion)
          .setRTxtOut(resourceContainer.getRTxt())
          .setSymbols(resourceContainer.getSymbols())
          .setSourceJarOut(resourceContainer.getJavaSourceJar());
    }
    ResourceContainer processed = builder.build(ruleContext);

    return new ResourceApk(
        resourceContainer.getApk(),
        processed.getJavaSourceJar(),
        processed.getJavaClassJar(),
        resourceDeps,
        processed,
        processed.getManifest(),
        proguardCfg,
        null,
        false);
  }

  /** Packages up the manifest with resource and assets from the LocalResourceContainer. */
  public ResourceApk packAarWithDataAndResources(
      RuleContext ruleContext,
      LocalResourceContainer data,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      Artifact manifestOut,
      Artifact mergedResources)
      throws InterruptedException, RuleErrorException {
    // Filter the resources during analysis to prevent processing of dependencies on unwanted
    // resources during execution.
    ResourceFilter resourceFilter =
        ResourceFilterFactory.fromRuleContext(ruleContext)
            .getResourceFilter(ruleContext, resourceDeps, data);
    data = data.filter(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(resourceFilter);

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        checkForInlinedResources(
            ResourceContainer.builderFromRule(ruleContext)
                .setRTxt(rTxt)
                .setSymbols(symbols)
                .setJavaPackageFrom(JavaPackageSource.MANIFEST)
                .setManifestExported(true)
                .setManifest(getManifest())
                .setAssetsAndResourcesFrom(data)
                .build(),
            resourceDeps
                .getResourceContainers(), // TODO(bazel-team): Figure out if we really need to check
            // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule
            // with resources would check for inline resources, we can rely on the previous rule to
            // have checked its dependencies.
            ruleContext);

    // android_library should only build the APK one way (!incremental).
    Artifact rJavaClassJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    if (resourceContainer.getSymbols() != null) {
      AndroidResourceParsingActionBuilder parsingBuilder =
          new AndroidResourceParsingActionBuilder(ruleContext)
              .withPrimary(resourceContainer)
              .setParse(data)
              .setOutput(resourceContainer.getSymbols())
              .setCompiledSymbolsOutput(resourceContainer.getCompiledSymbols());

      resourceContainer = parsingBuilder.build(ruleContext);
    }

    ResourceContainer merged =
        new AndroidResourceMergingActionBuilder(ruleContext)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setMergedResourcesOut(mergedResources)
            .setManifestOut(manifestOut)
            .setClassJarOut(rJavaClassJar)
            .setThrowOnResourceConflict(
                ruleContext
                    .getConfiguration()
                    .getFragment(AndroidConfiguration.class)
                    .throwOnResourceConflict())
            .build(ruleContext);

    ResourceContainer processed =
        new AndroidResourceValidatorActionBuilder(ruleContext)
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setMergedResources(mergedResources)
            .withPrimary(merged)
            .setRTxtOut(merged.getRTxt())
            .setSourceJarOut(merged.getJavaSourceJar())
            .setApkOut(resourceContainer.getApk())
            // aapt2 related artifacts. Will be generated if the targetAaptVersion is AAPT2.
            .withDependencies(resourceDeps)
            .setCompiledSymbols(merged.getCompiledSymbols())
            .setAapt2RTxtOut(merged.getAapt2RTxt())
            .setAapt2SourceJarOut(merged.getAapt2JavaSourceJar())
            .setStaticLibraryOut(merged.getStaticLibrary())
            .build(ruleContext);

    return new ResourceApk(
        resourceContainer.getApk(),
        processed.getJavaSourceJar(),
        processed.getJavaClassJar(),
        resourceDeps,
        processed,
        processed.getManifest(),
        null,
        null,
        false);
  }

  /* Creates an incremental apk from assets and data. */
  public ResourceApk packIncrementalBinaryWithDataAndResources(
      RuleContext ruleContext,
      Artifact resourceApk,
      ResourceDependencies resourceDeps,
      List<String> uncompressedExtensions,
      boolean crunchPng,
      Artifact proguardCfg)
      throws InterruptedException, RuleErrorException {
    LocalResourceContainer data =
        LocalResourceContainer.forAssetsAndResources(
            ruleContext, "assets", AndroidCommon.getAssetDir(ruleContext), "resource_files");

    // Filter the resources during analysis to prevent processing of dependencies on unwanted
    // resources during execution.
    ResourceFilterFactory resourceFilterFactory =
        ResourceFilterFactory.fromRuleContext(ruleContext);
    ResourceFilter resourceFilter =
        resourceFilterFactory.getResourceFilter(ruleContext, resourceDeps, data);
    data = data.filter(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(resourceFilter);

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        checkForInlinedResources(
            ResourceContainer.builderFromRule(ruleContext)
                .setApk(resourceApk)
                .setManifest(getManifest())
                .setAssetsAndResourcesFrom(data)
                .build(),
            resourceDeps
                .getResourceContainers(), // TODO(bazel-team): Figure out if we really need to check
            // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule
            // with resources would check for inline resources, we can rely on the previous rule to
            // have checked its dependencies.
            ruleContext);

    ResourceContainer processed =
        new AndroidResourcesProcessorBuilder(ruleContext)
            .setLibrary(false)
            .setApkOut(resourceContainer.getApk())
            .setResourceFilterFactory(resourceFilterFactory)
            .setUncompressedExtensions(uncompressedExtensions)
            .setCrunchPng(crunchPng)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setProguardOut(proguardCfg)
            .setApplicationId(manifestValues.get("applicationId"))
            .setVersionCode(manifestValues.get("versionCode"))
            .setVersionName(manifestValues.get("versionName"))
            .setThrowOnResourceConflict(
                ruleContext
                    .getConfiguration()
                    .getFragment(AndroidConfiguration.class)
                    .throwOnResourceConflict())
            .setPackageUnderTest(null)
            .build(ruleContext);

    return new ResourceApk(
        resourceContainer.getApk(),
        processed.getJavaSourceJar(),
        processed.getJavaClassJar(),
        resourceDeps,
        processed,
        processed.getManifest(),
        proguardCfg,
        null,
        false);
  }

  /** Packages up the manifest with resource and assets from the rule and dependent resources. */
  // TODO(bazel-team): this method calls for some refactoring, 15+ params including some nullables.
  public ResourceApk packBinaryWithDataAndResources(
      RuleContext ruleContext,
      Artifact resourceApk,
      ResourceDependencies resourceDeps,
      @Nullable Artifact rTxt,
      ResourceFilterFactory resourceFilterFactory,
      List<String> uncompressedExtensions,
      boolean crunchPng,
      Artifact proguardCfg,
      @Nullable Artifact mainDexProguardCfg,
      boolean conditionalKeepRules,
      Artifact manifestOut,
      Artifact mergedResources,
      @Nullable Artifact dataBindingInfoZip,
      @Nullable Artifact featureOf,
      @Nullable Artifact featureAfter)
      throws InterruptedException, RuleErrorException {
    LocalResourceContainer data =
        LocalResourceContainer.forAssetsAndResources(
            ruleContext, "assets", AndroidCommon.getAssetDir(ruleContext), "resource_files");

    ResourceFilter resourceFilter =
        resourceFilterFactory.getResourceFilter(ruleContext, resourceDeps, data);
    data = data.filter(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(resourceFilter);

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        checkForInlinedResources(
            ResourceContainer.builderFromRule(ruleContext)
                .setAssetsAndResourcesFrom(data)
                .setManifest(getManifest())
                .setRTxt(rTxt)
                .setApk(resourceApk)
                .build(),
            resourceDeps
                .getResourceContainers(), // TODO(bazel-team): Figure out if we really need to check
            // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule
            // with resources would check for inline resources, we can rely on the previous rule to
            // have checked its dependencies.
            ruleContext);

    AndroidConfiguration androidConfiguration = ruleContext.getConfiguration()
        .getFragment(AndroidConfiguration.class);

    boolean skipParsingAction =
        targetAaptVersion == AndroidAaptVersion.AAPT2 && androidConfiguration.skipParsingAction();

    if (conditionalKeepRules && targetAaptVersion != AndroidAaptVersion.AAPT2) {
      throw ruleContext.throwWithRuleError(
          "resource cycle shrinking can only be enabled for builds with aapt2");
    }

    ResourceContainer processed =
        new AndroidResourcesProcessorBuilder(ruleContext)
            .setLibrary(false)
            .setApkOut(resourceContainer.getApk())
            .setResourceFilterFactory(resourceFilterFactory)
            .setUncompressedExtensions(uncompressedExtensions)
            .setCrunchPng(crunchPng)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setManifestOut(manifestOut)
            .setMergedResourcesOut(mergedResources)
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setProguardOut(proguardCfg)
            .setMainDexProguardOut(mainDexProguardCfg)
            .conditionalKeepRules(conditionalKeepRules)
            .setDataBindingInfoZip(dataBindingInfoZip)
            .setApplicationId(manifestValues.get("applicationId"))
            .setVersionCode(manifestValues.get("versionCode"))
            .setVersionName(manifestValues.get("versionName"))
            .setFeatureOf(featureOf)
            .setFeatureAfter(featureAfter)
            .setThrowOnResourceConflict(androidConfiguration.throwOnResourceConflict())
            .setUseCompiledResourcesForMerge(skipParsingAction)
            .targetAaptVersion(targetAaptVersion)
            .setRTxtOut(resourceContainer.getRTxt())
            .setSymbols(resourceContainer.getSymbols())
            .setSourceJarOut(resourceContainer.getJavaSourceJar())
            .build(ruleContext);

    return new ResourceApk(
        resourceContainer.getApk(),
        processed.getJavaSourceJar(),
        processed.getJavaClassJar(),
        resourceDeps,
        processed,
        processed.getManifest(),
        proguardCfg,
        mainDexProguardCfg,
        false);
  }

  public ResourceApk packLibraryWithDataAndResources(
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      Artifact manifestOut,
      Artifact mergedResources,
      Artifact dataBindingInfoZip)
      throws InterruptedException, RuleErrorException {
    // Filter the resources during analysis to prevent processing of dependencies on unwanted
    // resources during execution.
    LocalResourceContainer data =
        LocalResourceContainer.forAssetsAndResources(
            ruleContext, "assets", AndroidCommon.getAssetDir(ruleContext), "resource_files");
    ResourceFilter resourceFilter =
        ResourceFilterFactory.fromRuleContext(ruleContext)
            .getResourceFilter(ruleContext, resourceDeps, data);
    data = data.filter(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(resourceFilter);

    ResourceContainer.Builder builder =
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setSymbols(symbols)
            .setRTxt(rTxt)
            // Request an APK so it can be inherited when a library is used in a binary's
            // resources attr.
            // TODO(b/30307842): Remove this once it is no longer needed for resources migration.
            .setApk(ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_APK));

    if (targetAaptVersion == AndroidAaptVersion.AAPT2) {
      builder
          .setAapt2JavaSourceJar(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_SOURCE_JAR))
          .setAapt2RTxt(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_R_TXT))
          .setCompiledSymbols(
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_COMPILED_SYMBOLS))
          .setStaticLibrary(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_LIBRARY_APK));
    }

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        checkForInlinedResources(
            builder.setManifest(getManifest()).setAssetsAndResourcesFrom(data).build(),
            resourceDeps
                .getResourceContainers(), // TODO(bazel-team): Figure out if we really need to check
            // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule
            // with resources would check for inline resources, we can rely on the previous rule to
            // have checked its dependencies.
            ruleContext);

    Artifact rJavaClassJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    AndroidConfiguration androidConfiguration = ruleContext.getConfiguration()
        .getFragment(AndroidConfiguration.class);

    boolean skipParsingAction =
        targetAaptVersion == AndroidAaptVersion.AAPT2 && androidConfiguration.skipParsingAction();

    if (resourceContainer.getSymbols() != null) {
      AndroidResourceParsingActionBuilder parsingBuilder =
          new AndroidResourceParsingActionBuilder(ruleContext)
              .withPrimary(resourceContainer)
              .setParse(data)
              .setOutput(resourceContainer.getSymbols())
              .setCompiledSymbolsOutput(resourceContainer.getCompiledSymbols());

      if (dataBindingInfoZip != null && resourceContainer.getCompiledSymbols() != null) {
        PathFragment unusedInfo = dataBindingInfoZip.getRootRelativePath();
        // TODO(corysmith): Centralize the data binding processing and zipping into a single
        // action. Data binding processing needs to be triggered here as well as the merger to
        // avoid aapt2 from throwing an error during compilation.
        parsingBuilder.setDataBindingInfoZip(
            ruleContext.getDerivedArtifact(
                unusedInfo.replaceName(unusedInfo.getBaseName() + "_unused.zip"),
                dataBindingInfoZip.getRoot()));
      }
      resourceContainer = parsingBuilder.build(ruleContext);
    }

    ResourceContainer merged =
        new AndroidResourceMergingActionBuilder(ruleContext)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setThrowOnResourceConflict(androidConfiguration.throwOnResourceConflict())
            .setUseCompiledMerge(skipParsingAction)
            .setDataBindingInfoZip(dataBindingInfoZip)
            .setMergedResourcesOut(mergedResources)
            .setManifestOut(manifestOut)
            .setClassJarOut(rJavaClassJar)
            .setDataBindingInfoZip(dataBindingInfoZip)
            .build(ruleContext);

    ResourceContainer processed =
        new AndroidResourceValidatorActionBuilder(ruleContext)
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setMergedResources(mergedResources)
            .withPrimary(merged)
            .setRTxtOut(merged.getRTxt())
            .setSourceJarOut(merged.getJavaSourceJar())
            .setApkOut(resourceContainer.getApk())
            // aapt2 related artifacts. Will be generated if the targetAaptVersion is AAPT2.
            .withDependencies(resourceDeps)
            .setCompiledSymbols(merged.getCompiledSymbols())
            .setAapt2RTxtOut(merged.getAapt2RTxt())
            .setAapt2SourceJarOut(merged.getAapt2JavaSourceJar())
            .setStaticLibraryOut(merged.getStaticLibrary())
            .build(ruleContext);

    return new ResourceApk(
        resourceContainer.getApk(),
        processed.getJavaSourceJar(),
        processed.getJavaClassJar(),
        resourceDeps,
        processed,
        processed.getManifest(),
        null,
        null,
        false);
  }

  private static ResourceContainer checkForInlinedResources(
      ResourceContainer resourceContainer,
      Iterable<ResourceContainer> resourceContainers,
      RuleContext ruleContext)
      throws RuleErrorException {
    // Dealing with Android library projects
    if (Iterables.size(resourceContainers) > 1) {
      if (resourceContainer.getConstantsInlined()
          && !resourceContainer.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
        ruleContext.ruleError(
            "This android binary depends on an android "
                + "library project, so the resources '"
                + AndroidCommon.getAndroidResources(ruleContext).getLabel()
                + "' should have the attribute inline_constants set to 0");
        throw new RuleErrorException();
      }
    }
    return resourceContainer;
  }

  /** Uses the resource apk from the resources attribute, as opposed to recompiling. */
  public ResourceApk useCurrentResources(
      RuleContext ruleContext, Artifact proguardCfg, @Nullable Artifact mainDexProguardCfg) {
    ResourceContainer resourceContainer =
        Iterables.getOnlyElement(
            AndroidCommon.getAndroidResources(ruleContext).getDirectAndroidResources());

    new AndroidAaptActionHelper(
            ruleContext, resourceContainer.getManifest(), Lists.newArrayList(resourceContainer))
        .createGenerateProguardAction(proguardCfg, mainDexProguardCfg);

    return new ResourceApk(
        resourceContainer.getApk(),
        null /* javaSrcJar */,
        null /* javaClassJar */,
        ResourceDependencies.empty(),
        resourceContainer,
        manifest,
        proguardCfg,
        mainDexProguardCfg,
        false);
  }

  /**
   * Packages up the manifest with resources, and generates the R.java.
   *
   * @throws InterruptedException
   * @deprecated in favor of {@link ApplicationManifest#packBinaryWithDataAndResources} and {@link
   *     ApplicationManifest#packLibraryWithDataAndResources}.
   */
  @Deprecated
  public ResourceApk packWithResources(
      Artifact resourceApk,
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      boolean createSource,
      Artifact proguardCfg,
      @Nullable Artifact mainDexProguardCfg)
      throws InterruptedException, RuleErrorException {

    TransitiveInfoCollection resourcesPrerequisite =
        ruleContext.getPrerequisite("resources", Mode.TARGET);
    ResourceContainer resourceContainer =
        Iterables.getOnlyElement(
            resourcesPrerequisite
                .getProvider(AndroidResourcesProvider.class)
                .getDirectAndroidResources());
    // It's ugly, but flattening now is more performant given the rest of the checks.
    List<ResourceContainer> resourceContainers =
        ImmutableList.<ResourceContainer>builder()
            // .add(resourceContainer)
            .addAll(resourceDeps.getResourceContainers())
            .build();

    // Dealing with Android library projects
    if (Iterables.size(resourceDeps.getResourceContainers()) > 1) {
      if (resourceContainer.getConstantsInlined()
          && !resourceContainer.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
        ruleContext.ruleError(
            "This android_binary depends on an android_library, so the"
                + " resources '"
                + AndroidCommon.getAndroidResources(ruleContext).getLabel()
                + "' should have the attribute inline_constants set to 0");
        return null;
      }
    }

    // This binary depends on a library project, so we need to regenerate the
    // resources. The resulting sources and apk will combine all the resources
    // contained in the transitive closure of the binary.
    AndroidAaptActionHelper aaptActionHelper =
        new AndroidAaptActionHelper(
            ruleContext, getManifest(), Lists.newArrayList(resourceContainers));

    ResourceFilterFactory resourceFilterFactory =
        ResourceFilterFactory.fromRuleContext(ruleContext);

    List<String> uncompressedExtensions;
    if (ruleContext.getRule().isAttrDefined(
        AndroidRuleClasses.NOCOMPRESS_EXTENSIONS_ATTR, Type.STRING_LIST)) {
      uncompressedExtensions =
          ruleContext
              .getExpander()
              .withDataLocations()
              .tokenized(AndroidRuleClasses.NOCOMPRESS_EXTENSIONS_ATTR);
    } else {
      // This code is also used by android_test, which doesn't have this attribute.
      uncompressedExtensions = ImmutableList.of();
    }

    ImmutableList.Builder<String> additionalAaptOpts = ImmutableList.builder();

    for (String extension : uncompressedExtensions) {
      additionalAaptOpts.add("-0").add(extension);
    }
    if (resourceFilterFactory.hasConfigurationFilters()) {
      additionalAaptOpts.add("-c").add(resourceFilterFactory.getConfigurationFilterString());
    }

    Artifact javaSourcesJar = null;

    if (createSource) {
      javaSourcesJar =
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR);
      aaptActionHelper.createGenerateResourceSymbolsAction(
          javaSourcesJar, null, resourceContainer.getJavaPackage(), true);
    }

    aaptActionHelper.createGenerateApkAction(
        resourceApk,
        resourceContainer.getRenameManifestPackage(),
        additionalAaptOpts.build(),
        resourceFilterFactory.getDensities());

    ResourceContainer updatedResources =
        resourceContainer
            .toBuilder()
            .setLabel(ruleContext.getLabel())
            .setJavaClassJar(null) // remove the resource class jar to force a regeneration.
            .setApk(resourceApk)
            .setManifest(getManifest())
            .setJavaSourceJar(javaSourcesJar)
            .build();

    aaptActionHelper.createGenerateProguardAction(proguardCfg, mainDexProguardCfg);

    return new ResourceApk(
        resourceApk,
        updatedResources.getJavaSourceJar(),
        updatedResources.getJavaClassJar(),
        resourceDeps,
        updatedResources,
        manifest,
        proguardCfg,
        mainDexProguardCfg,
        true);
  }

  public Artifact getManifest() {
    return manifest;
  }
}
