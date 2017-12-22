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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
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
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {

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
    Map<Artifact, Label> mergeeManifests = getMergeeManifests(resourceDeps.getResourceContainers());

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
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setApk(resourceApk)
            .setRTxt(rTxt)
            .build();

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
        null);
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
        ResourceContainer.builderFromRule(ruleContext)
            .setRTxt(rTxt)
            .setSymbols(symbols)
            .setJavaPackageFrom(JavaPackageSource.MANIFEST)
            .setManifestExported(true)
            .setManifest(getManifest())
            .setAssetsAndResourcesFrom(data)
            .build();

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
        null);
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
        ResourceContainer.builderFromRule(ruleContext)
            .setApk(resourceApk)
            .setManifest(getManifest())
            .setAssetsAndResourcesFrom(data)
            .build();

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
        null);
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
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setRTxt(rTxt)
            .setApk(resourceApk)
            .build();

    AndroidConfiguration androidConfiguration =
        ruleContext.getConfiguration().getFragment(AndroidConfiguration.class);

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
        mainDexProguardCfg);
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

    ResourceContainer resourceContainer = builder.build();

    Artifact rJavaClassJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    AndroidConfiguration androidConfiguration =
        ruleContext.getConfiguration().getFragment(AndroidConfiguration.class);

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
        null);
  }

  public Artifact getManifest() {
    return manifest;
  }
}
