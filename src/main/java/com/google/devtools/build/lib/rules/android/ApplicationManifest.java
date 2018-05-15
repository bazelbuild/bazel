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
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
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
import java.util.Optional;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {

  public ApplicationManifest createSplitManifest(
      RuleContext ruleContext, String splitName, boolean hasCode) {
    Artifact result = createSplitManifest(ruleContext, manifest, splitName, hasCode);
    return new ApplicationManifest(ruleContext, result, targetAaptVersion);
  }

  static Artifact createSplitManifest(
      RuleContext ruleContext, Artifact manifest, String splitName, boolean hasCode) {
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

    String overridePackage = getManifestValues(ruleContext).get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));
    return result;
  }

  public ApplicationManifest addMobileInstallStubApplication(RuleContext ruleContext)
      throws InterruptedException {
    Artifact stubManifest = addMobileInstallStubApplication(ruleContext, manifest);
    return new ApplicationManifest(ruleContext, stubManifest, targetAaptVersion);
  }

  static Artifact addMobileInstallStubApplication(RuleContext ruleContext, Artifact manifest)
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

    String overridePackage = getManifestValues(ruleContext).get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));

    return stubManifest;
  }

  public static Artifact getManifestFromAttributes(RuleContext ruleContext) {
    return ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET);
  }

  /**
   * Gets the manifest specified in the "manifest" attribute, renaming it if needed.
   *
   * <p>Unlike {@link AndroidSemantics#getManifestForRule(RuleContext)}, this method will not
   * perform AndroidSemantics-specific manifest processing. This method will do the same work
   * regardless of the AndroidSemantics implementation being used; that method may do different work
   * depending on the implementation.
   */
  public static ApplicationManifest renamedFromRule(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    return fromExplicitManifest(
        ruleContext, renameManifestIfNeeded(ruleContext, getManifestFromAttributes(ruleContext)));
  }

  static Artifact renameManifestIfNeeded(RuleContext ruleContext, Artifact manifest)
      throws InterruptedException {
    if (manifest.getFilename().equals("AndroidManifest.xml")) {
      return manifest;
    } else {
      /*
       * If the manifest file is not named AndroidManifest.xml, we create a symlink named
       * AndroidManifest.xml to it. aapt requires the manifest to be named as such.
       */
      Artifact manifestSymlink =
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_SYMLINKED_MANIFEST);
      SymlinkAction symlinkAction =
          new SymlinkAction(
              ruleContext.getActionOwner(),
              manifest,
              manifestSymlink,
              "Renaming Android manifest for " + ruleContext.getLabel());
      ruleContext.registerAction(symlinkAction);
      return manifestSymlink;
    }
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
    return fromExplicitManifest(
        ruleContext, generateManifest(ruleContext, AndroidCommon.getJavaPackage(ruleContext)));
  }

  /**
   * Creates an action to generate an empty manifest file with a specific package name.
   *
   * @return an artifact for the generated manifest
   */
  public static Artifact generateManifest(RuleContext ruleContext, String manifestPackage) {
    Artifact generatedManifest =
        ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_generated",
            PathFragment.create("AndroidManifest.xml"),
            ruleContext.getBinOrGenfilesDirectory());

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
    return generatedManifest;
  }

  /** Gets a map of manifest values from this rule's 'manifest_values' attribute */
  static ImmutableMap<String, String> getManifestValues(RuleContext context) {
    return getManifestValues(
        context,
        context.attributes().isAttributeValueExplicitlySpecified("manifest_values")
            ? context.attributes().get("manifest_values", Type.STRING_DICT)
            : null);
  }

  /** Gets and expands an expanded map of manifest values from some raw map of manifest values. */
  static ImmutableMap<String, String> getManifestValues(
      RuleContext ruleContext, @Nullable Map<String, String> rawMap) {
    Map<String, String> manifestValues = new TreeMap<>();
    if (rawMap != null) {
      manifestValues.putAll(rawMap);
    }

    for (String variable : manifestValues.keySet()) {
      manifestValues.put(
          variable,
          ruleContext.getExpander().expand("manifest_values", manifestValues.get(variable)));
    }
    return ImmutableMap.copyOf(manifestValues);
  }

  public ImmutableMap<String, String> getManifestValues() {
    return manifestValues;
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
    return maybeMergeWith(
            ruleContext,
            manifest,
            resourceDeps,
            manifestValues,
            useLegacyMerging(ruleContext),
            AndroidCommon.getJavaPackage(ruleContext))
        .map(merged -> new ApplicationManifest(ruleContext, merged, targetAaptVersion))
        .orElse(this);
  }

  static Optional<Artifact> maybeMergeWith(
      RuleContext ruleContext,
      Artifact primaryManifest,
      ResourceDependencies resourceDeps,
      Map<String, String> manifestValues,
      boolean useLegacyMerging,
      String customPackage) {
    Map<Artifact, Label> mergeeManifests = getMergeeManifests(resourceDeps.getResourceContainers());

    if (useLegacyMerging) {
      if (!mergeeManifests.isEmpty()) {

        Artifact outputManifest =
            ruleContext.getUniqueDirectoryArtifact(
                ruleContext.getRule().getName() + "_merged",
                "AndroidManifest.xml",
                ruleContext.getBinOrGenfilesDirectory());
        AndroidManifestMergeHelper.createMergeManifestAction(
            ruleContext,
            primaryManifest,
            mergeeManifests.keySet(),
            ImmutableList.of("all"),
            outputManifest);
        return Optional.of(outputManifest);
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
            .setManifest(primaryManifest)
            .setMergeeManifests(mergeeManifests)
            .setLibrary(false)
            .setManifestValues(manifestValues)
            .setCustomPackage(customPackage)
            .setManifestOutput(outputManifest)
            .setLogOut(mergeLog)
            .build(ruleContext);
        return Optional.of(outputManifest);
      }
    }
    return Optional.empty();
  }

  /** Checks if the legacy manifest merger should be used, based on a rule attribute */
  static boolean useLegacyMerging(RuleContext ruleContext) {
    return ruleContext.isLegalFragment(AndroidConfiguration.class)
        && ruleContext.getRule().isAttrDefined("manifest_merger", STRING)
        && useLegacyMerging(ruleContext, ruleContext.attributes().get("manifest_merger", STRING));
  }

  /**
   * Checks if the legacy manifest merger should be used, based on an optional string specifying the
   * merger to use.
   */
  public static boolean useLegacyMerging(RuleContext ruleContext, @Nullable String mergerString) {
    AndroidManifestMerger merger = AndroidManifestMerger.fromString(mergerString);
    if (merger == null) {
      merger = ruleContext.getFragment(AndroidConfiguration.class).getManifestMerger();
    }
    if (merger == AndroidManifestMerger.LEGACY) {
      ruleContext.ruleWarning(
          "manifest_merger 'legacy' is deprecated. Please update to 'android'.\n"
              + "See https://developer.android.com/studio/build/manifest-merge.html for more "
              + "information about the manifest merger.");
    }

    return merger == AndroidManifestMerger.LEGACY;
  }

  private static Map<Artifact, Label> getMergeeManifests(
      Iterable<ValidatedAndroidData> transitiveData) {
    ImmutableSortedMap.Builder<Artifact, Label> builder =
        ImmutableSortedMap.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
    for (ValidatedAndroidData d : transitiveData) {
      if (d.isManifestExported()) {
        builder.put(d.getManifest(), d.getLabel());
      }
    }
    return builder.build();
  }

  public ApplicationManifest renamePackage(RuleContext ruleContext, String customPackage) {
    Optional<Artifact> stamped = maybeSetManifestPackage(ruleContext, manifest, customPackage);

    if (!stamped.isPresent()) {
      return this;
    }

    return new ApplicationManifest(ruleContext, stamped.get(), targetAaptVersion);
  }

  static Optional<Artifact> maybeSetManifestPackage(
      RuleContext ruleContext, Artifact manifest, String customPackage) {
    if (isNullOrEmpty(customPackage)) {
      return Optional.empty();
    }
    Artifact outputManifest =
        ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_renamed",
            "AndroidManifest.xml",
            ruleContext.getBinOrGenfilesDirectory());
    new ManifestMergerActionBuilder(ruleContext)
        .setManifest(manifest)
        .setLibrary(true)
        .setCustomPackage(customPackage)
        .setManifestOutput(outputManifest)
        .build(ruleContext);

    return Optional.of(outputManifest);
  }

  public ResourceApk packTestWithDataAndResources(
      RuleContext ruleContext,
      Artifact resourceApk,
      ResourceDependencies resourceDeps,
      @Nullable Artifact rTxt,
      boolean incremental,
      Artifact proguardCfg,
      Artifact mainDexProguardCfg,
      @Nullable String packageUnderTest,
      boolean hasLocalResourceFiles)
      throws InterruptedException, RuleErrorException {

    ResourceContainer resourceContainer =
        ResourceContainer.builderFromRule(ruleContext)
            .setAndroidAssets(AndroidAssets.from(ruleContext))
            .setAndroidResources(AndroidResources.from(ruleContext, "local_resource_files"))
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
            .withResourceDependencies(resourceDeps)
            .setProguardOut(proguardCfg)
            .setMainDexProguardOut(mainDexProguardCfg)
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
    ResourceContainer processed = builder.build(resourceContainer);

    ResourceContainer finalContainer =
        new RClassGeneratorActionBuilder(ruleContext)
            .targetAaptVersion(AndroidAaptVersion.chooseTargetAaptVersion(ruleContext))
            .withDependencies(resourceDeps)
            .setClassJarOut(
                ruleContext.getImplicitOutputArtifact(
                    AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
            .build(processed);

    return ResourceApk.of(finalContainer, resourceDeps, proguardCfg, mainDexProguardCfg);
  }

  /** Packages up the manifest with resource and assets from the LocalResourceContainer. */
  public ResourceApk packAarWithDataAndResources(
      RuleContext ruleContext,
      AndroidAssets assets,
      AndroidResources resources,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      Artifact manifestOut,
      Artifact mergedResources)
      throws InterruptedException {
    ResourceContainer resourceContainer =
        ResourceContainer.builderFromRule(ruleContext)
            .setRTxt(rTxt)
            .setJavaPackageFrom(JavaPackageSource.MANIFEST)
            .setManifestExported(true)
            .setManifest(getManifest())
            .build();

    // android_library should only build the APK one way (!incremental).
    Artifact rJavaClassJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    resourceContainer =
        new AndroidResourceParsingActionBuilder(ruleContext)
            .setAssets(assets)
            .setResources(resources)
            .setOutput(symbols)
            .buildAndUpdate(ruleContext, resourceContainer);

    ResourceContainer merged =
        new AndroidResourceMergingActionBuilder(ruleContext)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .withDependencies(resourceDeps)
            .setMergedResourcesOut(mergedResources)
            .setManifestOut(manifestOut)
            .setClassJarOut(rJavaClassJar)
            .setThrowOnResourceConflict(
                ruleContext
                    .getConfiguration()
                    .getFragment(AndroidConfiguration.class)
                    .throwOnResourceConflict())
            .build(ruleContext, resourceContainer);

    ResourceContainer processed =
        new AndroidResourceValidatorActionBuilder(ruleContext)
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setMergedResources(mergedResources)
            .setRTxtOut(merged.getRTxt())
            .setSourceJarOut(merged.getJavaSourceJar())
            .setApkOut(resourceContainer.getApk())
            // aapt2 related artifacts. Will be generated if the targetAaptVersion is AAPT2.
            .withDependencies(resourceDeps)
            .setCompiledSymbols(merged.getCompiledSymbols())
            .setAapt2RTxtOut(merged.getAapt2RTxt())
            .setAapt2SourceJarOut(merged.getAapt2JavaSourceJar())
            .setStaticLibraryOut(merged.getStaticLibrary())
            .build(ruleContext, merged);

    return ResourceApk.of(processed, resourceDeps);
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
    AndroidResources resources = AndroidResources.from(ruleContext, "resource_files");

    // Filter the resources during analysis to prevent processing of dependencies on unwanted
    // resources during execution.
    ResourceFilterFactory resourceFilterFactory =
        ResourceFilterFactory.fromRuleContextAndAttrs(ruleContext);
    ResourceFilter resourceFilter =
        resourceFilterFactory.getResourceFilter(ruleContext, resourceDeps, resources);
    resources = resources.filterLocalResources(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(ruleContext, resourceFilter);

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        ResourceContainer.builderFromRule(ruleContext)
            .setApk(resourceApk)
            .setManifest(getManifest())
            .setAndroidAssets(AndroidAssets.from(ruleContext))
            .setAndroidResources(resources)
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
            .withResourceDependencies(resourceDeps)
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
            .build(resourceContainer);

    // Intentionally skip building an R class JAR - incremental binaries handle this separately.

    return ResourceApk.of(processed, resourceDeps, proguardCfg, null);
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

    AndroidResources resources = AndroidResources.from(ruleContext, "resource_files");
    ResourceFilter resourceFilter =
        resourceFilterFactory.getResourceFilter(ruleContext, resourceDeps, resources);
    resources = resources.filterLocalResources(ruleContext, resourceFilter);
    resourceDeps = resourceDeps.filter(ruleContext, resourceFilter);

    // Now that the LocalResourceContainer has been filtered, we can build a filtered resource
    // container from it.
    ResourceContainer resourceContainer =
        ResourceContainer.builderFromRule(ruleContext)
            .setAndroidAssets(AndroidAssets.from(ruleContext))
            .setAndroidResources(resources)
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
            .withResourceDependencies(resourceDeps)
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
            .build(resourceContainer);

    ResourceContainer finalContainer =
        new RClassGeneratorActionBuilder(ruleContext)
            .targetAaptVersion(AndroidAaptVersion.chooseTargetAaptVersion(ruleContext))
            .withDependencies(resourceDeps)
            .setClassJarOut(
                ruleContext.getImplicitOutputArtifact(
                    AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
            .build(processed);

    return ResourceApk.of(finalContainer, resourceDeps, proguardCfg, mainDexProguardCfg);
  }

  public ResourceApk packLibraryWithDataAndResources(
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      Artifact manifestOut,
      Artifact mergedResources,
      @Nullable Artifact dataBindingInfoZip)
      throws InterruptedException, RuleErrorException {
    AndroidResources resources = AndroidResources.from(ruleContext, "resource_files");
    AndroidAssets assets = AndroidAssets.from(ruleContext);

    ResourceContainer.Builder builder =
        ResourceContainer.builderFromRule(ruleContext)
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

    AndroidResourceParsingActionBuilder parsingBuilder =
        new AndroidResourceParsingActionBuilder(ruleContext)
            .setAssets(assets)
            .setResources(resources)
            .setOutput(resourceContainer.getSymbols())
            .setCompiledSymbolsOutput(resourceContainer.getCompiledSymbols());

    if (dataBindingInfoZip != null && resourceContainer.getCompiledSymbols() != null) {
      PathFragment unusedInfo = dataBindingInfoZip.getRootRelativePath();
      // TODO(corysmith): Centralize the data binding processing and zipping into a single
      // action. Data binding processing needs to be triggered here as well as the merger to
      // avoid aapt2 from throwing an error during compilation.
      parsingBuilder
          .setDataBindingInfoZip(
              ruleContext.getDerivedArtifact(
                  unusedInfo.replaceName(unusedInfo.getBaseName() + "_unused.zip"),
                  dataBindingInfoZip.getRoot()))
          .setManifest(resourceContainer.getManifest())
          .setJavaPackage(resourceContainer.getJavaPackage());
    }
    resourceContainer = parsingBuilder.buildAndUpdate(ruleContext, resourceContainer);

    ResourceContainer merged =
        new AndroidResourceMergingActionBuilder(ruleContext)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .withDependencies(resourceDeps)
            .setThrowOnResourceConflict(androidConfiguration.throwOnResourceConflict())
            .setUseCompiledMerge(skipParsingAction)
            .setDataBindingInfoZip(dataBindingInfoZip)
            .setMergedResourcesOut(mergedResources)
            .setManifestOut(manifestOut)
            .setClassJarOut(rJavaClassJar)
            .setDataBindingInfoZip(dataBindingInfoZip)
            .build(ruleContext, resourceContainer);

    ResourceContainer processed =
        new AndroidResourceValidatorActionBuilder(ruleContext)
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setMergedResources(mergedResources)
            .setRTxtOut(merged.getRTxt())
            .setSourceJarOut(merged.getJavaSourceJar())
            .setApkOut(resourceContainer.getApk())
            // aapt2 related artifacts. Will be generated if the targetAaptVersion is AAPT2.
            .withDependencies(resourceDeps)
            .setCompiledSymbols(merged.getCompiledSymbols())
            .setAapt2RTxtOut(merged.getAapt2RTxt())
            .setAapt2SourceJarOut(merged.getAapt2JavaSourceJar())
            .setStaticLibraryOut(merged.getStaticLibrary())
            .build(ruleContext, merged);

    return ResourceApk.of(processed, resourceDeps);
  }

  public Artifact getManifest() {
    return manifest;
  }
}
