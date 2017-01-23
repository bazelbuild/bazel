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
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidManifestMerger;
import com.google.devtools.build.lib.rules.android.ResourceContainer.Builder.JavaPackageSource;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {
  public static ApplicationManifest fromResourcesRule(RuleContext ruleContext) {
    final AndroidResourcesProvider resources = AndroidCommon.getAndroidResources(ruleContext);
    if (resources == null) {
      ruleContext.attributeError("manifest", "a resources or manifest attribute is mandatory.");
      return null;
    }
    return new ApplicationManifest(
        ruleContext, Iterables.getOnlyElement(resources.getDirectAndroidResources()).getManifest());
  }

  public ApplicationManifest createSplitManifest(
      RuleContext ruleContext, String splitName, boolean hasCode) {
    // aapt insists that manifests be called AndroidManifest.xml, even though they have to be
    // explicitly designated as manifests on the command line
    Artifact result = AndroidBinary.getDxArtifact(
        ruleContext, "split_" + splitName + "/AndroidManifest.xml");
    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$build_split_manifest", Mode.HOST))
        .setProgressMessage("Creating manifest for split " + splitName)
        .setMnemonic("AndroidBuildSplitManifest")
        .addArgument("--main_manifest")
        .addInputArgument(manifest)
        .addArgument("--split_manifest")
        .addOutputArgument(result)
        .addArgument("--split")
        .addArgument(splitName)
        .addArgument(hasCode ? "--hascode" : "--nohascode");

    String overridePackage = manifestValues.get("applicationId");
    if (overridePackage != null) {
      builder
          .addArgument("--override_package")
          .addArgument(overridePackage);
    }

    ruleContext.registerAction(builder.build(ruleContext));
    return new ApplicationManifest(ruleContext, result);
  }

  public ApplicationManifest addMobileInstallStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest = ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_MANIFEST);

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
        .setProgressMessage("Injecting mobile install stub application")
        .setMnemonic("InjectMobileInstallStubApplication")
        .addArgument("--mode=mobile_install")
        .addArgument("--input_manifest")
        .addInputArgument(manifest)
        .addArgument("--output_manifest")
        .addOutputArgument(stubManifest)
        .addArgument("--output_datafile")
        .addOutputArgument(ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_DATA));

    String overridePackage = manifestValues.get("applicationId");
    if (overridePackage != null) {
      builder.addArgument("--override_package");
      builder.addArgument(overridePackage);
    }

    ruleContext.registerAction(builder.build(ruleContext));

    return new ApplicationManifest(ruleContext, stubManifest);
  }

  public ApplicationManifest addInstantRunStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.INSTANT_RUN_STUB_APPLICATION_MANIFEST);

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
        .setProgressMessage("Injecting instant run stub application")
        .setMnemonic("InjectInstantRunStubApplication")
        .addArgument("--mode=instant_run")
        .addArgument("--input_manifest")
        .addInputArgument(manifest)
        .addArgument("--output_manifest")
        .addOutputArgument(stubManifest);

    ruleContext.registerAction(builder.build(ruleContext));

    return new ApplicationManifest(ruleContext, stubManifest);
  }

  public static ApplicationManifest fromRule(RuleContext ruleContext) {
    return new ApplicationManifest(
        ruleContext, ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET));
  }

  public static ApplicationManifest fromExplicitManifest(
      RuleContext ruleContext, Artifact manifest) {
    return new ApplicationManifest(ruleContext, manifest);
  }

  /**
   * Generates an empty manifest for a rule that does not directly specify resources.
   *
   * <p><strong>Note:</strong> This generated manifest can then be used as the primary manifest
   * when merging with dependencies.
   *
   * @return the generated ApplicationManifest
   */
  public static ApplicationManifest generatedManifest(RuleContext ruleContext) {
    Artifact generatedManifest = ruleContext.getUniqueDirectoryArtifact(
        ruleContext.getRule().getName() + "_generated", new PathFragment("AndroidManifest.xml"),
        ruleContext.getBinOrGenfilesDirectory());

    String manifestPackage = AndroidCommon.getJavaPackage(ruleContext);
    String contents = Joiner.on("\n").join(
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
    return new ApplicationManifest(ruleContext, generatedManifest);
  }

  private static ImmutableMap<String, String> getManifestValues(RuleContext context) {
    Map<String, String> manifestValues = new TreeMap<>();
    // applicationId is set from manifest_values or android_resources.rename_manifest_package
    // with descending priority.
    AndroidResourcesProvider resourcesProvider = AndroidCommon.getAndroidResources(context);
    if (resourcesProvider != null) {
      ResourceContainer resourceContainer = Iterables.getOnlyElement(
          resourcesProvider.getDirectAndroidResources());
      if (resourceContainer.getRenameManifestPackage() != null) {
        manifestValues.put("applicationId", resourceContainer.getRenameManifestPackage());
      }
    }
    if (context.attributes().isAttributeValueExplicitlySpecified("manifest_values")) {
      manifestValues.putAll(context.attributes().get("manifest_values", Type.STRING_DICT));
    }

    for (String variable : manifestValues.keySet()) {
      manifestValues.put(
          variable, context.expandMakeVariables("manifest_values", manifestValues.get(variable)));
    }
    return ImmutableMap.copyOf(manifestValues);
  }

  private final Artifact manifest;
  private final ImmutableMap<String, String> manifestValues;

  private ApplicationManifest(RuleContext ruleContext, Artifact manifest) {
    this.manifest = manifest;
    this.manifestValues = getManifestValues(ruleContext);
  }

  public ApplicationManifest mergeWith(RuleContext ruleContext, ResourceDependencies resourceDeps) {
    Map<Artifact, Label> mergeeManifests = getMergeeManifests(resourceDeps.getResources());

    boolean legacy = true;
    if (ruleContext.isLegalFragment(AndroidConfiguration.class)
        && ruleContext.getRule().isAttrDefined("manifest_merger", STRING)) {
      AndroidManifestMerger merger = AndroidManifestMerger.fromString(
          ruleContext.attributes().get("manifest_merger", STRING));
      if (merger == null) {
        merger = ruleContext.getFragment(AndroidConfiguration.class).getManifestMerger();
      }
      legacy = merger == AndroidManifestMerger.LEGACY;
    }

    if (legacy) {
      if (!mergeeManifests.isEmpty()) {
        Artifact outputManifest = ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_merged", "AndroidManifest.xml",
            ruleContext.getBinOrGenfilesDirectory());
        AndroidManifestMergeHelper.createMergeManifestAction(ruleContext, getManifest(),
            mergeeManifests.keySet(), ImmutableList.of("all"), outputManifest);
        return new ApplicationManifest(ruleContext, outputManifest);
      }
    } else {
      if (!mergeeManifests.isEmpty() || !manifestValues.isEmpty()) {
        Artifact outputManifest = ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_merged", "AndroidManifest.xml",
            ruleContext.getBinOrGenfilesDirectory());
        Artifact mergeLog = ruleContext.getUniqueDirectoryArtifact(
            ruleContext.getRule().getName() + "_merged", "manifest_merger_log.txt",
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
        return new ApplicationManifest(ruleContext, outputManifest);
      }
    }
    return this;
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
    Artifact outputManifest = ruleContext.getUniqueDirectoryArtifact(
        ruleContext.getRule().getName() + "_renamed", "AndroidManifest.xml",
        ruleContext.getBinOrGenfilesDirectory());
    new ManifestMergerActionBuilder(ruleContext)
        .setManifest(getManifest())
        .setLibrary(true)
        .setCustomPackage(customPackage)
        .setManifestOutput(outputManifest)
        .build(ruleContext);
    return new ApplicationManifest(ruleContext, outputManifest);
  }

  /** Packages up the manifest with assets from the rule and dependent resources.
   * @throws InterruptedException */
  public ResourceApk packWithAssets(
      Artifact resourceApk,
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      boolean incremental,
      Artifact proguardCfg) throws InterruptedException {
    LocalResourceContainer data = new LocalResourceContainer.Builder(ruleContext)
        .withAssets(
            AndroidCommon.getAssetDir(ruleContext),
            ruleContext.getPrerequisitesIf(
                // TODO(bazel-team): Remove the ResourceType construct.
                ResourceType.ASSETS.getAttribute(),
                Mode.TARGET,
                FileProvider.class)).build();

    return createApk(
        ruleContext,
        false, /* isLibrary */
        resourceDeps,
        ImmutableList.<String>of(), /* configurationFilters */
        ImmutableList.<String>of(), /* uncompressedExtensions */
        true, /* crunchPng */
        ImmutableList.<String>of(), /* densities */
        incremental,
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setRTxt(rTxt)
            .setApk(resourceApk)
            .build(),
        data,
        proguardCfg,
        null, /* Artifact mainDexProguardCfg */
        null, /* Artifact manifestOut */
        null /* Artifact mergedResources */,
        null /* Artifact dataBindingInfoZip */);
  }

  /** Packages up the manifest with resource and assets from the LocalResourceContainer. */
  public ResourceApk packWithDataAndResources(
      RuleContext ruleContext,
      LocalResourceContainer data,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      Artifact manifestOut,
      Artifact mergedResources,
      boolean alwaysExportManifest) throws InterruptedException {
    if (ruleContext.hasErrors()) {
      return null;
    }
    ResourceContainer.Builder resourceContainer =
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setRTxt(rTxt)
            .setSymbols(symbols)
            .setJavaPackageFrom(JavaPackageSource.MANIFEST);
    if (alwaysExportManifest) {
      resourceContainer.setManifestExported(true);
    }
    return createApk(
        ruleContext,
        true, /* isLibrary */
        resourceDeps,
        ImmutableList.<String>of(), /* List<String> configurationFilters */
        ImmutableList.<String>of(), /* List<String> uncompressedExtensions */
        false, /* crunchPng */
        ImmutableList.<String>of(), /* List<String> densities */
        false, /* incremental */
        resourceContainer.build(),
        data,
        null, /* Artifact proguardCfg */
        null, /* Artifact mainDexProguardCfg */
        manifestOut,
        mergedResources,
        null /* Artifact dataBindingInfoZip */);
  }

  /** Packages up the manifest with resource and assets from the rule and dependent resources. */
  public ResourceApk packWithDataAndResources(
      @Nullable Artifact resourceApk,
      RuleContext ruleContext,
      boolean isLibrary,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbols,
      List<String> configurationFilters,
      List<String> uncompressedExtensions,
      boolean crunchPng,
      List<String> densities,
      boolean incremental,
      Artifact proguardCfg,
      @Nullable Artifact mainDexProguardCfg,
      Artifact manifestOut,
      Artifact mergedResources,
      Artifact dataBindingInfoZip) throws InterruptedException {
    LocalResourceContainer data = new LocalResourceContainer.Builder(ruleContext)
        .withAssets(
            AndroidCommon.getAssetDir(ruleContext),
            ruleContext.getPrerequisitesIf(
                // TODO(bazel-team): Remove the ResourceType construct.
                ResourceType.ASSETS.getAttribute(),
                Mode.TARGET,
                FileProvider.class))
        .withResources(
            ruleContext.getPrerequisites(
                "resource_files",
                Mode.TARGET,
                FileProvider.class)).build();
    if (ruleContext.hasErrors()) {
      return null;
    }
    return createApk(
        ruleContext,
        isLibrary,
        resourceDeps,
        configurationFilters,
        uncompressedExtensions,
        crunchPng,
        densities,
        incremental,
        ResourceContainer.builderFromRule(ruleContext)
            .setAssetsAndResourcesFrom(data)
            .setManifest(getManifest())
            .setRTxt(rTxt)
            .setSymbols(symbols)
            .setApk(resourceApk)
            .build(),
        data,
        proguardCfg,
        mainDexProguardCfg,
        manifestOut,
        mergedResources, dataBindingInfoZip);
  }

  private ResourceApk createApk(
      RuleContext ruleContext,
      boolean isLibrary,
      ResourceDependencies resourceDeps,
      List<String> configurationFilters,
      List<String> uncompressedExtensions,
      boolean crunchPng,
      List<String> densities,
      boolean incremental,
      ResourceContainer maybeInlinedResourceContainer,
      LocalResourceContainer data,
      Artifact proguardCfg,
      @Nullable Artifact mainDexProguardCfg,
      Artifact manifestOut,
      Artifact mergedResources,
      Artifact dataBindingInfoZip) throws InterruptedException {
    ResourceContainer resourceContainer = checkForInlinedResources(
        maybeInlinedResourceContainer,
        resourceDeps.getResources(),  // TODO(bazel-team): Figure out if we really need to check
        // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule with
        // resources would check for inline resources, we can rely on the previous rule to have
        // checked its dependencies.
        ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }

    ResourceContainer processed;
    if (isLibrary && AndroidCommon.getAndroidConfig(ruleContext).useParallelResourceProcessing()) {
      // android_library should only build the APK one way (!incremental).
      Preconditions.checkArgument(!incremental);
      Artifact rJavaClassJar = ruleContext.getImplicitOutputArtifact(
          AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

      if (resourceContainer.getSymbols() != null) {
        new AndroidResourceParsingActionBuilder(ruleContext)
            .withPrimary(resourceContainer)
            .setParse(data)
            .setOutput(resourceContainer.getSymbols())
            .build(ruleContext);
      }

      AndroidResourceMergingActionBuilder resourcesMergerBuilder =
          new AndroidResourceMergingActionBuilder(ruleContext)
              .setJavaPackage(resourceContainer.getJavaPackage())
              .withPrimary(resourceContainer)
              .withDependencies(resourceDeps)
              .setMergedResourcesOut(mergedResources)
              .setManifestOut(manifestOut)
              .setClassJarOut(rJavaClassJar);
      ResourceContainer merged = resourcesMergerBuilder.build(ruleContext);

      AndroidResourceValidatorActionBuilder validatorBuilder =
          new AndroidResourceValidatorActionBuilder(ruleContext)
              .setJavaPackage(merged.getJavaPackage())
              .setDebug(
                  ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
              .setMergedResources(mergedResources)
              .withPrimary(merged)
              .setSourceJarOut(merged.getJavaSourceJar())
              .setRTxtOut(merged.getRTxt());
      processed = validatorBuilder.build(ruleContext);
    } else {
      AndroidResourcesProcessorBuilder builder =
          new AndroidResourcesProcessorBuilder(ruleContext)
              .setLibrary(isLibrary)
              .setApkOut(resourceContainer.getApk())
              .setConfigurationFilters(configurationFilters)
              .setUncompressedExtensions(uncompressedExtensions)
              .setCrunchPng(crunchPng)
              .setJavaPackage(resourceContainer.getJavaPackage())
              .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
              .setManifestOut(manifestOut)
              .setMergedResourcesOut(mergedResources)
              .withPrimary(resourceContainer)
              .withDependencies(resourceDeps)
              .setDensities(densities)
              .setProguardOut(proguardCfg)
              .setMainDexProguardOut(mainDexProguardCfg)
              .setDataBindingInfoZip(dataBindingInfoZip)
              .setApplicationId(manifestValues.get("applicationId"))
              .setVersionCode(manifestValues.get("versionCode"))
              .setVersionName(manifestValues.get("versionName"));
      if (!incremental) {
        builder
            .setRTxtOut(resourceContainer.getRTxt())
            .setSymbols(resourceContainer.getSymbols())
            .setSourceJarOut(resourceContainer.getJavaSourceJar());
      }
      processed = builder.build(ruleContext);
    }

    return new ResourceApk(
        resourceContainer.getApk(), processed.getJavaSourceJar(), processed.getJavaClassJar(),
        resourceDeps, processed, processed.getManifest(),
        proguardCfg, mainDexProguardCfg, false);
  }

  private static ResourceContainer checkForInlinedResources(ResourceContainer resourceContainer,
      Iterable<ResourceContainer> resourceContainers, RuleContext ruleContext) {
    // Dealing with Android library projects
    if (Iterables.size(resourceContainers) > 1) {
      if (resourceContainer.getConstantsInlined()
          && !resourceContainer.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
        ruleContext.ruleError("This android binary depends on an android "
            + "library project, so the resources '"
            + AndroidCommon.getAndroidResources(ruleContext).getLabel()
            + "' should have the attribute inline_constants set to 0");
        return null;
      }
    }
    return resourceContainer;
  }

  /** Uses the resource apk from the resources attribute, as opposed to recompiling. */
  public ResourceApk useCurrentResources(
      RuleContext ruleContext, Artifact proguardCfg, @Nullable Artifact mainDexProguardCfg) {
    ResourceContainer resourceContainer = Iterables.getOnlyElement(
        AndroidCommon.getAndroidResources(ruleContext).getDirectAndroidResources());

    new AndroidAaptActionHelper(
        ruleContext,
        resourceContainer.getManifest(),
        Lists.newArrayList(resourceContainer))
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
   * @throws InterruptedException
   *
   * @deprecated in favor of {@link ApplicationManifest#packWithDataAndResources}.
   */
  @Deprecated
  public ResourceApk packWithResources(
      Artifact resourceApk,
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      boolean createSource,
      Artifact proguardCfg,
      @Nullable Artifact mainDexProguardCfg) throws InterruptedException {

    TransitiveInfoCollection resourcesPrerequisite =
        ruleContext.getPrerequisite("resources", Mode.TARGET);
    ResourceContainer resourceContainer = Iterables.getOnlyElement(
        resourcesPrerequisite.getProvider(AndroidResourcesProvider.class)
        .getDirectAndroidResources());
    // It's ugly, but flattening now is more performant given the rest of the checks.
    List<ResourceContainer> resourceContainers =
        ImmutableList.<ResourceContainer>builder()
        //.add(resourceContainer)
        .addAll(resourceDeps.getResources()).build();

    // Dealing with Android library projects
    if (Iterables.size(resourceDeps.getResources()) > 1) {
      if (resourceContainer.getConstantsInlined()
          && !resourceContainer.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
        ruleContext.ruleError("This android_binary depends on an android_library, so the"
            + " resources '" + AndroidCommon.getAndroidResources(ruleContext).getLabel()
            + "' should have the attribute inline_constants set to 0");
        return null;
      }
    }

    // This binary depends on a library project, so we need to regenerate the
    // resources. The resulting sources and apk will combine all the resources
    // contained in the transitive closure of the binary.
    AndroidAaptActionHelper aaptActionHelper = new AndroidAaptActionHelper(ruleContext,
        getManifest(), Lists.newArrayList(resourceContainers));

    List<String> resourceConfigurationFilters =
        ruleContext.getTokenizedStringListAttr("resource_configuration_filters");
    List<String> uncompressedExtensions =
        ruleContext.getTokenizedStringListAttr("nocompress_extensions");

    ImmutableList.Builder<String> additionalAaptOpts = ImmutableList.<String>builder();

    for (String extension : uncompressedExtensions) {
      additionalAaptOpts.add("-0").add(extension);
    }
    if (!resourceConfigurationFilters.isEmpty()) {
      additionalAaptOpts.add("-c").add(Joiner.on(",").join(resourceConfigurationFilters));
    }

    Artifact javaSourcesJar = null;

    if (createSource) {
      javaSourcesJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR);
      aaptActionHelper.createGenerateResourceSymbolsAction(
          javaSourcesJar, null, resourceContainer.getJavaPackage(), true);
    }

    List<String> densities = ruleContext.getTokenizedStringListAttr("densities");
    aaptActionHelper.createGenerateApkAction(resourceApk,
        resourceContainer.getRenameManifestPackage(), additionalAaptOpts.build(), densities);

    ResourceContainer updatedResources = resourceContainer.toBuilder()
        .setLabel(ruleContext.getLabel())
        .setApk(resourceApk)
        .setManifest(getManifest())
        .setJavaSourceJar(javaSourcesJar)
        .setJavaClassJar(null)
        .setSymbols(null)
        .build();

    aaptActionHelper.createGenerateProguardAction(proguardCfg, mainDexProguardCfg);

    return new ResourceApk(resourceApk,
        updatedResources.getJavaSourceJar(),
        updatedResources.getJavaClassJar(),
        resourceDeps, updatedResources, manifest, proguardCfg, mainDexProguardCfg, true);
  }

  public Artifact getManifest() {
    return manifest;
  }
}
