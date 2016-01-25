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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
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
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {
  public static ApplicationManifest fromResourcesRule(RuleContext ruleContext) {
    final AndroidResourcesProvider resources = AndroidCommon.getAndroidResources(ruleContext);
    if (resources == null) {
      ruleContext.attributeError("manifest", "a resources or manifest attribute is mandatory.");
      return null;
    }
    return new ApplicationManifest(Iterables.getOnlyElement(
        resources.getDirectAndroidResources())
        .getManifest());
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

    String overridePackage = getOverridePackage(ruleContext);
    if (overridePackage != null) {
      builder
          .addArgument("--override_package")
          .addArgument(overridePackage);
    }

    ruleContext.registerAction(builder.build(ruleContext));
    return new ApplicationManifest(result);
  }

  private String getOverridePackage(RuleContext ruleContext) {
    // It seems that we sometimes rename the app for God-knows-what reason. If that is the case,
    // pass this information to the stubifier script.
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("application_id")) {
      return ruleContext.attributes().get("application_id", Type.STRING);
    }

    AndroidResourcesProvider resourcesProvider = AndroidCommon.getAndroidResources(ruleContext);
    if (resourcesProvider != null) {
      ResourceContainer resourceContainer = Iterables.getOnlyElement(
          resourcesProvider.getDirectAndroidResources());
      return resourceContainer.getRenameManifestPackage();
    } else {
      return null;
    }
  }

  public ApplicationManifest addStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.STUB_APPLICATON_MANIFEST);

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
        .setProgressMessage("Injecting stub application")
        .setMnemonic("InjectStubApplication")
        .addArgument("--input_manifest")
        .addInputArgument(manifest)
        .addArgument("--output_manifest")
        .addOutputArgument(stubManifest)
        .addArgument("--output_datafile")
        .addOutputArgument(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.STUB_APPLICATION_DATA));

    String overridePackage = getOverridePackage(ruleContext);
    if (overridePackage != null) {
      builder.addArgument("--override_package");
      builder.addArgument(overridePackage);
    }

    ruleContext.registerAction(builder.build(ruleContext));

    return new ApplicationManifest(stubManifest);
  }

  public static ApplicationManifest fromRule(RuleContext ruleContext) {
    return new ApplicationManifest(ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET));
  }

  public static ApplicationManifest fromExplicitManifest(Artifact manifest) {
    return new ApplicationManifest(manifest);
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
    ruleContext.getAnalysisEnvironment().registerAction(new FileWriteAction(
        ruleContext.getActionOwner(), generatedManifest, contents, false /* makeExecutable */));
    return new ApplicationManifest(generatedManifest);
  }

  private final Artifact manifest;

  private ApplicationManifest(Artifact manifest) {
    this.manifest = manifest;
  }

  public ApplicationManifest mergeWith(RuleContext ruleContext,
      ResourceDependencies resourceDeps) {
    Iterable<Artifact> mergeeManifests = getMergeeManifests(resourceDeps.getResources()); 
    if (!Iterables.isEmpty(mergeeManifests)) {
      Iterable<Artifact> exportedManifests = mergeeManifests;
      Artifact outputManifest = ruleContext.getUniqueDirectoryArtifact(
          ruleContext.getRule().getName() + "_merged", "AndroidManifest.xml",
          ruleContext.getBinOrGenfilesDirectory());
      AndroidManifestMergeHelper.createMergeManifestAction(ruleContext, getManifest(),
          exportedManifests, ImmutableList.of("all"), outputManifest);
      return new ApplicationManifest(outputManifest);
    }
    return this;
  }

  private static Iterable<Artifact> getMergeeManifests(
      Iterable<ResourceContainer> resourceContainers) {
    ImmutableSortedSet.Builder<Artifact> builder =
        ImmutableSortedSet.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
    for (ResourceContainer r : resourceContainers) {
      if (r.isManifestExported()) {
        builder.add(r.getManifest());
      }
    }
    return builder.build();
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
            ruleContext.getPrerequisites(
                // TODO(bazel-team): Remove the ResourceType construct.
                ResourceType.ASSETS.getAttribute(),
                Mode.TARGET,
                FileProvider.class)).build();

    return createApk(resourceApk,
        ruleContext,
        resourceDeps,
        rTxt,
        null, /* configurationFilters */
        ImmutableList.<String>of(), /* uncompressedExtensions */
        ImmutableList.<String>of(), /* densities */
        ImmutableList.<String>of(), /* String applicationId */
        null, /* String versionCode */
        null, /* String versionName */
        null, /* Artifact symbolsTxt */
        incremental,
        data,
        proguardCfg,
        null);
  }

  /** Packages up the manifest with resource and assets from the rule and dependent resources. 
   * @param manifestOut TODO(corysmith):
   * @throws InterruptedException */
  public ResourceApk packWithDataAndResources(
      Artifact resourceApk,
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbolsTxt,
      List<String> configurationFilters,
      List<String> uncompressedExtensions,
      List<String> densities,
      String applicationId,
      String versionCode,
      String versionName,
      boolean incremental,
      Artifact proguardCfg,
      Artifact manifestOut) throws InterruptedException {
    LocalResourceContainer data = new LocalResourceContainer.Builder(ruleContext)
        .withAssets(
            AndroidCommon.getAssetDir(ruleContext),
            ruleContext.getPrerequisites(
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
    return createApk(resourceApk,
        ruleContext,
        resourceDeps,
        rTxt,
        symbolsTxt,
        configurationFilters,
        uncompressedExtensions,
        densities,
        applicationId,
        versionCode,
        versionName,
        incremental,
        data,
        proguardCfg,
        manifestOut);
  }

  private ResourceApk createApk(Artifact resourceApk,
      RuleContext ruleContext,
      ResourceDependencies resourceDeps,
      Artifact rTxt,
      Artifact symbolsTxt,
      List<String> configurationFilters,
      List<String> uncompressedExtensions,
      List<String> densities,
      String applicationId,
      String versionCode,
      String versionName,
      boolean incremental,
      LocalResourceContainer data,
      Artifact proguardCfg,
      Artifact manifestOut) throws InterruptedException {
    ResourceContainer resourceContainer = checkForInlinedResources(
        new AndroidResourceContainerBuilder()
            .withData(data)
            .withManifest(getManifest())
            .withROutput(rTxt)
            .withSymbolsFile(symbolsTxt)
            .buildFromRule(ruleContext, resourceApk),
        resourceDeps.getResources(),  // TODO(bazel-team): Figure out if we really need to check
        // the ENTIRE transitive closure, or just the direct dependencies. Given that each rule with
        // resources would check for inline resources, we can rely on the previous rule to have
        // checked its dependencies.
        ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }

    AndroidResourcesProcessorBuilder builder =
        new AndroidResourcesProcessorBuilder(ruleContext)
            .setApkOut(resourceContainer.getApk())
            .setConfigurationFilters(configurationFilters)
            .setUncompressedExtensions(uncompressedExtensions)
            .setJavaPackage(resourceContainer.getJavaPackage())
            .setDebug(ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
            .setManifestOut(manifestOut)
            .withPrimary(resourceContainer)
            .withDependencies(resourceDeps)
            .setDensities(densities)
            .setProguardOut(proguardCfg)
            .setApplicationId(applicationId)
            .setVersionCode(versionCode)
            .setVersionName(versionName);

    if (!incremental) {
      builder
          .setRTxtOut(resourceContainer.getRTxt())
          .setSymbolsTxt(resourceContainer.getSymbolsTxt())
          .setSourceJarOut(resourceContainer.getJavaSourceJar());
    }

    ResourceContainer processed = builder.build(ruleContext);

    return new ResourceApk(
        resourceApk, processed.getJavaSourceJar(), resourceDeps, processed, manifest,
        proguardCfg, false);
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
  public ResourceApk useCurrentResources(RuleContext ruleContext, Artifact proguardCfg) {
    ResourceContainer resourceContainer = Iterables.getOnlyElement(
        AndroidCommon.getAndroidResources(ruleContext).getDirectAndroidResources());

    new AndroidAaptActionHelper(
        ruleContext,
        resourceContainer.getManifest(),
        Lists.newArrayList(resourceContainer)).createGenerateProguardAction(proguardCfg);

    return new ResourceApk(
        resourceContainer.getApk(),
        null /* javaSrcJar */,
        ResourceDependencies.empty(),
        resourceContainer,
        manifest,
        proguardCfg,
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
      Artifact proguardCfg) throws InterruptedException {

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

    ResourceContainer updatedResources = new ResourceContainer(
        ruleContext.getLabel(),
        resourceContainer.getJavaPackage(),
        resourceContainer.getRenameManifestPackage(),
        resourceContainer.getConstantsInlined(),
        resourceApk,
        getManifest(),
        javaSourcesJar,
        resourceContainer.getArtifacts(ResourceType.ASSETS),
        resourceContainer.getArtifacts(ResourceType.RESOURCES),
        resourceContainer.getRoots(ResourceType.ASSETS),
        resourceContainer.getRoots(ResourceType.RESOURCES),
        resourceContainer.isManifestExported(),
        resourceContainer.getRTxt(), null);

    aaptActionHelper.createGenerateProguardAction(proguardCfg);

    return new ResourceApk(resourceApk, updatedResources.getJavaSourceJar(),
        resourceDeps, updatedResources, manifest, proguardCfg, true);
  }

  public Artifact getManifest() {
    return manifest;
  }
}
