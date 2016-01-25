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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.ProguardHelper;
import com.google.devtools.build.lib.rules.java.ProguardHelper.ProguardOutput;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An implementation for the "android_binary" rule.
 */
public abstract class AndroidBinary implements RuleConfiguredTargetFactory {

  protected abstract JavaSemantics createJavaSemantics();
  protected abstract AndroidSemantics createAndroidSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    ImmutableList<TransitiveInfoCollection> deps = ImmutableList.<TransitiveInfoCollection>copyOf(
        ruleContext.getPrerequisites("deps", Mode.TARGET));
    JavaCommon javaCommon = new JavaCommon(
        ruleContext, javaSemantics, deps, deps, deps);

    AndroidCommon androidCommon = new AndroidCommon(
        javaCommon, true /* asNeverLink */, true /* exportDeps */);
    ResourceDependencies resourceDeps = LocalResourceContainer.definesAndroidResources(
        ruleContext.attributes())
        ? ResourceDependencies.fromRuleDeps(ruleContext, false /* neverlink */)
        : ResourceDependencies.fromRuleResourceAndDeps(ruleContext, false /* neverlink */);
    RuleConfiguredTargetBuilder builder = init(
        ruleContext,
        filesBuilder,
        resourceDeps,
        javaCommon,
        androidCommon,
        javaSemantics,
        androidSemantics,
        ImmutableList.<String>of("deps"));
    if (builder == null) {
      return null;
    }
    return builder.build();
  }

  private static RuleConfiguredTargetBuilder init(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      ResourceDependencies resourceDeps,
      JavaCommon javaCommon,
      AndroidCommon androidCommon,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics,
      List<String> depsAttributes) throws InterruptedException {

    if (getMultidexMode(ruleContext) != MultidexMode.LEGACY
        && ruleContext.attributes().isAttributeValueExplicitlySpecified(
            "main_dex_proguard_specs")) {
      ruleContext.attributeError("main_dex_proguard_specs", "The 'main_dex_proguard_specs' "
          + "attribute is only allowed if 'multidex' is set to 'legacy'");
      return null;
    }

    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("proguard_apply_mapping")
        && ruleContext.attributes()
            .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
            .isEmpty()) {
      ruleContext.attributeError("proguard_apply_mapping",
          "'proguard_apply_mapping' can only be used when 'proguard_specs' is also set");
      return null;
    }

    // TODO(bazel-team): Find a way to simplify this code.
    // treeKeys() means that the resulting map sorts the entries by key, which is necessary to
    // ensure determinism.
    Multimap<String, TransitiveInfoCollection> depsByArchitecture =
        MultimapBuilder.treeKeys().arrayListValues().build();
    AndroidConfiguration config = ruleContext.getFragment(AndroidConfiguration.class);
    if (config.isFatApk()) {
      for (String depsAttribute : depsAttributes) {
        for (Map.Entry<String, ? extends List<? extends TransitiveInfoCollection>> entry :
            ruleContext.getSplitPrerequisites(depsAttribute).entrySet()) {
          depsByArchitecture.putAll(entry.getKey(), entry.getValue());
        }
      }
    } else {
      for (String depsAttribute : depsAttributes) {
        depsByArchitecture.putAll(
            config.getCpu(), ruleContext.getPrerequisites(depsAttribute, Mode.TARGET));
      }
    }
    Map<String, BuildConfiguration> configurationMap = new LinkedHashMap<>();
    Map<String, CcToolchainProvider> toolchainMap = new LinkedHashMap<>();
    if (config.isFatApk()) {
      for (Map.Entry<String, ? extends List<? extends TransitiveInfoCollection>> entry :
          ruleContext.getSplitPrerequisites(":cc_toolchain_split").entrySet()) {
        TransitiveInfoCollection dep = Iterables.getOnlyElement(entry.getValue());
        CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext, dep);
        configurationMap.put(entry.getKey(), dep.getConfiguration());
        toolchainMap.put(entry.getKey(), toolchain);
      }
    } else {
      configurationMap.put(config.getCpu(), ruleContext.getConfiguration());
      toolchainMap.put(config.getCpu(), CppHelper.getToolchain(ruleContext));
    }

    NativeLibs nativeLibs = shouldLinkNativeDeps(ruleContext)
        ? NativeLibs.fromLinkedNativeDeps(ruleContext, androidSemantics.getNativeDepsFileName(),
            depsByArchitecture, toolchainMap, configurationMap)
        : NativeLibs.fromPrecompiledObjects(ruleContext, depsByArchitecture);

    // TODO(bazel-team): Resolve all the different cases of resource handling so this conditional
    // can go away: recompile from android_resources, and recompile from
    // android_binary attributes.
    ApplicationManifest applicationManifest;
    ResourceApk splitResourceApk;
    ResourceApk incrementalResourceApk;
    ResourceApk resourceApk;
    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())) {
      // Retrieve and compile the resources defined on the android_binary rule.
      if (!LocalResourceContainer.validateRuleContext(ruleContext)) {
        return null;
      }
      ApplicationManifest ruleManifest = androidSemantics.getManifestForRule(ruleContext);
      if (ruleManifest == null) {
        return null;
      }

      applicationManifest = ruleManifest.mergeWith(ruleContext, resourceDeps);
      resourceApk = applicationManifest.packWithDataAndResources(
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
          ruleContext,
          resourceDeps,
          null, /* Artifact rTxt */
          null, /* Artifact symbolsTxt */
          ruleContext.getTokenizedStringListAttr("resource_configuration_filters"),
          ruleContext.getTokenizedStringListAttr("nocompress_extensions"),
          ruleContext.getTokenizedStringListAttr("densities"),
          ruleContext.attributes().get("application_id", Type.STRING),
          getExpandedMakeVarsForAttr(ruleContext, "version_code"),
          getExpandedMakeVarsForAttr(ruleContext, "version_name"),
          false, ProguardHelper.getProguardConfigArtifact(ruleContext, ""));
      if (ruleContext.hasErrors()) {
        return null;
      }
      incrementalResourceApk = applicationManifest.addStubApplication(ruleContext)
          .packWithDataAndResources(ruleContext
                  .getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_INCREMENTAL_RESOURCES_APK),
              ruleContext,
              resourceDeps,
              null, /* Artifact rTxt */
              null, /* Artifact symbolsTxt */
              ruleContext.getTokenizedStringListAttr("resource_configuration_filters"),
              ruleContext.getTokenizedStringListAttr("nocompress_extensions"),
              ruleContext.getTokenizedStringListAttr("densities"),
              ruleContext.attributes().get("application_id", Type.STRING),
              getExpandedMakeVarsForAttr(ruleContext, "version_code"),
              getExpandedMakeVarsForAttr(ruleContext, "version_name"),
              true, ProguardHelper.getProguardConfigArtifact(ruleContext, "incremental"));
      if (ruleContext.hasErrors()) {
        return null;
      }
      splitResourceApk = applicationManifest
          .createSplitManifest(ruleContext, "android_resources", false)
          .packWithDataAndResources(getDxArtifact(ruleContext, "android_resources.ap_"),
              ruleContext,
              resourceDeps,
              null, /* Artifact rTxt */
              null, /* Artifact symbolsTxt */
              ruleContext.getTokenizedStringListAttr("resource_configuration_filters"),
              ruleContext.getTokenizedStringListAttr("nocompress_extensions"),
              ruleContext.getTokenizedStringListAttr("densities"),
              ruleContext.attributes().get("application_id", Type.STRING),
              getExpandedMakeVarsForAttr(ruleContext, "version_code"),
              getExpandedMakeVarsForAttr(ruleContext, "version_name"),
              true, ProguardHelper.getProguardConfigArtifact(ruleContext, "incremental_split"));
      if (ruleContext.hasErrors()) {
        return null;
      }
    } else {
      // Retrieve the resources from the resources attribute on the android_binary rule
      // and recompile them if necessary.
      ApplicationManifest resourcesManifest = ApplicationManifest.fromResourcesRule(ruleContext);
      if (resourcesManifest == null) {
        return null;
      }
      applicationManifest = resourcesManifest.mergeWith(ruleContext, resourceDeps);
      // Always recompiling resources causes AndroidTest to fail in certain circumstances.
      if (shouldRegenerate(ruleContext, resourceDeps)) {
        resourceApk = applicationManifest.packWithResources(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
            ruleContext,
            resourceDeps,
            true,
            ProguardHelper.getProguardConfigArtifact(ruleContext, ""));
      } else {
        resourceApk = applicationManifest.useCurrentResources(ruleContext,
            ProguardHelper.getProguardConfigArtifact(ruleContext, ""));
      }
      incrementalResourceApk = applicationManifest
          .addStubApplication(ruleContext)
          .packWithResources(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_INCREMENTAL_RESOURCES_APK),
              ruleContext,
              resourceDeps,
              false,
              ProguardHelper.getProguardConfigArtifact(ruleContext, "incremental"));
      if (ruleContext.hasErrors()) {
        return null;
      }

      splitResourceApk = applicationManifest
          .createSplitManifest(ruleContext, "android_resources", false)
          .packWithResources(getDxArtifact(ruleContext, "android_resources.ap_"),
            ruleContext,
            resourceDeps,
            false,
            ProguardHelper.getProguardConfigArtifact(ruleContext, "incremental_split"));
      if (ruleContext.hasErrors()) {
        return null;
      }
    }

    JavaTargetAttributes resourceClasses = androidCommon.init(
        javaSemantics,
        androidSemantics,
        resourceApk,
        ruleContext.getConfiguration().isCodeCoverageEnabled(),
        true /* collectJavaCompilationArgs */);
    if (resourceClasses == null) {
      return null;
    }

    Artifact deployJar = createDeployJar(ruleContext, javaSemantics, androidCommon, resourceClasses,
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_DEPLOY_JAR));

    Artifact proguardMapping = ruleContext.getPrerequisiteArtifact(
        "proguard_apply_mapping", Mode.TARGET);


    return createAndroidBinary(
        ruleContext,
        filesBuilder,
        deployJar,
        javaCommon,
        androidCommon,
        javaSemantics,
        androidSemantics,
        nativeLibs,
        applicationManifest,
        resourceApk,
        incrementalResourceApk,
        splitResourceApk,
        resourceClasses,
        ImmutableList.<Artifact>of(),
        proguardMapping);
  }


  public static RuleConfiguredTargetBuilder createAndroidBinary(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact deployJar,
      JavaCommon javaCommon,
      AndroidCommon androidCommon,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics,
      NativeLibs nativeLibs,
      ApplicationManifest applicationManifest,
      ResourceApk resourceApk,
      ResourceApk incrementalResourceApk,
      ResourceApk splitResourceApk,
      JavaTargetAttributes resourceClasses,
      ImmutableList<Artifact> apksUnderTest,
      Artifact proguardMapping) throws InterruptedException {
    ImmutableList<Artifact> proguardSpecs = ProguardHelper.collectTransitiveProguardSpecs(
        ruleContext, ImmutableList.of(resourceApk.getResourceProguardConfig()));

    ProguardOutput proguardOutput =
        applyProguard(
            ruleContext,
            androidCommon,
            deployJar,
            filesBuilder,
            proguardSpecs,
            proguardMapping);
    Artifact jarToDex = proguardOutput.getOutputJar();
    DexingOutput dexingOutput =
        shouldDexWithJack(ruleContext)
            ? dexWithJack(ruleContext, androidCommon, proguardSpecs)
            : dex(
                ruleContext,
                getMultidexMode(ruleContext),
                ruleContext.getTokenizedStringListAttr("dexopts"),
                deployJar,
                jarToDex,
                androidCommon,
                resourceClasses);
    if (dexingOutput == null) {
      return null;
    }

    Artifact unsignedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_UNSIGNED_APK);
    Artifact signedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_SIGNED_APK);

    ApkActionBuilder apkBuilder = new ApkActionBuilder(ruleContext, androidSemantics)
        .classesDex(dexingOutput.classesDexZip)
        .resourceApk(resourceApk.getArtifact())
        .javaResourceZip(dexingOutput.javaResourceJar)
        .nativeLibs(nativeLibs);

    ruleContext.registerAction(apkBuilder
        .message("Generating unsigned apk")
        .build(unsignedApk));

    ruleContext.registerAction(apkBuilder
        .message("Generating signed apk")
        .sign(true)
        .build(signedApk));

    Artifact zipAlignedApk = zipalignApk(
        ruleContext,
        signedApk,
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_APK));

    // Don't add blacklistedApk, so it's only built if explicitly requested.
    filesBuilder.add(deployJar);
    filesBuilder.add(unsignedApk);
    filesBuilder.add(zipAlignedApk);

    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    NestedSet<Artifact> coverageMetadata = (androidCommon.getInstrumentedJar() != null)
        ? NestedSetBuilder.create(Order.STABLE_ORDER, androidCommon.getInstrumentedJar())
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);

    Artifact incrementalApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_INCREMENTAL_APK);

    Artifact fullDeployMarker =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.FULL_DEPLOY_MARKER);
    Artifact incrementalDeployMarker =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.INCREMENTAL_DEPLOY_MARKER);
    Artifact splitDeployMarker =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.SPLIT_DEPLOY_MARKER);

    Artifact incrementalDexManifest =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.DEX_MANIFEST);
    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("AndroidDexManifest")
        .setProgressMessage("Generating incremental installation manifest for "
            + ruleContext.getLabel())
        .setExecutable(
            ruleContext.getExecutablePrerequisite("$build_incremental_dexmanifest", Mode.HOST))
        .addOutputArgument(incrementalDexManifest)
        .addInputArguments(dexingOutput.shardDexZips)
        .useParameterFile(ParameterFileType.UNQUOTED).build(ruleContext));

    Artifact stubData = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.STUB_APPLICATION_DATA);
    Artifact stubDex = getStubDex(ruleContext, javaSemantics, false);
    if (ruleContext.hasErrors()) {
      return null;
    }

    ApkActionBuilder incrementalActionBuilder = new ApkActionBuilder(ruleContext, androidSemantics)
        .classesDex(stubDex)
        .resourceApk(incrementalResourceApk.getArtifact())
        .javaResourceZip(dexingOutput.javaResourceJar)
        .sign(true)
        .javaResourceFile(stubData)
        .message("Generating incremental apk");

    if (!ruleContext.getFragment(AndroidConfiguration.class).useIncrementalNativeLibs()) {
      incrementalActionBuilder.nativeLibs(nativeLibs);
    }

    ruleContext.registerAction(incrementalActionBuilder.build(incrementalApk));

    Artifact argsArtifact =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.MOBILE_INSTALL_ARGS);
    ruleContext.registerAction(
        new WriteAdbArgsAction(ruleContext.getActionOwner(), argsArtifact));

    createInstallAction(ruleContext, false, fullDeployMarker, argsArtifact,
        incrementalDexManifest, incrementalResourceApk.getArtifact(), incrementalApk, nativeLibs,
        stubData);

    createInstallAction(ruleContext, true, incrementalDeployMarker,
        argsArtifact,
        incrementalDexManifest,
        incrementalResourceApk.getArtifact(),
        incrementalApk,
        nativeLibs,
        stubData);

    NestedSetBuilder<Artifact> splitApkSetBuilder = NestedSetBuilder.compileOrder();

    // Put the Android resource APK first so that this split gets installed first.
    //
    // This avoids some logcat spam during installation, because otherwise the Android package
    // manager would complain about references to missing resources in the manifest during the
    // installation of each split (said references would eventually get installed, but it cannot
    // know that in advance)
    Artifact resourceSplitApk = getDxArtifact(ruleContext, "android_resources.apk");
    ruleContext.registerAction(new ApkActionBuilder(ruleContext, androidSemantics)
        .resourceApk(splitResourceApk.getArtifact())
        .sign(true)
        .message("Generating split Android resource apk")
        .build(resourceSplitApk));
    splitApkSetBuilder.add(resourceSplitApk);

    for (int i = 0; i < dexingOutput.shardDexZips.size(); i++) {
      String splitName = "dex" + (i + 1);
      Artifact splitApkResources = createSplitApkResources(
          ruleContext, applicationManifest, splitName, true);
      Artifact splitApk = getDxArtifact(ruleContext, splitName + ".apk");
      ruleContext.registerAction(new ApkActionBuilder(ruleContext, androidSemantics)
          .classesDex(dexingOutput.shardDexZips.get(i))
          .resourceApk(splitApkResources)
          .sign(true)
          .message("Generating split dex apk " + (i + 1))
          .build(splitApk));
      splitApkSetBuilder.add(splitApk);
    }

    Artifact nativeSplitApkResources = createSplitApkResources(
        ruleContext, applicationManifest, "native", false);
    Artifact nativeSplitApk = getDxArtifact(ruleContext, "native.apk");
    ruleContext.registerAction(new ApkActionBuilder(ruleContext, androidSemantics)
        .resourceApk(nativeSplitApkResources)
        .sign(true)
        .message("Generating split native apk")
        .nativeLibs(nativeLibs)
        .build(nativeSplitApk));
    splitApkSetBuilder.add(nativeSplitApk);

    Artifact javaSplitApkResources = createSplitApkResources(
        ruleContext, applicationManifest, "java_resources", false);
    Artifact javaSplitApk = getDxArtifact(ruleContext, "java_resources.apk");
    ruleContext.registerAction(new ApkActionBuilder(ruleContext, androidSemantics)
        .resourceApk(javaSplitApkResources)
        .javaResourceZip(dexingOutput.javaResourceJar)
        .sign(true)
        .message("Generating split Java resource apk")
        .build(javaSplitApk));
    splitApkSetBuilder.add(javaSplitApk);

    Artifact splitMainApkResources = getDxArtifact(ruleContext, "split_main.ap_");
    ruleContext.registerAction(new SpawnAction.Builder()
        .setMnemonic("AndroidStripResources")
        .setProgressMessage("Stripping resources from split main apk")
        .setExecutable(ruleContext.getExecutablePrerequisite("$strip_resources", Mode.HOST))
        .addArgument("--input_resource_apk")
        .addInputArgument(resourceApk.getArtifact())
        .addArgument("--output_resource_apk")
        .addOutputArgument(splitMainApkResources)
        .build(ruleContext));

    NestedSet<Artifact> splitApks = splitApkSetBuilder.build();
    Artifact splitMainApk = getDxArtifact(ruleContext, "split_main.apk");
    Artifact splitStubDex = getStubDex(ruleContext, javaSemantics, true);
    if (ruleContext.hasErrors()) {
      return null;
    }
    ruleContext.registerAction(new ApkActionBuilder(ruleContext, androidSemantics)
        .resourceApk(splitMainApkResources)
        .classesDex(splitStubDex)
        .sign(true)
        .message("Generating split main apk")
        .build(splitMainApk));
    splitApkSetBuilder.add(splitMainApk);
    NestedSet<Artifact> allSplitApks = splitApkSetBuilder.build();

    createSplitInstallAction(ruleContext, splitDeployMarker, argsArtifact, splitMainApk,
        splitApks, stubData);

    NestedSet<Artifact> splitOutputGroup = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(allSplitApks)
        .add(splitDeployMarker)
        .build();

    androidCommon.addTransitiveInfoProviders(
        builder, androidSemantics, resourceApk, zipAlignedApk, apksUnderTest);
    androidSemantics.addTransitiveInfoProviders(
        builder, ruleContext, javaCommon, androidCommon, jarToDex);

    if (proguardOutput.getMapping() != null) {
      builder.add(ProguardMappingProvider.class,
          new ProguardMappingProvider(proguardOutput.getMapping()));
    }

    return builder
        .setFilesToBuild(filesToBuild)
        .add(
            RunfilesProvider.class,
            RunfilesProvider.simple(
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
                    .addTransitiveArtifacts(filesToBuild)
                    .build()))
        .add(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(resourceClasses, javaSemantics))
        .add(
            ApkProvider.class,
            new ApkProvider(
                NestedSetBuilder.create(Order.STABLE_ORDER, zipAlignedApk), coverageMetadata))
        .add(AndroidPreDexJarProvider.class, new AndroidPreDexJarProvider(jarToDex))
        .addOutputGroup("mobile_install_full", fullDeployMarker)
        .addOutputGroup("mobile_install_incremental", incrementalDeployMarker)
        .addOutputGroup("mobile_install_split", splitOutputGroup);
  }

  private static void createSplitInstallAction(RuleContext ruleContext,
      Artifact marker, Artifact argsArtifact, Artifact splitMainApk, NestedSet<Artifact> splitApks,
      Artifact stubDataFile) {
    FilesToRunProvider adb = AndroidSdkProvider.fromRuleContext(ruleContext).getAdb();
    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$incremental_install", Mode.HOST))
        .addTool(adb)
        .executeUnconditionally()
        .setMnemonic("AndroidInstall")
        .setProgressMessage("Installing " + ruleContext.getLabel() + " using split apks")
        .setExecutionInfo(ImmutableMap.of("local", ""))
        .addArgument("--output_marker")
        .addOutputArgument(marker)
        .addArgument("--stub_datafile")
        .addInputArgument(stubDataFile)
        .addArgument("--adb")
        .addArgument(adb.getExecutable().getExecPathString())
        .addTool(adb)
        .addArgument("--flagfile")
        .addInputArgument(argsArtifact)
        .addArgument("--split_main_apk")
        .addInputArgument(splitMainApk);

    for (Artifact splitApk : splitApks) {
      builder
          .addArgument("--split_apk")
          .addInputArgument(splitApk);
    }

    ruleContext.registerAction(builder.build(ruleContext));
  }

  private static void createInstallAction(RuleContext ruleContext,
      boolean incremental, Artifact marker, Artifact argsArtifact,
      Artifact dexmanifest, Artifact resourceApk, Artifact apk, NativeLibs nativeLibs,
      Artifact stubDataFile) {
    FilesToRunProvider adb = AndroidSdkProvider.fromRuleContext(ruleContext).getAdb();
    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$incremental_install", Mode.HOST))
        // We cannot know if the user connected a new device, uninstalled the app from the device
        // or did anything strange to it, so we always run this action.
        .executeUnconditionally()
        .setMnemonic("AndroidInstall")
        .setProgressMessage(
            "Installing " + ruleContext.getLabel() + (incremental ? " incrementally" : ""))
        .setExecutionInfo(ImmutableMap.of("local", ""))
        .addArgument("--output_marker")
        .addOutputArgument(marker)
        .addArgument("--dexmanifest")
        .addInputArgument(dexmanifest)
        .addArgument("--resource_apk")
        .addInputArgument(resourceApk)
        .addArgument("--stub_datafile")
        .addInputArgument(stubDataFile)
        .addArgument("--adb")
        .addArgument(adb.getExecutable().getExecPathString())
        .addTool(adb)
        .addArgument("--flagfile")
        .addInputArgument(argsArtifact);

    if (!incremental) {
      builder
          .addArgument("--apk")
          .addInputArgument(apk);
    }

    if (ruleContext.getFragment(AndroidConfiguration.class).useIncrementalNativeLibs()) {
      for (Map.Entry<String, Iterable<Artifact>> arch : nativeLibs.getMap().entrySet()) {
        for (Artifact lib : arch.getValue()) {
          builder
              .addArgument("--native_lib")
              .addArgument(arch.getKey() + ":" + lib.getExecPathString())
              .addInput(lib);
        }
      }
    }

    ruleContext.registerAction(builder.build(ruleContext));
  }

  private static Artifact getStubDex(
      RuleContext ruleContext, JavaSemantics javaSemantics, boolean split) {
    String attribute =
        split ? "$incremental_split_stub_application" : "$incremental_stub_application";

    TransitiveInfoCollection dep = ruleContext.getPrerequisite(attribute, Mode.TARGET);
    if (dep == null) {
      ruleContext.attributeError(attribute, "Stub application cannot be found");
      return null;
    }

    JavaCompilationArgsProvider provider = dep.getProvider(JavaCompilationArgsProvider.class);
    if (provider == null) {
      ruleContext.attributeError(attribute, "'" + dep.getLabel() + "' should be a Java target");
      return null;
    }

    JavaTargetAttributes attributes = new JavaTargetAttributes.Builder(javaSemantics)
        .addRuntimeClassPathEntries(provider.getJavaCompilationArgs().getRuntimeJars())
        .build();

    Artifact stubDeployJar = getDxArtifact(ruleContext,
        split ? "split_stub_deploy.jar" : "stub_deploy.jar");
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(stubDeployJar)
        .setAttributes(attributes)
        .build();

    Artifact stubDex = getDxArtifact(ruleContext,
        split ? "split_stub_application.dex" : "stub_application.dex");
    AndroidCommon.createDexAction(
        ruleContext,
        stubDeployJar,
        stubDex,
        ImmutableList.<String>of(),
        false,
        null);

    return stubDex;
  }

  /** Generates an uncompressed _deploy.jar of all the runtime jars. */
  public static Artifact createDeployJar(
      RuleContext ruleContext, JavaSemantics javaSemantics, AndroidCommon common,
      JavaTargetAttributes attributes, Artifact deployJar) {
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(deployJar)
        .setAttributes(attributes)
        .addRuntimeJars(common.getRuntimeJars())
        .build();
    return deployJar;
  }

  private static String getExpandedMakeVarsForAttr(RuleContext context, String attr) {
    final String value = context.attributes().get(attr, Type.STRING);
    if (isNullOrEmpty(value)) {
      return null;
    }
    return context.expandMakeVariables(attr, value);
  }

  private static JavaOptimizationMode getJavaOptimizationMode(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(JavaConfiguration.class)
        .getJavaOptimizationMode();
  }

  /**
   * Applies the proguard specifications, and creates a ProguardedJar. Proguard's output artifacts
   * are added to the given {@code filesBuilder}.
   */
  private static ProguardOutput applyProguard(
      RuleContext ruleContext,
      AndroidCommon common,
      Artifact deployJarArtifact,
      NestedSetBuilder<Artifact> filesBuilder,
      ImmutableList<Artifact> proguardSpecs,
      Artifact proguardMapping) throws InterruptedException {
    Artifact proguardOutputJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_PROGUARD_JAR);

    // Proguard will be only used for binaries which specify a proguard_spec
    if (proguardSpecs.isEmpty()) {
      // Although normally the Proguard jar artifact is not needed for binaries which do not specify
      // proguard_specs, targets which use a select to provide an empty list to proguard_specs will
      // still have a Proguard jar implicit output, as it is impossible to tell what a select will
      // produce at the time of implicit output determination. As a result, this artifact must
      // always be created.
      return createEmptyProguardAction(ruleContext, proguardOutputJar, deployJarArtifact);
    }

    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    NestedSet<Artifact> libraryJars = NestedSetBuilder.<Artifact>naiveLinkOrder()
        .add(sdk.getAndroidJar())
        .addTransitive(common.getTransitiveNeverLinkLibraries())
        .build();
    ProguardOutput result = ProguardHelper.createProguardAction(ruleContext, sdk.getProguard(),
        deployJarArtifact, proguardSpecs, proguardMapping, libraryJars, proguardOutputJar,
        ruleContext.attributes().get("proguard_generate_mapping", Type.BOOLEAN));
    // Since Proguard is being run, add its output artifacts to the given filesBuilder
    result.addAllToSet(filesBuilder);
    return result;
  }

  private static ProguardOutput createEmptyProguardAction(RuleContext ruleContext,
      Artifact proguardOutputJar, Artifact deployJarArtifact) throws InterruptedException {
    ImmutableList.Builder<Artifact> failures =
        ImmutableList.<Artifact>builder().add(proguardOutputJar);
    if (ruleContext.attributes().get("proguard_generate_mapping", Type.BOOLEAN)) {
      failures.add(ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP));
    }
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    ruleContext.registerAction(
        new FailAction(
            ruleContext.getActionOwner(),
            failures.build(),
            String.format("Can't generate Proguard jar or mapping %s.",
                optMode == JavaOptimizationMode.LEGACY
                    ? "without proguard_specs"
                    : "in optimization mode " + optMode)));
    return new ProguardOutput(deployJarArtifact, null);
  }

  @Immutable
  private static final class DexingOutput {
    private final Artifact classesDexZip;
    private final Artifact javaResourceJar;
    private final ImmutableList<Artifact> shardDexZips;

    private DexingOutput(
        Artifact classesDexZip, Artifact javaResourceJar, Iterable<Artifact> shardDexZips) {
      this.classesDexZip = classesDexZip;
      this.javaResourceJar = javaResourceJar;
      this.shardDexZips = ImmutableList.copyOf(shardDexZips);
    }
  }

  static boolean shouldDexWithJack(RuleContext ruleContext) {
    return ruleContext
        .getFragment(AndroidConfiguration.class)
        .isJackUsedForDexing();
  }

  static DexingOutput dexWithJack(
      RuleContext ruleContext, AndroidCommon androidCommon, ImmutableList<Artifact> proguardSpecs) {
    Artifact classesDexZip =
        androidCommon.compileDexWithJack(
            getMultidexMode(ruleContext),
            Optional.fromNullable(
                ruleContext.getPrerequisiteArtifact("main_dex_list", Mode.TARGET)),
            proguardSpecs);
    return new DexingOutput(classesDexZip, null, ImmutableList.of(classesDexZip));
  }

  /** Dexes the ProguardedJar to generate ClassesDex that has a reference classes.dex. */
  private static DexingOutput dex(RuleContext ruleContext, MultidexMode multidexMode,
      List<String> dexopts, Artifact deployJar,  Artifact proguardedJar, AndroidCommon common,
      JavaTargetAttributes attributes) throws InterruptedException {
    String classesDexFileName = getMultidexMode(ruleContext).getOutputDexFilename();
    Artifact classesDex = AndroidBinary.getDxArtifact(ruleContext, classesDexFileName);
    if (!AndroidBinary.supportsMultidexMode(ruleContext, multidexMode)) {
      ruleContext.ruleError("Multidex mode \"" + multidexMode.getAttributeValue()
          + "\" not supported by this version of the Android SDK");
      return null;
    }

    int dexShards = ruleContext.attributes().get("dex_shards", Type.INTEGER);
    if (dexShards > 1 && multidexMode == MultidexMode.OFF) {
      ruleContext.ruleError(".dex sharding is only available in multidex mode");
      return null;
    }

    Artifact mainDexList = ruleContext.getPrerequisiteArtifact("main_dex_list", Mode.TARGET);
    if ((mainDexList != null && multidexMode != MultidexMode.MANUAL_MAIN_DEX)
        || (mainDexList == null && multidexMode == MultidexMode.MANUAL_MAIN_DEX)) {
      ruleContext.ruleError(
          "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified.");
      return null;
    }

    if (multidexMode == MultidexMode.OFF) {
      // Single dex mode: generate classes.dex directly from the input jar.
      AndroidCommon.createDexAction(
          ruleContext, proguardedJar, classesDex, dexopts, false, null);
      return new DexingOutput(classesDex, deployJar, ImmutableList.of(classesDex));
    } else {
      // Multidex mode: generate classes.dex.zip, where the zip contains [classes.dex,
      // classes2.dex, ... classesN.dex]. Because the dexer also places resources into this zip,
      // we also need to create a cleanup action that removes all non-.dex files before staging
      // for apk building.
      if (multidexMode == MultidexMode.LEGACY) {
        // For legacy multidex, we need to generate a list for the dexer's --main-dex-list flag.
        mainDexList = createMainDexListAction(ruleContext, proguardedJar);
      }

      if (dexShards > 1) {
        List<Artifact> shardJars = new ArrayList<>(dexShards);
        for (int i = 1; i <= dexShards; i++) {
          shardJars.add(getDxArtifact(ruleContext, "shard" + i + ".jar"));
        }

        CustomCommandLine.Builder shardCommandLine = CustomCommandLine.builder()
            .addBeforeEachExecPath("--output_jar", shardJars);

        Artifact javaResourceJar =
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.JAVA_RESOURCES_JAR);

        if (mainDexList != null) {
          shardCommandLine.addExecPath("--main_dex_filter", mainDexList);
        }

        // If we need to run Proguard, all the class files will be in the Proguarded jar and the
        // deploy jar will already have been built (since it's the input of Proguard) and it will
        // contain all the Java resources. Otherwise, we don't want to have deploy jar creation on
        // the critical path, so we put all the jar files that constitute it on the inputs of the
        // jar shuffler.
        if (proguardedJar != deployJar) {
          shardCommandLine.addExecPath("--input_jar", proguardedJar);
        } else {
          shardCommandLine
              .addBeforeEachExecPath("--input_jar", common.getRuntimeJars())
              .addBeforeEachExecPath("--input_jar", attributes.getRuntimeClassPathForArchive());
        }

        shardCommandLine.addExecPath("--output_resources", javaResourceJar);

        SpawnAction.Builder shardAction = new SpawnAction.Builder()
            .setMnemonic("ShardClassesToDex")
            .setProgressMessage("Sharding classes for dexing for " + ruleContext.getLabel())
            .setExecutable(ruleContext.getExecutablePrerequisite("$shuffle_jars", Mode.HOST))
            .addOutputs(shardJars)
            .addOutput(javaResourceJar)
            .setCommandLine(shardCommandLine.build());

        if (mainDexList != null) {
          shardAction.addInput(mainDexList);
        }
        if (proguardedJar != deployJar) {
          shardAction.addInput(proguardedJar);
        } else {
          shardAction
              .addInputs(common.getRuntimeJars())
              .addInputs(attributes.getRuntimeClassPathForArchive());
        }

        ruleContext.registerAction(shardAction.build(ruleContext));

        List<Artifact> shardDexes = new ArrayList<>(dexShards);
        for (int i = 1; i <= dexShards; i++) {
          Artifact shardJar = shardJars.get(i - 1);
          Artifact shard = getDxArtifact(ruleContext, "shard" + i + ".dex.zip");
          shardDexes.add(shard);
          AndroidCommon.createDexAction(
              ruleContext, shardJar, shard, dexopts, true, null);
        }

        CommandLine mergeCommandLine = CustomCommandLine.builder()
            .addBeforeEachExecPath("--input_zip", shardDexes)
            .addExecPath("--output_zip", classesDex)
            .build();
        ruleContext.registerAction(new SpawnAction.Builder()
            .setMnemonic("MergeDexZips")
            .setProgressMessage("Merging dex shards for " + ruleContext.getLabel())
            .setExecutable(ruleContext.getExecutablePrerequisite("$merge_dexzips", Mode.HOST))
            .addInputs(shardDexes)
            .addOutput(classesDex)
            .setCommandLine(mergeCommandLine)
            .build(ruleContext));
        return new DexingOutput(classesDex, javaResourceJar, shardDexes);
      } else {
        // Create an artifact for the intermediate zip output that includes non-.dex files.
        Artifact classesDexIntermediate = AndroidBinary.getDxArtifact(
            ruleContext,
            "intermediate_" + classesDexFileName);

        // Have the dexer generate the intermediate file and the "cleaner" action consume this to
        // generate the final archive with only .dex files.
        AndroidCommon.createDexAction(ruleContext, proguardedJar,
            classesDexIntermediate, dexopts, true, mainDexList);
        createCleanDexZipAction(ruleContext, classesDexIntermediate, classesDex);
        return new DexingOutput(classesDex, deployJar, ImmutableList.of(classesDex));
      }
    }
  }

  /**
   * Creates an action that copies a .zip file to a specified path, filtering all non-.dex files
   * out of the output.
   */
  static void createCleanDexZipAction(RuleContext ruleContext, Artifact inputZip,
      Artifact outputZip) {
    ruleContext.registerAction(new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$zip", Mode.HOST))
        .addInput(inputZip)
        .addOutput(outputZip)
        .addArgument(inputZip.getExecPathString())
        .addArgument("--out")
        .addArgument(outputZip.getExecPathString())
        .addArgument("--copy")
        .addArgument("classes*.dex")
        .setProgressMessage("Trimming " + inputZip.getExecPath().getBaseName())
        .setMnemonic("TrimDexZip")
        .build(ruleContext));
  }

  /**
   * Creates an action that generates a list of classes to be passed to the dexer's
   * --main-dex-list flag (which specifies the classes that need to be directly in classes.dex).
   * Returns the file containing the list.
   */
  static Artifact createMainDexListAction(RuleContext ruleContext, Artifact jar) {
    // Process the input jar through Proguard into an intermediate, streamlined jar.
    Artifact strippedJar = AndroidBinary.getDxArtifact(ruleContext, "main_dex_intermediate.jar");
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    SpawnAction.Builder streamlinedBuilder = new SpawnAction.Builder()
        .addOutput(strippedJar)
        .setExecutable(sdk.getProguard())
        .setProgressMessage("Generating streamlined input jar for main dex classes list")
        .setMnemonic("MainDexClassesIntermediate")
        .addArgument("-injars")
        .addInputArgument(jar)
        .addArgument("-libraryjars")
        .addInputArgument(sdk.getShrinkedAndroidJar())
        .addArgument("-outjars")
        .addArgument(strippedJar.getExecPathString())
        .addArgument("-dontwarn")
        .addArgument("-dontnote")
        .addArgument("-forceprocessing")
        .addArgument("-dontoptimize")
        .addArgument("-dontobfuscate")
        .addArgument("-dontpreverify");

    List<Artifact> specs = ruleContext.getPrerequisiteArtifacts(
        "main_dex_proguard_specs", Mode.TARGET).list();
    if (specs.isEmpty()) {
      specs = ImmutableList.of(sdk.getMainDexClasses());
    }

    for (Artifact spec : specs) {
      streamlinedBuilder.addArgument("-include");
      streamlinedBuilder.addInputArgument(spec);
    }

    ruleContext.registerAction(streamlinedBuilder.build(ruleContext));

    // Create the main dex classes list.
    Artifact mainDexList = AndroidBinary.getDxArtifact(ruleContext, "main_dex_list.txt");
    Builder builder = new Builder()
        .setMnemonic("MainDexClasses")
        .setProgressMessage("Generating main dex classes list");

    ruleContext.registerAction(builder
        .setExecutable(sdk.getMainDexListCreator())
        .addOutputArgument(mainDexList)
        .addInputArgument(strippedJar)
        .addInputArgument(jar)
        .addArguments(ruleContext.getTokenizedStringListAttr("main_dex_list_opts"))
        .build(ruleContext));
    return mainDexList;
  }

  private static Artifact createSplitApkResources(RuleContext ruleContext,
      ApplicationManifest mainManifest, String splitName, boolean hasCode) {
    Artifact splitManifest = mainManifest.createSplitManifest(ruleContext, splitName, hasCode)
        .getManifest();
    Artifact splitResources = getDxArtifact(ruleContext, "split_" + splitName + ".ap_");
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    ruleContext.registerAction(new SpawnAction.Builder()
        .setExecutable(sdk.getAapt())
        .setMnemonic("AndroidAapt")
        .setProgressMessage("Generating resource apk for split " + splitName)
        .addArgument("package")
        .addArgument("-F")
        .addOutputArgument(splitResources)
        .addArgument("-M")
        .addInputArgument(splitManifest)
        .addArgument("-I")
        .addInputArgument(sdk.getAndroidJar())
        .build(ruleContext));

    return splitResources;
  }

  /**
   * Builder class for {@link com.google.devtools.build.lib.analysis.actions.SpawnAction}s that
   * generate APKs.
   *
   * <p>Instances of this class can be reused after calling {@code build()}.
   */
  private static final class ApkActionBuilder {
    private final RuleContext ruleContext;
    private final AndroidSemantics semantics;

    private boolean sign;
    private String message;
    private Artifact classesDex;
    private Artifact resourceApk;
    private Artifact javaResourceZip;
    // javaResourceFile adds Java resources just like javaResourceZip. We should make the stub
    // manifest writer output a zip file, then we could do away with this input to APK building.
    private Artifact javaResourceFile;
    private NativeLibs nativeLibs = NativeLibs.EMPTY;

    private ApkActionBuilder(
        RuleContext ruleContext, AndroidSemantics semantics) {
      this.ruleContext = ruleContext;
      this.semantics = semantics;
    }

    /**
     * Sets the user-visible message that is displayed when the action is running.
     */
    public ApkActionBuilder message(String message) {
      this.message = message;
      return this;
    }

    /**
     * Sets the native libraries to be included in the APK.
     */
    public ApkActionBuilder nativeLibs(NativeLibs nativeLibs) {
      this.nativeLibs = nativeLibs;
      return this;
    }

    /**
     * Sets the dex file to be included in the APK.
     *
     * <p>Can be either a plain .dex or a .zip file containing dexes.
     */
    public ApkActionBuilder classesDex(Artifact classesDex) {
      this.classesDex = classesDex;
      return this;
    }

    /**
     * Sets the resource APK that contains the Android resources to be bundled into the output.
     */
    public ApkActionBuilder resourceApk(Artifact resourceApk) {
      this.resourceApk = resourceApk;
      return this;
    }

    /**
     * Sets the file where Java resources are taken.
     *
     * <p>Everything in this will will be put directly into the APK except files with the extension
     * {@code .class}.
     */
    public ApkActionBuilder javaResourceZip(Artifact javaResourcezip) {
      this.javaResourceZip = javaResourcezip;
      return this;
    }

    /**
     * Adds an individual resource file to the root directory of the APK.
     *
     * <p>This provides the same functionality as {@code javaResourceZip}, except much more hacky.
     * Will most probably won't work if there is an input artifact in the same directory as this
     * file.
     */
    public ApkActionBuilder javaResourceFile(Artifact javaResourceFile) {
      this.javaResourceFile = javaResourceFile;
      return this;
    }

    /**
     * Sets if the APK will be signed. By default, it won't be.
     */
    public ApkActionBuilder sign(boolean sign) {
      this.sign = sign;
      return this;
    }

    /**
     * Creates a generating action for {@code outApk} that builds the APK specified.
     */
    public Action[] build(Artifact outApk) {
      Builder actionBuilder = new SpawnAction.Builder()
          .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkBuilder())
          .setProgressMessage(message)
          .setMnemonic("AndroidApkBuilder")
          .addOutputArgument(outApk);

      if (javaResourceZip != null) {
        actionBuilder
            .addArgument("-rj")
            .addInputArgument(javaResourceZip);
      }

      Artifact nativeSymlinks = nativeLibs.createApkBuilderSymlinks(ruleContext);
      if (nativeSymlinks != null) {
        PathFragment nativeSymlinksDir = nativeSymlinks.getExecPath().getParentDirectory();
        actionBuilder
            .addInputManifest(nativeSymlinks, nativeSymlinksDir)
            .addInput(nativeSymlinks)
            .addInputs(nativeLibs.getAllNativeLibs())
            .addArgument("-nf")
            // If the native libs are "foo/bar/x86/foo.so", we need to pass "foo/bar" here
            .addArgument(nativeSymlinksDir.getPathString());
      }

      if (nativeLibs.getName() != null) {
        actionBuilder
            .addArgument("-rf")
            .addArgument(nativeLibs.getName().getExecPath().getParentDirectory().getPathString())
            .addInput(nativeLibs.getName());
      }

      if (javaResourceFile != null) {
        actionBuilder
            .addArgument("-rf")
            .addArgument((javaResourceFile.getExecPath().getParentDirectory().getPathString()))
            .addInput(javaResourceFile);
      }

      if (sign) {
        Artifact signingKey = semantics.getApkDebugSigningKey(ruleContext);
        actionBuilder.addArgument("-ks").addArgument(signingKey.getExecPathString());
        actionBuilder.addInput(signingKey);
      } else {
        actionBuilder.addArgument("-u");
      }

      actionBuilder
          .addArgument("-z")
          .addInputArgument(resourceApk);

      if (classesDex != null) {
        actionBuilder
            .addArgument(classesDex.getFilename().endsWith(".dex") ? "-f" : "-z")
            .addInputArgument(classesDex);
      }

      return actionBuilder.build(ruleContext);
    }
  }

  /** Last step in buildings an apk: align the zip boundaries by 4 bytes. */
  static Artifact zipalignApk(RuleContext ruleContext,
      Artifact signedApk, Artifact zipAlignedApk) {
    List<String> args = new ArrayList<>();
    // "4" is the only valid value for zipalign, according to:
    // http://developer.android.com/guide/developing/tools/zipalign.html
    args.add("4");
    args.add(signedApk.getExecPathString());
    args.add(zipAlignedApk.getExecPathString());

    ruleContext.registerAction(new SpawnAction.Builder()
        .addInput(signedApk)
        .addOutput(zipAlignedApk)
        .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
        .addArguments(args)
        .setProgressMessage("Zipaligning apk")
        .setMnemonic("AndroidZipAlign")
        .build(ruleContext));
    args.add(signedApk.getExecPathString());
    args.add(zipAlignedApk.getExecPathString());
    return zipAlignedApk;
  }

  /**
   * Tests if the resources need to be regenerated.
   *
   * <p>The resources should be regenerated (using aapt) if any of the following are true:
   * <ul>
   *    <li>There is more than one resource container
   *    <li>There are densities to filter by.
   *    <li>There are resource configuration filters.
   *    <li>There are extensions that should be compressed.
   * </ul>
   */
  public static boolean shouldRegenerate(RuleContext ruleContext,
      ResourceDependencies resourceDeps) {
    return Iterables.size(resourceDeps.getResources()) > 1
        || ruleContext.attributes().isAttributeValueExplicitlySpecified("densities")
        || ruleContext.attributes().isAttributeValueExplicitlySpecified(
            "resource_configuration_filters")
        || ruleContext.attributes().isAttributeValueExplicitlySpecified("nocompress_extensions");
  }

  /**
   * Returns whether to use NativeDepsHelper to link native dependencies.
   */
  public static boolean shouldLinkNativeDeps(RuleContext ruleContext) {
    TriState attributeValue = ruleContext.attributes().get(
        "legacy_native_support", BuildType.TRISTATE);
    if (attributeValue == TriState.AUTO) {
      return !ruleContext.getFragment(AndroidConfiguration.class).getLegacyNativeSupport();
    } else {
      return attributeValue == TriState.NO;
    }
  }

  /**
   * Returns the multidex mode to apply to this target.
   */
  public static MultidexMode getMultidexMode(RuleContext ruleContext) {
    if (ruleContext.getRule().isAttrDefined("multidex", Type.STRING)) {
      return Preconditions.checkNotNull(
          MultidexMode.fromValue(ruleContext.attributes().get("multidex", Type.STRING)));
    } else {
      return MultidexMode.OFF;
    }
  }

  /**
   * List of Android SDKs that contain runtimes that do not support the native multidexing
   * introduced in Android L. If someone tries to build an android_binary that has multidex=native
   * set with an old SDK, we will exit with an error to alert the developer that his application
   * might not run on devices that the used SDK still supports.
   */
  private static final Set<String> RUNTIMES_THAT_DONT_SUPPORT_NATIVE_MULTIDEXING = ImmutableSet.of(
      "/android_sdk_linux/platforms/android_10/", "/android_sdk_linux/platforms/android_13/",
      "/android_sdk_linux/platforms/android_15/", "/android_sdk_linux/platforms/android_16/",
      "/android_sdk_linux/platforms/android_17/", "/android_sdk_linux/platforms/android_18/",
      "/android_sdk_linux/platforms/android_19/", "/android_sdk_linux/platforms/android_20/");

  /**
   * Returns true if the runtime contained in the Android SDK used to build this rule supports the
   * given version of multidex mode specified, false otherwise.
   */
  public static boolean supportsMultidexMode(RuleContext ruleContext, MultidexMode mode) {
    if (mode == MultidexMode.NATIVE) {
      // Native mode is not supported by Android devices running Android before v21.
      String runtime =
          AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar().getExecPathString();
      for (String blacklistedRuntime : RUNTIMES_THAT_DONT_SUPPORT_NATIVE_MULTIDEXING) {
        if (runtime.contains(blacklistedRuntime)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns an intermediate artifact used to support dex generation.
   */
  public static Artifact getDxArtifact(RuleContext ruleContext, String baseName) {
    return ruleContext.getUniqueDirectoryArtifact("_dx", baseName,
        ruleContext.getBinOrGenfilesDirectory());
  }
}
