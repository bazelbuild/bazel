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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.auto.value.AutoValue;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.AndroidBinaryMobileInstall.MobileInstallResourceApks;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.android.ProguardHelper.ProguardOutput;
import com.google.devtools.build.lib.rules.android.ZipFilterBuilder.CheckHashMismatchMode;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.OneVersionCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** An implementation for the "android_binary" rule. */
public abstract class AndroidBinary implements RuleConfiguredTargetFactory {

  private static final String DX_MINIMAL_MAIN_DEX_OPTION = "--minimal-main-dex";

  protected abstract JavaSemantics createJavaSemantics();

  protected abstract AndroidSemantics createAndroidSemantics();

  protected abstract CppSemantics createCppSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CppSemantics cppSemantics = createCppSemantics();
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    androidSemantics.checkForMigrationTag(ruleContext);
    androidSemantics.validateAndroidBinaryRuleContext(ruleContext);
    AndroidSdkProvider.verifyPresence(ruleContext);

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    RuleConfiguredTargetBuilder builder =
        init(ruleContext, filesBuilder, cppSemantics, javaSemantics, androidSemantics);
    return builder.build();
  }

  /** Checks expected rule invariants, throws rule errors if anything is set wrong. */
  private static void validateRuleContext(RuleContext ruleContext, AndroidDataContext dataContext)
      throws RuleErrorException {
    if (getMultidexMode(ruleContext) != MultidexMode.LEGACY
        && ruleContext
            .attributes()
            .isAttributeValueExplicitlySpecified("main_dex_proguard_specs")) {
      ruleContext.throwWithAttributeError(
          "main_dex_proguard_specs",
          "The 'main_dex_proguard_specs' attribute is only allowed if 'multidex' is"
              + " set to 'legacy'");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("proguard_apply_mapping")) {
      if (dataContext.throwOnProguardApplyMapping()) {
        ruleContext.throwWithAttributeError(
            "proguard_apply_mapping", "This attribute is not supported");
      }
      if (ruleContext
          .attributes()
          .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
          .isEmpty()) {
        ruleContext.throwWithAttributeError(
            "proguard_apply_mapping",
            "'proguard_apply_mapping' can only be used when 'proguard_specs' is also set");
      }
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("proguard_apply_dictionary")) {
      if (dataContext.throwOnProguardApplyDictionary()) {
        ruleContext.throwWithAttributeError(
            "proguard_apply_dictionary", "This attribute is not supported");
      }
      if (ruleContext
          .attributes()
          .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
          .isEmpty()) {
        ruleContext.throwWithAttributeError(
            "proguard_apply_dictionary",
            "'proguard_apply_dictionary' can only be used when 'proguard_specs' is also set");
      }
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("shrink_resources")
        && dataContext.throwOnShrinkResources()) {
      ruleContext.throwWithAttributeError("shrink_resources", "This attribute is not supported");
    }

    if (AndroidCommon.getAndroidConfig(ruleContext).desugarJava8Libs()
        && getMultidexMode(ruleContext) == MultidexMode.OFF) {
      // Multidex is required so we can include legacy libs as a separate .dex file.
      ruleContext.throwWithAttributeError(
          "multidex", "Support for Java 8 libraries on legacy devices requires multidex");
    }

    if (ruleContext.getFragment(JavaConfiguration.class).enforceProguardFileExtension()
        && ruleContext.attributes().has(ProguardHelper.PROGUARD_SPECS)) {
      List<PathFragment> pathsWithUnexpectedExtension =
          ruleContext
              .getPrerequisiteArtifacts(ProguardHelper.PROGUARD_SPECS, TransitionMode.TARGET)
              .list()
              .stream()
              .filter(Artifact::isSourceArtifact)
              .map(Artifact::getRootRelativePath)
              .filter(
                  // This checks the filename directly instead of using FileType because we want to
                  // exclude third_party/, but FileType is generally only given the basename.
                  //
                  // See e.g. RuleContext#validateDirectPrerequisiteType and
                  // PrerequisiteArtifacts#filter.
                  path ->
                      !path.getFileExtension().equals("pgcfg")
                          && !path.startsWith(RuleClass.THIRD_PARTY_PREFIX)
                          && !path.startsWith(RuleClass.EXPERIMENTAL_PREFIX))
              .collect(toImmutableList());
      if (!pathsWithUnexpectedExtension.isEmpty()) {
        ruleContext.throwWithAttributeError(
            ProguardHelper.PROGUARD_SPECS,
            "Proguard spec files must use the .pgcfg extension. These files do not end in .pgcfg: "
                + pathsWithUnexpectedExtension);
      }
    }
  }

  private static RuleConfiguredTargetBuilder init(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      CppSemantics cppSemantics,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics)
      throws InterruptedException, RuleErrorException {

    ResourceDependencies resourceDeps =
        ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink= */ false);

    AndroidDataContext dataContext = androidSemantics.makeContextForNative(ruleContext);
    validateRuleContext(ruleContext, dataContext);

    NativeLibs nativeLibs =
        NativeLibs.fromLinkedNativeDeps(
            ruleContext,
            ImmutableList.of("deps"),
            androidSemantics.getNativeDepsFileName(),
            cppSemantics);

    // Retrieve and compile the resources defined on the android_binary rule.
    AndroidResources.validateRuleContext(ruleContext);

    Map<String, String> manifestValues = StampedAndroidManifest.getManifestValues(ruleContext);

    StampedAndroidManifest manifest;
    if (isInstrumentation(ruleContext)
        && dataContext.getAndroidConfig().disableInstrumentationManifestMerging()) {
      manifest =
          AndroidManifest.fromAttributes(ruleContext, dataContext, androidSemantics)
              .stamp(dataContext);
    } else {
      manifest =
          AndroidManifest.fromAttributes(ruleContext, dataContext, androidSemantics)
              .mergeWithDeps(
                  dataContext,
                  androidSemantics,
                  ruleContext,
                  resourceDeps,
                  manifestValues,
                  ruleContext.getRule().isAttrDefined("manifest_merger", STRING)
                      ? ruleContext.attributes().get("manifest_merger", STRING)
                      : null);
    }

    boolean shrinkResourceCycles =
        shouldShrinkResourceCycles(
            dataContext.getAndroidConfig(), ruleContext, dataContext.isResourceShrinkingEnabled());
    ProcessedAndroidData processedAndroidData =
        ProcessedAndroidData.processBinaryDataFrom(
            dataContext,
            ruleContext,
            manifest,
            /* conditionalKeepRules= */ shrinkResourceCycles,
            manifestValues,
            AndroidResources.from(ruleContext, "resource_files"),
            AndroidAssets.from(ruleContext),
            resourceDeps,
            AssetDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
            ResourceFilterFactory.fromRuleContextAndAttrs(ruleContext),
            ruleContext.getExpander().withDataLocations().tokenized("nocompress_extensions"),
            ruleContext.attributes().get("crunch_png", Type.BOOLEAN),
            DataBinding.contextFrom(ruleContext, dataContext.getAndroidConfig()));

    AndroidApplicationResourceInfo androidApplicationResourceInfo =
        ruleContext.getPrerequisite(
            "application_resources",
            TransitionMode.TARGET,
            AndroidApplicationResourceInfo.PROVIDER);

    final ResourceApk resourceApk;
    if (androidApplicationResourceInfo == null) {
      resourceApk =
          new RClassGeneratorActionBuilder()
              .withDependencies(resourceDeps)
              .finalFields(!shrinkResourceCycles)
              .setClassJarOut(
                  dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR))
              .build(dataContext, processedAndroidData);
    } else {
      resourceApk =
          ResourceApk.fromAndroidApplicationResourceInfo(
              dataContext, androidApplicationResourceInfo);
    }

    if (dataContext.useResourcePathShortening()) {
      filesBuilder.add(
          ruleContext.getImplicitOutputArtifact(
              AndroidRuleClasses.ANDROID_RESOURCE_PATH_SHORTENING_MAP));
    }

    ruleContext.assertNoErrors();

    JavaCommon javaCommon =
        AndroidCommon.createJavaCommonWithAndroidDataBinding(
            ruleContext, javaSemantics, resourceApk.asDataBindingContext(), /* isLibrary */ false);
    javaSemantics.checkRule(ruleContext, javaCommon);
    javaSemantics.checkForProtoLibraryAndJavaProtoLibraryOnSameProto(ruleContext, javaCommon);

    AndroidCommon androidCommon = new AndroidCommon(javaCommon, /* asNeverLink= */ true);

    // Remove the library resource JARs from the binary's runtime classpath.
    // Resource classes from android_library dependencies are replaced by the binary's resource
    // class. We remove them only at the top level so that resources included by a library that is
    // a dependency of a java_library are still included, since these resources are propagated via
    // android-specific providers and won't show up when we collect the library resource JARs.
    // TODO(b/69552500): Instead, handle this properly so R JARs aren't put on the classpath for
    // both binaries and libraries.
    NestedSet<Artifact> excludedRuntimeArtifacts = getLibraryResourceJars(ruleContext);

    JavaTargetAttributes resourceClasses =
        androidCommon.init(
            javaSemantics,
            androidSemantics,
            resourceApk,
            ruleContext.getConfiguration().isCodeCoverageEnabled(),
            /* collectJavaCompilationArgs= */ true,
            /* isBinary= */ true,
            excludedRuntimeArtifacts,
            /* generateExtensionRegistry= */ true);
    ruleContext.assertNoErrors();

    Function<Artifact, Artifact> derivedJarFunction =
        collectDesugaredJars(ruleContext, androidCommon, androidSemantics, resourceClasses);
    Artifact deployJar =
        createDeployJar(
            ruleContext,
            javaSemantics,
            androidCommon,
            resourceClasses,
            AndroidCommon.getAndroidConfig(ruleContext).checkDesugarDeps(),
            derivedJarFunction);

    boolean isBinaryJarFiltered = isInstrumentation(ruleContext);
    if (isBinaryJarFiltered) {
      deployJar = getFilteredDeployJar(ruleContext, deployJar);
    }

    OneVersionEnforcementLevel oneVersionEnforcementLevel =
        ruleContext.getFragment(JavaConfiguration.class).oneVersionEnforcementLevel();
    Artifact oneVersionOutputArtifact = null;
    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF) {
      NestedSet<Artifact> transitiveDependencies =
          NestedSetBuilder.<Artifact>stableOrder()
              .addAll(
                  Iterables.transform(
                      resourceClasses.getRuntimeClassPath().toList(), derivedJarFunction))
              .addAll(
                  Iterables.transform(
                      androidCommon.getJarsProducedForRuntime().toList(), derivedJarFunction))
              .build();

      oneVersionOutputArtifact =
          OneVersionCheckActionBuilder.newBuilder()
              .withEnforcementLevel(oneVersionEnforcementLevel)
              .outputArtifact(
                  ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_ONE_VERSION_ARTIFACT))
              .useToolchain(JavaToolchainProvider.from(ruleContext))
              .checkJars(transitiveDependencies)
              .build(ruleContext);
    }

    Artifact proguardMapping =
        ruleContext.getPrerequisiteArtifact("proguard_apply_mapping", TransitionMode.TARGET);

    MobileInstallResourceApks mobileInstallResourceApks =
        AndroidBinaryMobileInstall.createMobileInstallResourceApks(
            ruleContext, dataContext, manifest);

    return createAndroidBinary(
        ruleContext,
        dataContext,
        filesBuilder,
        deployJar,
        derivedJarFunction,
        isBinaryJarFiltered,
        androidCommon,
        javaSemantics,
        androidSemantics,
        nativeLibs,
        resourceApk,
        mobileInstallResourceApks,
        resourceClasses,
        ImmutableList.<Artifact>of(),
        ImmutableList.<Artifact>of(),
        proguardMapping,
        oneVersionOutputArtifact);
  }

  public static RuleConfiguredTargetBuilder createAndroidBinary(
      RuleContext ruleContext,
      AndroidDataContext dataContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact binaryJar,
      Function<Artifact, Artifact> derivedJarFunction,
      boolean isBinaryJarFiltered,
      AndroidCommon androidCommon,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics,
      NativeLibs nativeLibs,
      ResourceApk resourceApk,
      @Nullable MobileInstallResourceApks mobileInstallResourceApks,
      JavaTargetAttributes resourceClasses,
      ImmutableList<Artifact> apksUnderTest,
      ImmutableList<Artifact> additionalMergedManifests,
      Artifact proguardMapping,
      @Nullable Artifact oneVersionEnforcementArtifact)
      throws InterruptedException, RuleErrorException {

    List<ProguardSpecProvider> proguardDeps = new ArrayList<>();
    Iterables.addAll(
        proguardDeps,
        ruleContext.getPrerequisites("deps", TransitionMode.TARGET, ProguardSpecProvider.PROVIDER));
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()
        && ruleContext.attributes().has("$jacoco_runtime", BuildType.LABEL)) {
      proguardDeps.add(
          ruleContext.getPrerequisite(
              "$jacoco_runtime", TransitionMode.TARGET, ProguardSpecProvider.PROVIDER));
    }
    ImmutableList<Artifact> proguardSpecs =
        getProguardSpecs(
            dataContext,
            androidSemantics,
            resourceApk.getResourceProguardConfig(),
            resourceApk.getManifest(),
            ruleContext.attributes().has(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
                ? ruleContext
                    .getPrerequisiteArtifacts(ProguardHelper.PROGUARD_SPECS, TransitionMode.TARGET)
                    .list()
                : ImmutableList.<Artifact>of(),
            ruleContext
                .getPrerequisiteArtifacts(":extra_proguard_specs", TransitionMode.TARGET)
                .list(),
            proguardDeps);
    boolean hasProguardSpecs = !proguardSpecs.isEmpty();

    // TODO(bazel-team): Verify that proguard spec files don't contain -printmapping directions
    // which this -printmapping command line flag will override.
    Artifact proguardOutputMap = null;
    if (ProguardHelper.genProguardMapping(ruleContext.attributes())
        || dataContext.isResourceShrinkingEnabled()) {
      proguardOutputMap = androidSemantics.getProguardOutputMap(ruleContext);
    }

    ProguardOutput proguardOutput =
        applyProguard(
            ruleContext,
            androidCommon,
            javaSemantics,
            binaryJar,
            proguardSpecs,
            proguardMapping,
            proguardOutputMap);

    if (dataContext.useResourceShrinking(hasProguardSpecs)) {
      resourceApk =
          shrinkResources(
              ruleContext,
              androidSemantics.makeContextForNative(ruleContext),
              resourceApk,
              proguardOutput,
              filesBuilder);
    }

    resourceApk = maybeOptimizeResources(dataContext, resourceApk, hasProguardSpecs);

    Artifact jarToDex = proguardOutput.getOutputJar();
    DexingOutput dexingOutput =
        dex(
            ruleContext,
            androidSemantics,
            binaryJar,
            jarToDex,
            isBinaryJarFiltered,
            androidCommon,
            resourceApk.getMainDexProguardConfig(),
            resourceClasses,
            derivedJarFunction,
            proguardOutputMap);

    // Collect all native shared libraries across split transitions. Some AARs contain shared
    // libraries across multiple architectures, e.g. x86 and armeabi-v7a, and need to be packed
    // into the APK.
    NestedSetBuilder<Artifact> transitiveNativeLibs = NestedSetBuilder.naiveLinkOrder();
    for (Map.Entry<
            com.google.common.base.Optional<String>,
            ? extends List<? extends TransitiveInfoCollection>>
        entry : ruleContext.getSplitPrerequisites("deps").entrySet()) {
      for (AndroidNativeLibsInfo provider :
          AnalysisUtils.getProviders(entry.getValue(), AndroidNativeLibsInfo.PROVIDER)) {
        transitiveNativeLibs.addTransitive(provider.getNativeLibs());
      }
    }
    NestedSet<Artifact> nativeLibsAar = transitiveNativeLibs.build();

    DexPostprocessingOutput dexPostprocessingOutput =
        androidSemantics.postprocessClassesDexZip(
            ruleContext, filesBuilder, dexingOutput.classesDexZip, proguardOutput);

    if (!proguardSpecs.isEmpty()) {
      proguardOutput.addAllToSet(filesBuilder, dexPostprocessingOutput.proguardMap());
    }

    Artifact unsignedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_UNSIGNED_APK);
    Artifact zipAlignedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_APK);
    Artifact signingKey = AndroidCommon.getApkDebugSigningKey(ruleContext);
    FilesToRunProvider resourceExtractor =
        ruleContext.getExecutablePrerequisite("$resource_extractor", TransitionMode.HOST);

    Artifact finalClassesDex;
    ImmutableList<Artifact> finalShardDexZips = dexingOutput.shardDexZips;
    if (AndroidCommon.getAndroidConfig(ruleContext).desugarJava8Libs()
        && dexPostprocessingOutput.classesDexZip().getFilename().endsWith(".zip")) {
      Artifact java8LegacyDex;
      if (binaryJar.equals(jarToDex)) {
        // No Proguard: use canned Java 8 legacy .dex file
        java8LegacyDex =
            ruleContext.getPrerequisiteArtifact("$java8_legacy_dex", TransitionMode.TARGET);
      } else {
        // Proguard is used: build custom Java 8 legacy .dex file
        java8LegacyDex = getDxArtifact(ruleContext, "_java8_legacy.dex.zip");
        Artifact androidJar = AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar();
        ruleContext.registerAction(
            new SpawnAction.Builder()
                .setExecutable(
                    ruleContext.getExecutablePrerequisite(
                        "$build_java8_legacy_dex", TransitionMode.HOST))
                .addInput(jarToDex)
                .addInput(androidJar)
                .addOutput(java8LegacyDex)
                .addCommandLine(
                    CustomCommandLine.builder()
                        .addExecPath("--binary", jarToDex)
                        .addExecPath("--android_jar", androidJar)
                        .addExecPath("--output", java8LegacyDex)
                        .build())
                .setMnemonic("BuildLegacyDex")
                .setProgressMessage("Building Java 8 legacy library for %s", ruleContext.getLabel())
                .build(ruleContext));
      }

      // Append legacy .dex library to app's .dex files
      finalClassesDex = getDxArtifact(ruleContext, "_final_classes.dex.zip");
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .useDefaultShellEnvironment()
              .setMnemonic("AppendJava8LegacyDex")
              .setProgressMessage("Adding Java 8 legacy library for %s", ruleContext.getLabel())
              .setExecutable(
                  ruleContext.getExecutablePrerequisite("$merge_dexzips", TransitionMode.HOST))
              .addInput(dexPostprocessingOutput.classesDexZip())
              .addInput(java8LegacyDex)
              .addOutput(finalClassesDex)
              // Order matters here: we want java8LegacyDex to be the highest-numbered classesN.dex
              .addCommandLine(
                  CustomCommandLine.builder()
                      .addExecPath("--input_zip", dexPostprocessingOutput.classesDexZip())
                      .addExecPath("--input_zip", java8LegacyDex)
                      .addExecPath("--output_zip", finalClassesDex)
                      .build())
              .build(ruleContext));
      finalShardDexZips =
          ImmutableList.<Artifact>builder().addAll(finalShardDexZips).add(java8LegacyDex).build();
    } else {
      finalClassesDex = dexPostprocessingOutput.classesDexZip();
    }

    ApkActionsBuilder.create("apk")
        .setClassesDex(finalClassesDex)
        .addInputZip(resourceApk.getArtifact())
        .setJavaResourceZip(dexingOutput.javaResourceJar, resourceExtractor)
        .addInputZips(nativeLibsAar.toList())
        .setNativeLibs(nativeLibs)
        .setUnsignedApk(unsignedApk)
        .setSignedApk(zipAlignedApk)
        .setSigningKey(signingKey)
        .setZipalignApk(true)
        .registerActions(ruleContext);

    filesBuilder.add(binaryJar);
    filesBuilder.add(unsignedApk);
    filesBuilder.add(zipAlignedApk);
    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    Artifact deployInfo = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.DEPLOY_INFO);
    AndroidDeployInfoAction.createDeployInfoAction(
        ruleContext,
        deployInfo,
        resourceApk.getManifest(),
        additionalMergedManifests,
        ImmutableList.<Artifact>builder().add(zipAlignedApk).addAll(apksUnderTest).build());

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);

    // If this is an instrumentation APK, create the provider for android_instrumentation_test.
    if (isInstrumentation(ruleContext)) {
      ApkInfo targetApkProvider =
          ruleContext.getPrerequisite("instruments", TransitionMode.TARGET, ApkInfo.PROVIDER);

      AndroidInstrumentationInfo instrumentationProvider =
          new AndroidInstrumentationInfo(targetApkProvider);

      builder.addNativeDeclaredProvider(instrumentationProvider);

      // At this point, the Android manifests of both target and instrumentation APKs are finalized.
      FilesToRunProvider checker =
          ruleContext.getExecutablePrerequisite("$instrumentation_test_check", TransitionMode.HOST);
      Artifact targetManifest = targetApkProvider.getMergedManifest();
      Artifact instrumentationManifest = resourceApk.getManifest();
      Artifact checkOutput =
          ruleContext.getImplicitOutputArtifact(
              AndroidRuleClasses.INSTRUMENTATION_TEST_CHECK_RESULTS);

      SpawnAction.Builder checkAction =
          new SpawnAction.Builder()
              .setExecutable(checker)
              .addInput(targetManifest)
              .addInput(instrumentationManifest)
              .addOutput(checkOutput)
              .setProgressMessage(
                  "Validating the merged manifests of the target and instrumentation APKs")
              .setMnemonic("AndroidManifestInstrumentationCheck");

      CustomCommandLine commandLine =
          CustomCommandLine.builder()
              .addExecPath("--instrumentation_manifest", instrumentationManifest)
              .addExecPath("--target_manifest", targetManifest)
              .addExecPath("--output", checkOutput)
              .build();

      builder.addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, checkOutput);
      checkAction.addCommandLine(commandLine);
      ruleContext.registerAction(checkAction.build(ruleContext));
    }

    androidCommon.addTransitiveInfoProviders(
        builder,
        /* aar= */ null,
        resourceApk,
        zipAlignedApk,
        apksUnderTest,
        nativeLibs,
        androidCommon.isNeverLink(),
        /* isLibrary = */ false);

    if (dexPostprocessingOutput.proguardMap() != null) {
      builder.addNativeDeclaredProvider(
          new ProguardMappingProvider(dexPostprocessingOutput.proguardMap()));
    }

    if (oneVersionEnforcementArtifact != null) {
      builder.addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, oneVersionEnforcementArtifact);
    }

    if (mobileInstallResourceApks != null) {
      AndroidBinaryMobileInstall.addMobileInstall(
          ruleContext,
          builder,
          dexingOutput.javaResourceJar,
          finalShardDexZips,
          javaSemantics,
          nativeLibs,
          resourceApk,
          mobileInstallResourceApks,
          resourceExtractor,
          nativeLibsAar,
          signingKey,
          additionalMergedManifests);
    }

    return builder
        .setFilesToBuild(filesToBuild)
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.simple(
                new Runfiles.Builder(
                        ruleContext.getWorkspaceName(),
                        ruleContext.getConfiguration().legacyExternalRunfiles())
                    .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
                    .addTransitiveArtifacts(filesToBuild)
                    .build()))
        .addProvider(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(resourceClasses, javaSemantics))
        .addNativeDeclaredProvider(
            new ApkInfo(
                zipAlignedApk,
                unsignedApk,
                getCoverageInstrumentationJarForApk(ruleContext),
                resourceApk.getManifest(),
                AndroidCommon.getApkDebugSigningKey(ruleContext)))
        .addNativeDeclaredProvider(new AndroidPreDexJarProvider(jarToDex))
        .addNativeDeclaredProvider(
            AndroidFeatureFlagSetProvider.create(
                AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext)))
        // Report set feature flags as required "config fragments".
        // While these aren't technically fragments, in practice they're user-defined settings with
        // the same meaning: pieces of configuration the rule requires to work properly. So it makes
        // sense to treat them equivalently for "requirements" reporting purposes.
        .addRequiredConfigFragments(AndroidFeatureFlagSetProvider.getFlagNames(ruleContext))
        .addOutputGroup("android_deploy_info", deployInfo);
  }

  /**
   * For coverage builds, this returns a Jar containing <b>un</b>instrumented bytecode for the
   * coverage reporter's consumption. This method simply returns the deploy Jar. Note the deploy Jar
   * is built anyway for Android binaries.
   *
   * @return A Jar containing uninstrumented bytecode or {@code null} for non-coverage builds
   */
  @Nullable
  private static Artifact getCoverageInstrumentationJarForApk(RuleContext ruleContext)
      throws InterruptedException {
    return ruleContext.getConfiguration().isCodeCoverageEnabled()
        ? ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_DEPLOY_JAR)
        : null;
  }

  public static NestedSet<Artifact> getLibraryResourceJars(RuleContext ruleContext) {
    Iterable<AndroidLibraryResourceClassJarProvider> libraryResourceJarProviders =
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, TransitionMode.TARGET, AndroidLibraryResourceClassJarProvider.PROVIDER);

    NestedSetBuilder<Artifact> libraryResourceJarsBuilder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidLibraryResourceClassJarProvider provider : libraryResourceJarProviders) {
      libraryResourceJarsBuilder.addTransitive(provider.getResourceClassJars());
    }
    return libraryResourceJarsBuilder.build();
  }

  /** Generates an uncompressed _deploy.jar of all the runtime jars. */
  public static Artifact createDeployJar(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      AndroidCommon common,
      JavaTargetAttributes attributes,
      boolean checkDesugarDeps,
      Function<Artifact, Artifact> derivedJarFunction)
      throws InterruptedException {
    Artifact deployJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_DEPLOY_JAR);
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(deployJar)
        .setAttributes(attributes)
        .addRuntimeJars(common.getRuntimeJars())
        .setDerivedJarFunction(derivedJarFunction)
        .setCheckDesugarDeps(checkDesugarDeps)
        .build();
    return deployJar;
  }

  /**
   * Applies the proguard specifications, and creates a ProguardedJar. Proguard's output artifacts
   * are added to the given {@code filesBuilder}.
   */
  private static ProguardOutput applyProguard(
      RuleContext ruleContext,
      AndroidCommon common,
      JavaSemantics javaSemantics,
      Artifact deployJarArtifact,
      ImmutableList<Artifact> proguardSpecs,
      Artifact proguardMapping,
      @Nullable Artifact proguardOutputMap)
      throws InterruptedException {
    Artifact proguardOutputJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_PROGUARD_JAR);

    // Proguard will be only used for binaries which specify a proguard_spec
    if (proguardSpecs.isEmpty()) {
      // Although normally the Proguard jar artifact is not needed for binaries which do not specify
      // proguard_specs, targets which use a select to provide an empty list to proguard_specs will
      // still have a Proguard jar implicit output, as it is impossible to tell what a select will
      // produce at the time of implicit output determination. As a result, this artifact must
      // always be created.
      return createEmptyProguardAction(
          ruleContext, javaSemantics, proguardOutputJar, deployJarArtifact, proguardOutputMap);
    }

    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    NestedSetBuilder<Artifact> libraryJars =
        NestedSetBuilder.<Artifact>naiveLinkOrder().add(sdk.getAndroidJar());
    if (AndroidCommon.getAndroidConfig(ruleContext).desugarJava8Libs()) {
      // Proguard sees the desugared app, so it needs legacy APIs to resolve symbols
      libraryJars.addTransitive(
          ruleContext
              .getPrerequisite("$desugared_java8_legacy_apis", TransitionMode.TARGET)
              .getProvider(FileProvider.class)
              .getFilesToBuild());
    }
    libraryJars.addTransitive(common.getTransitiveNeverLinkLibraries());

    Artifact proguardSeeds =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_SEEDS);
    Artifact proguardUsage =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_USAGE);
    Artifact proguardDictionary =
        ruleContext.getPrerequisiteArtifact("proguard_apply_dictionary", TransitionMode.TARGET);
    ProguardOutput result =
        ProguardHelper.createOptimizationActions(
            ruleContext,
            sdk.getProguard(),
            deployJarArtifact,
            proguardSpecs,
            proguardSeeds,
            proguardUsage,
            proguardMapping,
            proguardDictionary,
            libraryJars.build(),
            proguardOutputJar,
            javaSemantics,
            getProguardOptimizationPasses(ruleContext),
            proguardOutputMap);
    return result;
  }

  @Nullable
  private static Integer getProguardOptimizationPasses(RuleContext ruleContext) {
    if (ruleContext.attributes().has("proguard_optimization_passes", Type.INTEGER)) {
      return ruleContext.attributes().get("proguard_optimization_passes", Type.INTEGER);
    } else {
      return null;
    }
  }

  private static ProguardOutput createEmptyProguardAction(
      RuleContext ruleContext,
      JavaSemantics semantics,
      Artifact proguardOutputJar,
      Artifact deployJarArtifact,
      Artifact proguardOutputMap)
      throws InterruptedException {
    NestedSetBuilder<Artifact> failures = NestedSetBuilder.<Artifact>stableOrder();
    ProguardOutput outputs =
        ProguardHelper.getProguardOutputs(
            proguardOutputJar,
            /* proguardSeeds */ (Artifact) null,
            /* proguardUsage */ (Artifact) null,
            ruleContext,
            semantics,
            proguardOutputMap);
    outputs.addAllToSet(failures);
    ruleContext.registerAction(
        new FailAction(
            ruleContext.getActionOwner(),
            failures.build().toSet(),
            String.format("Can't run Proguard without proguard_specs")));
    return new ProguardOutput(deployJarArtifact, null, null, null, null, null, null);
  }

  static ImmutableList<Artifact> getProguardSpecs(
      AndroidDataContext dataContext,
      AndroidSemantics androidSemantics,
      Artifact resourceProguardConfig,
      Artifact mergedManifest,
      ImmutableList<Artifact> localProguardSpecs,
      ImmutableList<Artifact> extraProguardSpecs,
      Iterable<ProguardSpecProvider> proguardDeps) {

    ImmutableList<Artifact> proguardSpecs =
        ProguardHelper.collectTransitiveProguardSpecs(
            dataContext.getLabel(),
            dataContext.getActionConstructionContext(),
            Iterables.concat(ImmutableList.of(resourceProguardConfig), extraProguardSpecs),
            localProguardSpecs,
            proguardDeps);

    boolean assumeMinSdkVersion = dataContext.getAndroidConfig().assumeMinSdkVersion();
    if (!proguardSpecs.isEmpty() && assumeMinSdkVersion) {
      // NB: Order here is important. We're including generated Proguard specs before the user's
      // specs so that they can override values.
      proguardSpecs =
          ImmutableList.<Artifact>builder()
              .addAll(androidSemantics.getProguardSpecsForManifest(dataContext, mergedManifest))
              .addAll(proguardSpecs)
              .build();
    }

    return proguardSpecs;
  }

  /** Returns {@code true} if resource shrinking should be performed. */
  static boolean shouldShrinkResourceCycles(
      AndroidConfiguration androidConfig, RuleErrorConsumer errorConsumer, boolean shrinkResources)
      throws RuleErrorException {
    boolean global = androidConfig.useAndroidResourceCycleShrinking();
    if (global && !shrinkResources) {
      throw errorConsumer.throwWithRuleError(
          "resource cycle shrinking can only be enabled when resource shrinking is enabled");
    }
    return global;
  }

  private static ResourceApk shrinkResources(
      RuleContext ruleContext,
      AndroidDataContext dataContext,
      ResourceApk resourceApk,
      ProguardOutput proguardOutput,
      NestedSetBuilder<Artifact> filesBuilder)
      throws RuleErrorException, InterruptedException {

    Artifact shrunkApk =
        shrinkResources(
            dataContext,
            resourceApk.getValidatedResources(),
            resourceApk.getResourceDependencies(),
            proguardOutput.getOutputJar(),
            proguardOutput.getMapping(),
            ResourceFilterFactory.fromRuleContextAndAttrs(ruleContext),
            ruleContext.getExpander().withDataLocations().tokenized("nocompress_extensions"));

    filesBuilder.add(
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCE_SHRINKER_LOG));
    return resourceApk.withApk(shrunkApk);
  }

  static Artifact shrinkResources(
      AndroidDataContext dataContext,
      ValidatedAndroidResources validatedResources,
      ResourceDependencies resourceDeps,
      Artifact proguardOutputJar,
      Artifact proguardMapping,
      ResourceFilterFactory resourceFilterFactory,
      List<String> noCompressExtensions)
      throws InterruptedException {

    ResourceShrinkerActionBuilder resourceShrinkerActionBuilder =
        new ResourceShrinkerActionBuilder()
            .setResourceApkOut(
                dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_SHRUNK_APK))
            .setShrunkResourcesOut(
                dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_SHRUNK_ZIP))
            .setLogOut(
                dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCE_SHRINKER_LOG))
            .withResourceFiles(
                dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP))
            .withShrunkJar(proguardOutputJar)
            .withProguardMapping(proguardMapping)
            .withPrimary(validatedResources)
            .withDependencies(resourceDeps)
            .setResourceFilterFactory(resourceFilterFactory)
            .setUncompressedExtensions(noCompressExtensions)
            .setResourceOptimizationConfigOut(
                dataContext.createOutputArtifact(
                    AndroidRuleClasses.ANDROID_RESOURCE_OPTIMIZATION_CONFIG));
    return resourceShrinkerActionBuilder.build(dataContext);
  }

  private static ResourceApk maybeOptimizeResources(
      AndroidDataContext dataContext, ResourceApk resourceApk, boolean hasProguardSpecs)
      throws InterruptedException {
    boolean useResourcePathShortening = dataContext.useResourcePathShortening();
    boolean useResourceNameObfuscation = dataContext.useResourceNameObfuscation(hasProguardSpecs);
    if (!useResourcePathShortening && !useResourceNameObfuscation) {
      return resourceApk;
    }

    Artifact optimizedApk =
        dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_OPTIMIZED_APK);

    Aapt2OptimizeActionBuilder.Builder builder =
        Aapt2OptimizeActionBuilder.builder()
            .setResourceApk(resourceApk.getArtifact())
            .setOptimizedApkOut(optimizedApk);
    if (useResourcePathShortening) {
      builder.setResourcePathShorteningMapOut(
          dataContext.createOutputArtifact(
              AndroidRuleClasses.ANDROID_RESOURCE_PATH_SHORTENING_MAP));
    }
    if (useResourceNameObfuscation) {
      builder.setResourceOptimizationConfig(
          dataContext.createOutputArtifact(
              AndroidRuleClasses.ANDROID_RESOURCE_OPTIMIZATION_CONFIG));
    }
    builder.build().registerAction(dataContext);

    return resourceApk.withApk(optimizedApk);
  }

  @Immutable
  static final class DexingOutput {
    private final Artifact classesDexZip;
    final Artifact javaResourceJar;
    final ImmutableList<Artifact> shardDexZips;

    private DexingOutput(
        Artifact classesDexZip, Artifact javaResourceJar, ImmutableList<Artifact> shardDexZips) {
      this.classesDexZip = classesDexZip;
      this.javaResourceJar = javaResourceJar;
      this.shardDexZips = Preconditions.checkNotNull(shardDexZips);
    }
  }

  /** All artifacts modified by any dex post-processing steps. */
  @AutoValue
  public abstract static class DexPostprocessingOutput {

    public static DexPostprocessingOutput create(Artifact classesDexZip, Artifact proguardMap) {
      return new AutoValue_AndroidBinary_DexPostprocessingOutput(classesDexZip, proguardMap);
    }

    /** A .zip of .dex files to include in the APK. */
    abstract Artifact classesDexZip();

    /**
     * The proguard mapping corresponding to the post-processed dex files. This may be null if
     * proguard was not run.
     */
    @Nullable
    abstract Artifact proguardMap();
  }

  /** Creates one or more classes.dex files that correspond to {@code proguardedJar}. */
  private static DexingOutput dex(
      RuleContext ruleContext,
      AndroidSemantics androidSemantics,
      Artifact binaryJar,
      Artifact proguardedJar,
      boolean isBinaryJarFiltered,
      AndroidCommon common,
      @Nullable Artifact mainDexProguardSpec,
      JavaTargetAttributes attributes,
      Function<Artifact, Artifact> derivedJarFunction,
      @Nullable Artifact proguardOutputMap)
      throws InterruptedException, RuleErrorException {
    List<String> dexopts = ruleContext.getExpander().withDataLocations().tokenized("dexopts");
    MultidexMode multidexMode = getMultidexMode(ruleContext);
    if (!supportsMultidexMode(ruleContext, multidexMode)) {
      ruleContext.throwWithRuleError(
          "Multidex mode \""
              + multidexMode.getAttributeValue()
              + "\" not supported by this version of the Android SDK");
    }

    int dexShards = ruleContext.attributes().get("dex_shards", Type.INTEGER);
    if (dexShards > 1) {
      if (multidexMode == MultidexMode.OFF) {
        ruleContext.throwWithRuleError(".dex sharding is only available in multidex mode");
      }

      if (multidexMode == MultidexMode.MANUAL_MAIN_DEX) {
        ruleContext.throwWithRuleError(".dex sharding is not available in manual multidex mode");
      }
    }

    Artifact mainDexList =
        ruleContext.getPrerequisiteArtifact("main_dex_list", TransitionMode.TARGET);
    if ((mainDexList != null && multidexMode != MultidexMode.MANUAL_MAIN_DEX)
        || (mainDexList == null && multidexMode == MultidexMode.MANUAL_MAIN_DEX)) {
      ruleContext.throwWithRuleError(
          "Both \"main_dex_list\" and \"multidex='manual_main_dex'\" must be specified.");
    }

    boolean usesDexArchives =
        getEffectiveIncrementalDexing(
            ruleContext, dexopts, !Objects.equals(binaryJar, proguardedJar));
    Artifact inclusionFilterJar =
        isBinaryJarFiltered && Objects.equals(binaryJar, proguardedJar) ? binaryJar : null;
    Artifact singleJarToDex = !Objects.equals(binaryJar, proguardedJar) ? proguardedJar : null;
    if (multidexMode == MultidexMode.OFF) {
      // Single dex mode: generate classes.dex directly from the input jar.
      if (usesDexArchives) {
        Artifact classesDex = getDxArtifact(ruleContext, "classes.dex.zip");
        createIncrementalDexingActions(
            ruleContext,
            singleJarToDex,
            common,
            inclusionFilterJar,
            dexopts,
            androidSemantics,
            attributes,
            derivedJarFunction,
            /*multidex=*/ false,
            /*mainDexList=*/ null,
            classesDex);
        return new DexingOutput(classesDex, binaryJar, ImmutableList.of(classesDex));
      } else {
        // By *not* writing a zip we get dx to drop resources on the floor.
        Artifact classesDex = getDxArtifact(ruleContext, "classes.dex");
        AndroidCommon.createDexAction(
            ruleContext,
            proguardedJar,
            classesDex,
            dexopts,
            /*multidex=*/ false,
            /*mainDexList=*/ null);
        return new DexingOutput(classesDex, binaryJar, ImmutableList.of(classesDex));
      }
    } else {
      // Multidex mode: generate classes.dex.zip, where the zip contains [classes.dex,
      // classes2.dex, ... classesN.dex].

      if (multidexMode == MultidexMode.LEGACY) {
        // For legacy multidex, we need to generate a list for the dexer's --main-dex-list flag.
        mainDexList =
            createMainDexListAction(
                ruleContext,
                androidSemantics,
                proguardedJar,
                mainDexProguardSpec,
                proguardOutputMap);
      } else if (multidexMode == MultidexMode.MANUAL_MAIN_DEX) {
        mainDexList =
            transformDexListThroughProguardMapAction(ruleContext, proguardOutputMap, mainDexList);
      }

      Artifact classesDex = getDxArtifact(ruleContext, "classes.dex.zip");
      if (dexShards > 1) {
        ImmutableList<Artifact> shards =
            makeShardArtifacts(ruleContext, dexShards, usesDexArchives ? ".jar.dex.zip" : ".jar");

        Artifact javaResourceJar =
            createShuffleJarActions(
                ruleContext,
                usesDexArchives,
                singleJarToDex,
                shards,
                common,
                inclusionFilterJar,
                dexopts,
                androidSemantics,
                attributes,
                derivedJarFunction,
                mainDexList);

        ImmutableList.Builder<Artifact> shardDexesBuilder = ImmutableList.builder();
        for (int i = 1; i <= dexShards; i++) {
          Artifact shard = shards.get(i - 1);
          Artifact shardDex = getDxArtifact(ruleContext, "shard" + i + ".dex.zip");
          shardDexesBuilder.add(shardDex);
          if (usesDexArchives) {
            // If there's a main dex list then the first shard contains exactly those files.
            // To work with devices that lack native multi-dex support we need to make sure that
            // the main dex list becomes one dex file if at all possible.
            // Note shard here (mostly) contains of .class.dex files from shuffled dex archives,
            // instead of being a conventional Jar file with .class files.
            createDexMergerAction(
                ruleContext,
                mainDexList != null && i == 1 ? "minimal" : "best_effort",
                ImmutableList.of(shard),
                shardDex,
                /*mainDexList=*/ null,
                dexopts);
          } else {
            AndroidCommon.createDexAction(
                ruleContext, shard, shardDex, dexopts, /*multidex=*/ true, /*mainDexList=*/ null);
          }
        }
        ImmutableList<Artifact> shardDexes = shardDexesBuilder.build();

        CommandLine mergeCommandLine =
            CustomCommandLine.builder()
                .addExecPaths(VectorArg.addBefore("--input_zip").each(shardDexes))
                .addExecPath("--output_zip", classesDex)
                .build();
        ruleContext.registerAction(
            new SpawnAction.Builder()
                .useDefaultShellEnvironment()
                .setMnemonic("MergeDexZips")
                .setProgressMessage("Merging dex shards for %s", ruleContext.getLabel())
                .setExecutable(
                    ruleContext.getExecutablePrerequisite("$merge_dexzips", TransitionMode.HOST))
                .addInputs(shardDexes)
                .addOutput(classesDex)
                .addCommandLine(mergeCommandLine)
                .build(ruleContext));
        if (usesDexArchives) {
          // Using the deploy jar for java resources gives better "bazel mobile-install" performance
          // with incremental dexing b/c bazel can create the "incremental" and "split resource"
          // APKs earlier (b/c these APKs don't depend on code being dexed here).  This is also done
          // for other multidex modes.
          javaResourceJar = binaryJar;
        }
        return new DexingOutput(classesDex, javaResourceJar, shardDexes);
      } else {
        if (usesDexArchives) {
          createIncrementalDexingActions(
              ruleContext,
              singleJarToDex,
              common,
              inclusionFilterJar,
              dexopts,
              androidSemantics,
              attributes,
              derivedJarFunction,
              /*multidex=*/ true,
              mainDexList,
              classesDex);
        } else {
          // Because the dexer also places resources into this zip, we also need to create a cleanup
          // action that removes all non-.dex files before staging for apk building.
          // Create an artifact for the intermediate zip output that includes non-.dex files.
          Artifact classesDexIntermediate =
              AndroidBinary.getDxArtifact(ruleContext, "intermediate_classes.dex.zip");
          // Have the dexer generate the intermediate file and the "cleaner" action consume this to
          // generate the final archive with only .dex files.
          AndroidCommon.createDexAction(
              ruleContext,
              proguardedJar,
              classesDexIntermediate,
              dexopts,
              /*multidex=*/ true,
              mainDexList);
          createCleanDexZipAction(ruleContext, classesDexIntermediate, classesDex);
        }
        return new DexingOutput(classesDex, binaryJar, ImmutableList.of(classesDex));
      }
    }
  }

  /**
   * Helper that sets up dexbuilder/dexmerger actions when dex_shards attribute is not set, for use
   * with or without multidex.
   */
  private static void createIncrementalDexingActions(
      RuleContext ruleContext,
      @Nullable Artifact proguardedJar,
      AndroidCommon common,
      @Nullable Artifact inclusionFilterJar,
      List<String> dexopts,
      AndroidSemantics androidSemantics,
      JavaTargetAttributes attributes,
      Function<Artifact, Artifact> derivedJarFunction,
      boolean multidex,
      @Nullable Artifact mainDexList,
      Artifact classesDex)
      throws InterruptedException, RuleErrorException {
    ImmutableList<Artifact> dexArchives;
    if (proguardedJar == null
        && (multidex || inclusionFilterJar == null)
        && AndroidCommon.getAndroidConfig(ruleContext).incrementalDexingUseDexSharder()) {
      dexArchives =
          toDexedClasspath(
              ruleContext,
              collectRuntimeJars(common, attributes),
              collectDexArchives(
                  ruleContext, common, dexopts, androidSemantics, derivedJarFunction));
    } else {
      if (proguardedJar != null
          && AndroidCommon.getAndroidConfig(ruleContext).incrementalDexingShardsAfterProguard()
              > 1) {
        // TODO(b/69816569): Also use this logic if #shards > #Jars on runtime classpath
        dexArchives =
            makeShardArtifacts(
                ruleContext,
                AndroidCommon.getAndroidConfig(ruleContext).incrementalDexingShardsAfterProguard(),
                ".jar.dex.zip");
      } else {
        dexArchives = ImmutableList.of(AndroidBinary.getDxArtifact(ruleContext, "classes.jar"));
      }

      if (proguardedJar != null && dexArchives.size() == 1) {
        // No need to shuffle, just run proguarded Jar through dexbuilder
        DexArchiveAspect.createDexArchiveAction(
            ruleContext,
            "$dexbuilder_after_proguard",
            proguardedJar,
            DexArchiveAspect.topLevelDexbuilderDexopts(dexopts),
            dexArchives.get(0));
      } else {
        createShuffleJarActions(
            ruleContext,
            /*makeDexArchives=*/ true,
            proguardedJar,
            dexArchives,
            common,
            inclusionFilterJar,
            dexopts,
            androidSemantics,
            attributes,
            derivedJarFunction,
            (Artifact) null);
        inclusionFilterJar = null;
      }
    }

    if (dexArchives.size() == 1 || !multidex) {
      checkState(inclusionFilterJar == null);
      createDexMergerAction(
          ruleContext, multidex ? "minimal" : "off", dexArchives, classesDex, mainDexList, dexopts);
    } else {
      SpecialArtifact shardsToMerge =
          createSharderAction(ruleContext, dexArchives, mainDexList, dexopts, inclusionFilterJar);
      Artifact multidexShards = createTemplatedMergerActions(ruleContext, shardsToMerge, dexopts);
      // TODO(b/69431301): avoid this action and give the files to apk build action directly
      createZipMergeAction(ruleContext, multidexShards, classesDex);
    }
  }

  private static ImmutableList<Artifact> makeShardArtifacts(
      RuleContext ruleContext, int shardCount, String suffix) {
    ImmutableList.Builder<Artifact> shardsBuilder = ImmutableList.builder();
    for (int i = 1; i <= shardCount; i++) {
      shardsBuilder.add(getDxArtifact(ruleContext, "shard" + i + suffix));
    }
    return shardsBuilder.build();
  }

  /**
   * Returns whether incremental dexing should actually be used based on the --incremental_dexing
   * flag, the incremental_dexing attribute and the target's dexopts.
   */
  private static boolean getEffectiveIncrementalDexing(
      RuleContext ruleContext, List<String> dexopts, boolean isBinaryProguarded) {
    TriState override = ruleContext.attributes().get("incremental_dexing", BuildType.TRISTATE);
    AndroidConfiguration config = AndroidCommon.getAndroidConfig(ruleContext);
    // Ignore --incremental_dexing if the incremental_dexing attribute is set, but require the
    // attribute to be YES for proguarded binaries and binaries with blacklisted dexopts.
    if (isBinaryProguarded
        && override == TriState.YES
        && config.incrementalDexingShardsAfterProguard() <= 0) {
      ruleContext.attributeError(
          "incremental_dexing", "target cannot be incrementally dexed because it uses Proguard");
      return false;
    }

    if (override == TriState.NO) {
      return false;
    }
    if (override == TriState.YES || config.useIncrementalDexing()) {
      if (isBinaryProguarded) {
        return override == TriState.YES || config.incrementalDexingAfterProguardByDefault();
      }
      Iterable<String> blacklistedDexopts =
          DexArchiveAspect.blacklistedDexopts(ruleContext, dexopts);
      if (Iterables.isEmpty(blacklistedDexopts)) {
        // target's dexopts are all compatible with incremental dexing.
        return true;
      } else if (override == TriState.YES) {
        // target's dexopts include flags blacklisted with --non_incremental_per_target_dexopts. If
        // incremental_dexing attribute is explicitly set for this target then we'll warn and
        // incrementally dex anyway.  Otherwise, just don't incrementally dex.
        Iterable<String> ignored =
            Iterables.filter(
                blacklistedDexopts,
                Predicates.not(Predicates.in(config.getDexoptsSupportedInIncrementalDexing())));
        ruleContext.attributeWarning(
            "incremental_dexing",
            String.format(
                "Using incremental dexing even though dexopts %s indicate this target "
                    + "may be unsuitable for incremental dexing for the moment.%s",
                blacklistedDexopts,
                Iterables.isEmpty(ignored) ? "" : " Ignored dexopts: " + ignored));
        return true;
      } else {
        // If there are incompatible dexopts and the attribute is not set, we silently don't run
        // incremental dexing.
        return false;
      }
    } else {
      // attribute is auto and flag is false
      return false;
    }
  }

  /**
   * Sets up a {@code $dexsharder} action for the given {@code dexArchives} and returns the output
   * tree artifact.
   *
   * @return Tree artifact containing dex archives to merge into exactly one .dex file each
   */
  private static SpecialArtifact createSharderAction(
      RuleContext ruleContext,
      ImmutableList<Artifact> dexArchives,
      @Nullable Artifact mainDexList,
      Collection<String> dexopts,
      @Nullable Artifact inclusionFilterJar) {
    SpecialArtifact outputTree =
        ruleContext.getTreeArtifact(
            ruleContext.getUniqueDirectory("dexsplits"), ruleContext.getBinOrGenfilesDirectory());

    SpawnAction.Builder shardAction =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setMnemonic("ShardForMultidex")
            .setProgressMessage(
                "Assembling dex files for %s", ruleContext.getLabel().getCanonicalForm())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$dexsharder", TransitionMode.HOST))
            .addInputs(dexArchives)
            .addOutput(outputTree);

    CustomCommandLine.Builder shardCommandLine =
        CustomCommandLine.builder()
            .addExecPaths(VectorArg.addBefore("--input").each(dexArchives))
            .addExecPath("--output", outputTree);

    if (mainDexList != null) {
      shardAction.addInput(mainDexList);
      shardCommandLine.addExecPath("--main-dex-list", mainDexList);
    }
    shardCommandLine.addAll(DexArchiveAspect.sharderDexopts(ruleContext, dexopts));
    if (inclusionFilterJar != null) {
      shardCommandLine.addExecPath("--inclusion_filter_jar", inclusionFilterJar);
      shardAction.addInput(inclusionFilterJar);
    }
    ruleContext.registerAction(
        shardAction
            .addCommandLine(
                shardCommandLine.build(),
                // Classpaths can be long--overflow into @params file if necessary
                ParamFileInfo.builder(ParameterFile.ParameterFileType.SHELL_QUOTED).build())
            .build(ruleContext));

    return outputTree;
  }

  /**
   * Sets up a monodex {@code $dexmerger} actions for each dex archive in the given tree artifact
   * and returns the output tree artifact.
   *
   * @return Tree artifact containing zips with final dex files named for inclusion in an APK.
   */
  private static Artifact createTemplatedMergerActions(
      RuleContext ruleContext, SpecialArtifact inputTree, Collection<String> dexopts) {
    SpecialArtifact outputTree =
        ruleContext.getTreeArtifact(
            ruleContext.getUniqueDirectory("dexfiles"), ruleContext.getBinOrGenfilesDirectory());
    SpawnActionTemplate.Builder dexmerger =
        new SpawnActionTemplate.Builder(inputTree, outputTree)
            .setExecutable(ruleContext.getExecutablePrerequisite("$dexmerger", TransitionMode.HOST))
            .setMnemonics("DexShardsToMerge", "DexMerger")
            .setOutputPathMapper(
                (OutputPathMapper & Serializable) TreeFileArtifact::getParentRelativePath);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addPlaceholderTreeArtifactExecPath("--input", inputTree)
            .addPlaceholderTreeArtifactExecPath("--output", outputTree)
            .add("--multidex=given_shard")
            .addAll(
                DexArchiveAspect.mergerDexopts(
                    ruleContext,
                    Iterables.filter(
                        dexopts, Predicates.not(Predicates.equalTo(DX_MINIMAL_MAIN_DEX_OPTION)))));
    dexmerger.setCommandLineTemplate(commandLine.build());
    ruleContext.registerAction(dexmerger.build(ruleContext.getActionOwner()));

    return outputTree;
  }

  private static void createZipMergeAction(
      RuleContext ruleContext, Artifact inputTree, Artifact outputZip) {
    CustomCommandLine args =
        CustomCommandLine.builder()
            .add("--normalize")
            .add("--exclude_build_data")
            .add("--dont_change_compression")
            .add("--sources")
            .addExpandedTreeArtifactExecPaths(inputTree)
            .addExecPath("--output", outputZip)
            .add("--no_duplicates") // safety: expect distinct entry names in all inputs
            .build();
    // Must use params file as otherwise expanding the input tree artifact doesn't work
    Artifact paramFile =
        ruleContext.getDerivedArtifact(
            ParameterFile.derivePath(outputZip.getRootRelativePath()), outputZip.getRoot());
    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            NestedSetBuilder.create(Order.STABLE_ORDER, inputTree),
            paramFile,
            args,
            ParameterFile.ParameterFileType.SHELL_QUOTED));
    ruleContext.registerAction(
        singleJarSpawnActionBuilder(ruleContext)
            .setMnemonic("MergeDexZips")
            .setProgressMessage("Merging dex shards for %s", ruleContext.getLabel())
            .addInput(inputTree)
            .addInput(paramFile)
            .addOutput(outputZip)
            .addCommandLine(CustomCommandLine.builder().addPrefixedExecPath("@", paramFile).build())
            .build(ruleContext));
  }

  private static void createDexMergerAction(
      RuleContext ruleContext,
      String multidexStrategy,
      ImmutableList<Artifact> dexArchives,
      Artifact classesDex,
      @Nullable Artifact mainDexList,
      Collection<String> dexopts) {
    SpawnAction.Builder dexmerger =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(ruleContext.getExecutablePrerequisite("$dexmerger", TransitionMode.HOST))
            .setMnemonic("DexMerger")
            .setProgressMessage("Assembling dex files into %s", classesDex.getRootRelativePath())
            .addInputs(dexArchives)
            .addOutput(classesDex);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPaths(VectorArg.addBefore("--input").each(dexArchives))
            .addExecPath("--output", classesDex)
            .addAll(DexArchiveAspect.mergerDexopts(ruleContext, dexopts))
            .addPrefixed("--multidex=", multidexStrategy);
    if (mainDexList != null) {
      dexmerger.addInput(mainDexList);
      commandLine.addExecPath("--main-dex-list", mainDexList);
    }
    dexmerger.addCommandLine(
        commandLine.build(),
        // Classpaths can be long--overflow into @params file if necessary
        ParamFileInfo.builder(ParameterFile.ParameterFileType.SHELL_QUOTED).build());
    ruleContext.registerAction(dexmerger.build(ruleContext));
  }

  /**
   * Returns a {@link DexArchiveProvider} of all transitively generated dex archives as well as dex
   * archives for the Jars produced by the binary target itself.
   */
  public static Function<Artifact, Artifact> collectDesugaredJars(
      RuleContext ruleContext,
      AndroidCommon common,
      AndroidSemantics semantics,
      JavaTargetAttributes attributes) {
    if (!AndroidCommon.getAndroidConfig(ruleContext).desugarJava8()) {
      return Functions.identity();
    }
    AndroidRuntimeJarProvider.Builder result =
        collectDesugaredJarsFromAttributes(
            ruleContext, semantics.getAttributesWithJavaRuntimeDeps(ruleContext));
    for (Artifact jar : common.getJarsProducedForRuntime().toList()) {
      // Create dex archives next to all Jars produced by AndroidCommon for this rule.  We need to
      // do this (instead of placing dex archives into the _dx subdirectory like DexArchiveAspect)
      // because for "legacy" ResourceApks, AndroidCommon produces Jars per resource dependency that
      // can theoretically have duplicate basenames, so they go into special directories, and we
      // piggyback on that naming scheme here by placing dex archives into the same directories.
      PathFragment jarPath = jar.getRootRelativePath();
      Artifact desugared =
          DexArchiveAspect.desugar(
              ruleContext,
              jar,
              attributes.getBootClassPath().bootclasspath(),
              attributes.getCompileTimeClassPath(),
              ruleContext.getDerivedArtifact(
                  jarPath.replaceName(jarPath.getBaseName() + "_desugared.jar"), jar.getRoot()));
      result.addDesugaredJar(jar, desugared);
    }
    return result.build().collapseToFunction();
  }

  static AndroidRuntimeJarProvider.Builder collectDesugaredJarsFromAttributes(
      RuleContext ruleContext, ImmutableList<String> attributes) {
    AndroidRuntimeJarProvider.Builder result = new AndroidRuntimeJarProvider.Builder();
    for (String attr : attributes) {
      // Use all available AndroidRuntimeJarProvider from attributes that carry runtime dependencies
      result.addTransitiveProviders(
          ruleContext.getPrerequisites(
              attr, TransitionMode.TARGET, AndroidRuntimeJarProvider.class));
    }
    return result;
  }

  /**
   * Returns a {@link Map} of all transitively generated dex archives as well as dex archives for
   * the Jars produced by the binary target itself.
   */
  private static Map<Artifact, Artifact> collectDexArchives(
      RuleContext ruleContext,
      AndroidCommon common,
      List<String> dexopts,
      AndroidSemantics semantics,
      Function<Artifact, Artifact> derivedJarFunction) {
    DexArchiveProvider.Builder result = new DexArchiveProvider.Builder();
    for (String attr : semantics.getAttributesWithJavaRuntimeDeps(ruleContext)) {
      // Use all available DexArchiveProviders from attributes that carry runtime dependencies
      result.addTransitiveProviders(
          ruleContext.getPrerequisites(attr, TransitionMode.TARGET, DexArchiveProvider.class));
    }
    ImmutableSet<String> incrementalDexopts =
        DexArchiveAspect.incrementalDexopts(ruleContext, dexopts);
    for (Artifact jar : common.getJarsProducedForRuntime().toList()) {
      // Create dex archives next to all Jars produced by AndroidCommon for this rule.  We need to
      // do this (instead of placing dex archives into the _dx subdirectory like DexArchiveAspect)
      // because for "legacy" ResourceApks, AndroidCommon produces Jars per resource dependency that
      // can theoretically have duplicate basenames, so they go into special directories, and we
      // piggyback on that naming scheme here by placing dex archives into the same directories.
      PathFragment jarPath = jar.getRootRelativePath();
      Artifact dexArchive =
          DexArchiveAspect.createDexArchiveAction(
              ruleContext,
              "$dexbuilder",
              derivedJarFunction.apply(jar),
              incrementalDexopts,
              ruleContext.getDerivedArtifact(
                  jarPath.replaceName(jarPath.getBaseName() + ".dex.zip"), jar.getRoot()));
      result.addDexArchive(incrementalDexopts, dexArchive, jar);
    }
    return result.build().archivesForDexopts(incrementalDexopts);
  }

  private static Artifact createShuffleJarActions(
      RuleContext ruleContext,
      boolean makeDexArchives,
      @Nullable Artifact proguardedJar,
      ImmutableList<Artifact> shards,
      AndroidCommon common,
      @Nullable Artifact inclusionFilterJar,
      List<String> dexopts,
      AndroidSemantics semantics,
      JavaTargetAttributes attributes,
      Function<Artifact, Artifact> derivedJarFunction,
      @Nullable Artifact mainDexList)
      throws InterruptedException, RuleErrorException {
    checkArgument(!shards.isEmpty());
    checkArgument(mainDexList == null || shards.size() > 1);
    checkArgument(proguardedJar == null || inclusionFilterJar == null);

    Artifact javaResourceJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.JAVA_RESOURCES_JAR);
    ImmutableList<Artifact> shuffleOutputs;
    if (makeDexArchives && proguardedJar != null) {
      checkArgument(shards.size() > 1);
      // Split proguardedJar into N shards and run dexbuilder over each one below
      shuffleOutputs = makeShardArtifacts(ruleContext, shards.size(), ".jar");
    } else {
      shuffleOutputs = shards;
    }

    SpawnAction.Builder shardAction =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setMnemonic("ShardClassesToDex")
            .setProgressMessage("Sharding classes for dexing for %s", ruleContext.getLabel())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$shuffle_jars", TransitionMode.HOST))
            .addOutputs(shuffleOutputs)
            .addOutput(javaResourceJar);

    CustomCommandLine.Builder shardCommandLine =
        CustomCommandLine.builder()
            .addExecPaths(VectorArg.addBefore("--output_jar").each(shuffleOutputs))
            .addExecPath("--output_resources", javaResourceJar);

    if (mainDexList != null) {
      shardCommandLine.addExecPath("--main_dex_filter", mainDexList);
      shardAction.addInput(mainDexList);
    }

    // If we need to run Proguard, all the class files will be in the Proguarded jar, which has to
    // be converted to dex. Otherwise we can use the transitive classpath directly and can leverage
    // incremental dexing outputs for classpath Jars if applicable.
    if (proguardedJar != null) {
      shardCommandLine.addExecPath("--input_jar", proguardedJar);
      shardAction.addInput(proguardedJar);
    } else {
      ImmutableList<Artifact> classpath = collectRuntimeJars(common, attributes);
      // Check whether we can use dex archives.  Besides the --incremental_dexing flag, also
      // make sure the "dexopts" attribute on this target doesn't mention any problematic flags.
      if (makeDexArchives) {
        // Use dex archives instead of their corresponding Jars wherever we can.  At this point
        // there should be very few or no Jar files that still end up in shards.  The dexing
        // step below will have to deal with those in addition to merging .dex files together.
        Map<Artifact, Artifact> dexArchives =
            collectDexArchives(ruleContext, common, dexopts, semantics, derivedJarFunction);
        classpath = toDexedClasspath(ruleContext, classpath, dexArchives);
        shardCommandLine.add("--split_dexed_classes");
      } else {
        classpath = classpath.stream().map(derivedJarFunction).collect(toImmutableList());
      }
      shardCommandLine.addExecPaths(VectorArg.addBefore("--input_jar").each(classpath));
      shardAction.addInputs(classpath);

      if (inclusionFilterJar != null) {
        shardCommandLine.addExecPath("--inclusion_filter_jar", inclusionFilterJar);
        shardAction.addInput(inclusionFilterJar);
      }
    }

    shardAction.addCommandLine(
        shardCommandLine.build(), ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).build());
    ruleContext.registerAction(shardAction.build(ruleContext));

    if (makeDexArchives && proguardedJar != null) {
      for (int i = 0; i < shards.size(); ++i) {
        checkState(!shuffleOutputs.get(i).equals(shards.get(i)));
        DexArchiveAspect.createDexArchiveAction(
            ruleContext,
            "$dexbuilder_after_proguard",
            shuffleOutputs.get(i),
            DexArchiveAspect.topLevelDexbuilderDexopts(dexopts),
            shards.get(i));
      }
    }
    return javaResourceJar;
  }

  private static ImmutableList<Artifact> collectRuntimeJars(
      AndroidCommon common, JavaTargetAttributes attributes) {
    return ImmutableList.<Artifact>builder()
        .addAll(common.getRuntimeJars())
        .addAll(attributes.getRuntimeClassPathForArchive().toList())
        .build();
  }

  private static ImmutableList<Artifact> toDexedClasspath(
      RuleContext ruleContext,
      ImmutableList<Artifact> classpath,
      Map<Artifact, Artifact> dexArchives)
      throws RuleErrorException {
    // This is a simple Iterables.transform but with useful error message in case of missed Jars.
    ImmutableList.Builder<Artifact> dexedClasspath = ImmutableList.builder();
    for (Artifact jar : classpath) {
      Artifact dexArchive = dexArchives.get(jar);
      if (dexArchive == null) {
        // Users can create this situation by directly depending on a .jar artifact (checked in
        // or coming from a genrule or similar, b/11285003).  This will also catch new  implicit
        // dependencies that incremental dexing would need to be extended to (b/34949364).
        // Typically the fix for the latter involves propagating DexArchiveAspect along the
        // attribute defining the new implicit dependency.
        ruleContext.throwWithAttributeError(
            "deps",
            "Dependencies on .jar artifacts are not "
                + "allowed in Android binaries, please use a java_import to depend on "
                + jar.prettyPrint()
                + ". If this is an implicit dependency then the rule that "
                + "introduces it will need to be fixed to account for it correctly.");
      }
      dexedClasspath.add(dexArchive);
    }
    return dexedClasspath.build();
  }

  // Adds the appropriate SpawnAction options depending on if SingleJar is a jar or not.
  private static SpawnAction.Builder singleJarSpawnActionBuilder(RuleContext ruleContext) {
    Artifact singleJar = JavaToolchainProvider.from(ruleContext).getSingleJar();
    SpawnAction.Builder builder = new SpawnAction.Builder().useDefaultShellEnvironment();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              JavaToolchainProvider.from(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaRuntimeInfo.forHost(ruleContext).javaBaseInputsMiddleman());
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }

  /**
   * Creates an action that copies a .zip file to a specified path, filtering all non-.dex files out
   * of the output.
   */
  static void createCleanDexZipAction(
      RuleContext ruleContext, Artifact inputZip, Artifact outputZip) {
    ruleContext.registerAction(
        singleJarSpawnActionBuilder(ruleContext)
            .setProgressMessage("Trimming %s", inputZip.getExecPath().getBaseName())
            .setMnemonic("TrimDexZip")
            .addInput(inputZip)
            .addOutput(outputZip)
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("--exclude_build_data")
                    .add("--dont_change_compression")
                    .addExecPath("--sources", inputZip)
                    .addExecPath("--output", outputZip)
                    .add("--include_prefixes")
                    .add("classes")
                    .build())
            .build(ruleContext));
  }

  /**
   * Creates an action that generates a list of classes to be passed to the dexer's --main-dex-list
   * flag (which specifies the classes that need to be directly in classes.dex). Returns the file
   * containing the list.
   */
  static Artifact createMainDexListAction(
      RuleContext ruleContext,
      AndroidSemantics androidSemantics,
      Artifact jar,
      @Nullable Artifact mainDexProguardSpec,
      @Nullable Artifact proguardOutputMap)
      throws InterruptedException {
    AndroidConfiguration config = AndroidCommon.getAndroidConfig(ruleContext);
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    // Create the main dex classes list.
    Artifact mainDexList = AndroidBinary.getDxArtifact(ruleContext, "main_dex_list.txt");

    List<Artifact> proguardSpecs = new ArrayList<>();
    proguardSpecs.addAll(
        ruleContext
            .getPrerequisiteArtifacts("main_dex_proguard_specs", TransitionMode.TARGET)
            .list());
    if (proguardSpecs.isEmpty()) {
      proguardSpecs.add(sdk.getMainDexClasses());
    }
    if (mainDexProguardSpec != null) {
      proguardSpecs.add(mainDexProguardSpec);
    }

    // If --legacy_main_dex_list_generator is not set, use ProGuard and the main dext list creator
    // specified by the android_sdk rule. If --legacy_main_dex_list_generator is provided, use that
    // tool instead.
    // TODO(b/147692286): Remove the old main-dex list generation that relied on ProGuard.
    if (config.getLegacyMainDexListGenerator() == null) {
      // Process the input jar through Proguard into an intermediate, streamlined jar.
      Artifact strippedJar = AndroidBinary.getDxArtifact(ruleContext, "main_dex_intermediate.jar");
      SpawnAction.Builder streamlinedBuilder =
          new SpawnAction.Builder()
              .useDefaultShellEnvironment()
              .addOutput(strippedJar)
              .setExecutable(sdk.getProguard())
              .setProgressMessage("Generating streamlined input jar for main dex classes list")
              .setMnemonic("MainDexClassesIntermediate")
              .addInput(jar)
              .addInput(sdk.getShrinkedAndroidJar());
      CustomCommandLine.Builder streamlinedCommandLine =
          CustomCommandLine.builder()
              .add("-forceprocessing")
              .addExecPath("-injars", jar)
              .addExecPath("-libraryjars", sdk.getShrinkedAndroidJar())
              .addExecPath("-outjars", strippedJar)
              .add("-dontwarn")
              .add("-dontnote")
              .add("-dontoptimize")
              .add("-dontobfuscate");

      for (Artifact spec : proguardSpecs) {
        streamlinedBuilder.addInput(spec);
        streamlinedCommandLine.addExecPath("-include", spec);
      }

      androidSemantics.addMainDexListActionArguments(
          ruleContext, streamlinedBuilder, streamlinedCommandLine, proguardOutputMap);

      streamlinedBuilder.addCommandLine(streamlinedCommandLine.build());
      ruleContext.registerAction(streamlinedBuilder.build(ruleContext));

      SpawnAction.Builder builder =
          new SpawnAction.Builder()
              .setMnemonic("MainDexClasses")
              .setProgressMessage("Generating main dex classes list");

      ruleContext.registerAction(
          builder
              .setExecutable(sdk.getMainDexListCreator())
              .addOutput(mainDexList)
              .addInput(strippedJar)
              .addInput(jar)
              .addCommandLine(
                  CustomCommandLine.builder()
                      .addExecPath(mainDexList)
                      .addExecPath(strippedJar)
                      .addExecPath(jar)
                      .addAll(
                          ruleContext
                              .getExpander()
                              .withDataLocations()
                              .tokenized("main_dex_list_opts"))
                      .build())
              .build(ruleContext));
    } else {
      FilesToRunProvider legacyMainDexListGenerator =
          ruleContext.getExecutablePrerequisite(
              ":legacy_main_dex_list_generator", TransitionMode.HOST);
      // Use the newer legacy multidex main-dex list generation.
      SpawnAction.Builder actionBuilder =
          new SpawnAction.Builder()
              .setMnemonic("MainDexClasses")
              .setProgressMessage("Generating main dex classes list");

      CustomCommandLine.Builder commandLineBuilder =
          CustomCommandLine.builder()
              .addExecPath("--main-dex-list-output", mainDexList)
              .addExecPath("--lib", sdk.getAndroidJar());
      if (AndroidCommon.getAndroidConfig(ruleContext).desugarJava8Libs()) {
        NestedSet<Artifact> legacyApis =
            ruleContext
                .getPrerequisite("$desugared_java8_legacy_apis", TransitionMode.TARGET)
                .getProvider(FileProvider.class)
                .getFilesToBuild();
        for (Artifact lib : legacyApis.toList()) {
          actionBuilder.addInput(lib);
          commandLineBuilder.addExecPath("--lib", lib);
        }
      }
      for (Artifact spec : proguardSpecs) {
        actionBuilder.addInput(spec);
        commandLineBuilder.addExecPath("--main-dex-rules", spec);
      }

      commandLineBuilder.addExecPath(jar);

      ruleContext.registerAction(
          actionBuilder
              .setExecutable(legacyMainDexListGenerator)
              .addOutput(mainDexList)
              .addInput(jar)
              .addInput(sdk.getAndroidJar())
              .addCommandLine(commandLineBuilder.build())
              .build(ruleContext));
    }
    return mainDexList;
  }

  /** Transforms manual main_dex_list through proguard obfuscation map. */
  static Artifact transformDexListThroughProguardMapAction(
      RuleContext ruleContext, @Nullable Artifact proguardOutputMap, Artifact mainDexList)
      throws InterruptedException {
    if (proguardOutputMap == null
        || !ruleContext.attributes().get("proguard_generate_mapping", Type.BOOLEAN)) {
      return mainDexList;
    }
    Artifact obfuscatedMainDexList =
        AndroidBinary.getDxArtifact(ruleContext, "main_dex_list_obfuscated.txt");
    SpawnAction.Builder actionBuilder =
        new SpawnAction.Builder()
            .setMnemonic("MainDexProguardClasses")
            .setProgressMessage("Obfuscating main dex classes list")
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$dex_list_obfuscator", TransitionMode.HOST))
            .addInput(mainDexList)
            .addInput(proguardOutputMap)
            .addOutput(obfuscatedMainDexList)
            .addCommandLine(
                CustomCommandLine.builder()
                    .addExecPath("--input", mainDexList)
                    .addExecPath("--output", obfuscatedMainDexList)
                    .addExecPath("--obfuscation_map", proguardOutputMap)
                    .build());
    ruleContext.registerAction(actionBuilder.build(ruleContext));
    return obfuscatedMainDexList;
  }

  public static Artifact createMainDexProguardSpec(Label label, ActionConstructionContext context) {
    return ProguardHelper.getProguardConfigArtifact(label, context, "main_dex");
  }

  /** Returns the multidex mode to apply to this target. */
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
   * set with an old SDK, we will exit with an error to alert the developer that their application
   * might not run on devices that the used SDK still supports.
   */
  private static final ImmutableSet<String> RUNTIMES_THAT_DONT_SUPPORT_NATIVE_MULTIDEXING =
      ImmutableSet.of(
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

  /** Returns an intermediate artifact used to support dex generation. */
  public static Artifact getDxArtifact(RuleContext ruleContext, String baseName) {
    return ruleContext.getUniqueDirectoryArtifact(
        "_dx", baseName, ruleContext.getBinOrGenfilesDirectory());
  }

  /** Returns true if this android_binary target is an instrumentation binary */
  private static boolean isInstrumentation(RuleContext ruleContext) {
    return ruleContext.attributes().isAttributeValueExplicitlySpecified("instruments");
  }

  /**
   * Perform class filtering using the target APK's predexed JAR. Filter duplicate .class and
   * R.class files based on name. Prevents runtime crashes on ART. See b/19713845 for details.
   */
  private static Artifact getFilteredDeployJar(RuleContext ruleContext, Artifact deployJar)
      throws InterruptedException {
    Artifact filterJar =
        ruleContext
            .getPrerequisite("instruments", TransitionMode.TARGET)
            .get(AndroidPreDexJarProvider.PROVIDER)
            .getPreDexJar();
    Artifact filteredDeployJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_TEST_FILTERED_JAR);
    AndroidCommon.createZipFilterAction(
        ruleContext,
        deployJar,
        filterJar,
        filteredDeployJar,
        CheckHashMismatchMode.NONE,
        ruleContext
            .getFragment(AndroidConfiguration.class)
            .removeRClassesFromInstrumentationTestJar());
    return filteredDeployJar;
  }
}
