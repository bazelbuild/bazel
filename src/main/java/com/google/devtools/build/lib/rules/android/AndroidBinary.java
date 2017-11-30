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
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidBinaryMobileInstall.MobileInstallResourceApks;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.OneVersionCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.ProguardHelper;
import com.google.devtools.build.lib.rules.java.ProguardHelper.ProguardOutput;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * An implementation for the "android_binary" rule.
 */
public abstract class AndroidBinary implements RuleConfiguredTargetFactory {

  private static final String DX_MINIMAL_MAIN_DEX_OPTION = "--minimal-main-dex";

  protected abstract JavaSemantics createJavaSemantics();
  protected abstract AndroidSemantics createAndroidSemantics();

  protected abstract CppSemantics createCppSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    CppSemantics cppSemantics = createCppSemantics();
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    JavaCommon javaCommon =
        AndroidCommon.createJavaCommonWithAndroidDataBinding(ruleContext, javaSemantics, false);
    javaSemantics.checkRule(ruleContext, javaCommon);
    javaSemantics.checkForProtoLibraryAndJavaProtoLibraryOnSameProto(ruleContext, javaCommon);

    AndroidCommon androidCommon =
        new AndroidCommon(javaCommon, /* asNeverLink= */ true, /* exportDeps= */ true);
    ResourceDependencies resourceDeps =
        LocalResourceContainer.definesAndroidResources(ruleContext.attributes())
            ? ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink= */ false)
            : ResourceDependencies.fromRuleResourceAndDeps(ruleContext, /* neverlink= */ false);
    RuleConfiguredTargetBuilder builder =
        init(
            ruleContext,
            filesBuilder,
            resourceDeps,
            androidCommon,
            cppSemantics,
            javaSemantics,
            androidSemantics);
    return builder.build();
  }

  /**
   * Checks expected rule invariants, throws rule errors if anything is set wrong.
   */
  private static void validateRuleContext(RuleContext ruleContext) throws RuleErrorException {
    if (getMultidexMode(ruleContext) != MultidexMode.LEGACY
        && ruleContext.attributes().isAttributeValueExplicitlySpecified(
            "main_dex_proguard_specs")) {
      ruleContext.throwWithAttributeError("main_dex_proguard_specs", "The "
          + "'main_dex_proguard_specs' attribute is only allowed if 'multidex' is set to 'legacy'");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("proguard_apply_mapping")
        && ruleContext.attributes()
            .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
            .isEmpty()) {
      ruleContext.throwWithAttributeError("proguard_apply_mapping",
          "'proguard_apply_mapping' can only be used when 'proguard_specs' is also set");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("proguard_apply_dictionary")
        && ruleContext.attributes()
            .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
            .isEmpty()) {
      ruleContext.throwWithAttributeError("proguard_apply_mapping",
          "'proguard_apply_dictionary' can only be used when 'proguard_specs' is also set");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("rex_package_map")
        && !ruleContext.attributes().get("rewrite_dexes_with_rex", Type.BOOLEAN)) {
      ruleContext.throwWithAttributeError(
          "rex_package_map",
          "'rex_package_map' can only be used when 'rewrite_dexes_with_rex' is also set");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("rex_package_map")
        && ruleContext.attributes()
        .get(ProguardHelper.PROGUARD_SPECS, BuildType.LABEL_LIST)
        .isEmpty()) {
      ruleContext.throwWithAttributeError("rex_package_map",
          "'rex_package_map' can only be used when 'proguard_specs' is also set");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("rexopts")
        && !ruleContext.attributes().get("rewrite_dexes_with_rex", Type.BOOLEAN)) {
      ruleContext.throwWithAttributeError(
          "rexopts", "'rexopts' can only be used when 'rewrite_dexes_with_rex' is also set");
    }
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("resources")
      && DataBinding.isEnabled(ruleContext)) {
      ruleContext.throwWithRuleError("Data binding doesn't work with the \"resources\" attribute. "
          + "Use \"resource_files\" instead.");
    }
    AndroidCommon.validateResourcesAttribute(ruleContext);
  }

  private static RuleConfiguredTargetBuilder init(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      ResourceDependencies resourceDeps,
      AndroidCommon androidCommon,
      CppSemantics cppSemantics,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics)
      throws InterruptedException, RuleErrorException {

    validateRuleContext(ruleContext);

    // TODO(bazel-team): Find a way to simplify this code.
    // treeKeys() means that the resulting map sorts the entries by key, which is necessary to
    // ensure determinism.
    Multimap<String, TransitiveInfoCollection> depsByArchitecture =
        MultimapBuilder.treeKeys().arrayListValues().build();
    AndroidConfiguration androidConfig = ruleContext.getFragment(AndroidConfiguration.class);
    for (Map.Entry<Optional<String>, ? extends List<? extends TransitiveInfoCollection>> entry :
        ruleContext.getSplitPrerequisites("deps").entrySet()) {
      String cpu = entry.getKey().or(androidConfig.getCpu());
      depsByArchitecture.putAll(cpu, entry.getValue());
    }
    Map<String, BuildConfiguration> configurationMap = new LinkedHashMap<>();
    Map<String, CcToolchainProvider> toolchainMap = new LinkedHashMap<>();
    for (Map.Entry<Optional<String>, ? extends List<? extends TransitiveInfoCollection>> entry :
        ruleContext.getSplitPrerequisites(":cc_toolchain_split").entrySet()) {
      String cpu = entry.getKey().or(androidConfig.getCpu());
      TransitiveInfoCollection dep = Iterables.getOnlyElement(entry.getValue());
      CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext, dep);
      configurationMap.put(cpu, dep.getConfiguration());
      toolchainMap.put(cpu, toolchain);
    }

    NativeLibs nativeLibs =
        NativeLibs.fromLinkedNativeDeps(
            ruleContext,
            androidSemantics.getNativeDepsFileName(),
            depsByArchitecture,
            toolchainMap,
            configurationMap,
            cppSemantics);

    boolean shrinkResources = shouldShrinkResources(ruleContext);

    // TODO(bazel-team): Resolve all the different cases of resource handling so this conditional
    // can go away: recompile from android_resources, and recompile from android_binary attributes.
    ApplicationManifest applicationManifest;
    ResourceApk resourceApk;
    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())) {
      // Retrieve and compile the resources defined on the android_binary rule.
      LocalResourceContainer.validateRuleContext(ruleContext);
      ApplicationManifest ruleManifest = androidSemantics.getManifestForRule(ruleContext);

      applicationManifest = ruleManifest.mergeWith(ruleContext, resourceDeps);

      Artifact featureOfArtifact =
          ruleContext.attributes().isAttributeValueExplicitlySpecified("feature_of")
              ? ruleContext.getPrerequisite("feature_of", Mode.TARGET, ApkProvider.class).getApk()
              : null;
      Artifact featureAfterArtifact =
          ruleContext.attributes().isAttributeValueExplicitlySpecified("feature_after")
              ? ruleContext
                  .getPrerequisite("feature_after", Mode.TARGET, ApkProvider.class)
                  .getApk()
              : null;

      resourceApk =
          applicationManifest.packBinaryWithDataAndResources(
              ruleContext,
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
              resourceDeps,
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
              ResourceFilterFactory.fromRuleContext(ruleContext),
              ruleContext.getExpander().withDataLocations().tokenized("nocompress_extensions"),
              ruleContext.attributes().get("crunch_png", Type.BOOLEAN),
              ProguardHelper.getProguardConfigArtifact(ruleContext, ""),
              createMainDexProguardSpec(ruleContext),
              shouldShrinkResourceCycles(ruleContext, shrinkResources),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
              ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP),
              DataBinding.isEnabled(ruleContext)
                  ? DataBinding.getLayoutInfoFile(ruleContext)
                  : null,
              featureOfArtifact,
              featureAfterArtifact);
      ruleContext.assertNoErrors();

    } else {

      if (!ruleContext.attributes().get("crunch_png", Type.BOOLEAN)) {
        ruleContext.throwWithRuleError("Setting crunch_png = 0 is not supported for android_binary"
            + " rules which depend on android_resources rules.");
      }

      // Retrieve the resources from the resources attribute on the android_binary rule
      // and recompile them if necessary.
      ApplicationManifest resourcesManifest = ApplicationManifest.fromResourcesRule(ruleContext);
      if (resourcesManifest == null) {
        throw new RuleErrorException();
      }
      applicationManifest = resourcesManifest.mergeWith(ruleContext, resourceDeps);

      // Always recompiling resources causes AndroidTest to fail in certain circumstances.
      if (shouldRegenerate(ruleContext, resourceDeps)) {
        resourceApk = applicationManifest.packWithResources(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
            ruleContext,
            resourceDeps,
            true, /* createSource */
            ProguardHelper.getProguardConfigArtifact(ruleContext, ""),
            createMainDexProguardSpec(ruleContext));
        ruleContext.assertNoErrors();
      } else {
        resourceApk = applicationManifest.useCurrentResources(
            ruleContext,
            ProguardHelper.getProguardConfigArtifact(ruleContext, ""),
            createMainDexProguardSpec(ruleContext));
        ruleContext.assertNoErrors();
      }

    }

    // Remove the library resource JARs from the binary's runtime classpath.
    // Resource classes from android_library dependencies are replaced by the binary's resource
    // class. We remove them only at the top level so that resources included by a library that is
    // a dependency of a java_library are still included, since these resources are propagated via
    // android-specific providers and won't show up when we collect the library resource JARs.
    // TODO(b/69552500): Instead, handle this properly so R JARs aren't put on the classpath for
    // both binaries and libraries.
    NestedSet<Artifact> excludedRuntimeArtifacts = getLibraryResourceJars(ruleContext);

    JavaTargetAttributes resourceClasses = androidCommon.init(
        javaSemantics,
        androidSemantics,
        resourceApk,
        ruleContext.getConfiguration().isCodeCoverageEnabled(),
        true /* collectJavaCompilationArgs */,
        true, /* isBinary */
        excludedRuntimeArtifacts);
    ruleContext.assertNoErrors();

    Function<Artifact, Artifact> derivedJarFunction =
        collectDesugaredJars(ruleContext, androidCommon, androidSemantics, resourceClasses);
    Artifact deployJar = createDeployJar(ruleContext, javaSemantics, androidCommon, resourceClasses,
        AndroidCommon.getAndroidConfig(ruleContext).checkDesugarDeps(), derivedJarFunction);

    OneVersionEnforcementLevel oneVersionEnforcementLevel =
        ruleContext.getFragment(JavaConfiguration.class).oneVersionEnforcementLevel();
    Artifact oneVersionOutputArtifact = null;
    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF) {
      NestedSet<Artifact> transitiveDependencies =
          NestedSetBuilder.<Artifact>stableOrder()
              .addAll(
                  Iterables.transform(resourceClasses.getRuntimeClassPath(), derivedJarFunction))
              .addAll(
                  Iterables.transform(
                      androidCommon.getJarsProducedForRuntime(), derivedJarFunction))
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

    Artifact proguardMapping = ruleContext.getPrerequisiteArtifact(
        "proguard_apply_mapping", Mode.TARGET);
    Artifact proguardDictionary = ruleContext.getPrerequisiteArtifact(
        "proguard_apply_dictionary", Mode.TARGET);

    MobileInstallResourceApks mobileInstallResourceApks =
        AndroidBinaryMobileInstall.createMobileInstallResourceApks(
            ruleContext,
            applicationManifest,
            resourceDeps);

    return createAndroidBinary(
        ruleContext,
        filesBuilder,
        deployJar,
        derivedJarFunction,
        /* isBinaryJarFiltered */ false,
        androidCommon,
        javaSemantics,
        androidSemantics,
        nativeLibs,
        applicationManifest,
        resourceApk,
        mobileInstallResourceApks,
        shrinkResources,
        resourceClasses,
        ImmutableList.<Artifact>of(),
        ImmutableList.<Artifact>of(),
        proguardMapping,
        proguardDictionary,
        oneVersionOutputArtifact);
  }

  public static RuleConfiguredTargetBuilder createAndroidBinary(
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> filesBuilder,
      Artifact binaryJar,
      Function<Artifact, Artifact> derivedJarFunction,
      boolean isBinaryJarFiltered,
      AndroidCommon androidCommon,
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics,
      NativeLibs nativeLibs,
      ApplicationManifest applicationManifest,
      ResourceApk resourceApk,
      @Nullable MobileInstallResourceApks mobileInstallResourceApks,
      boolean shrinkResources,
      JavaTargetAttributes resourceClasses,
      ImmutableList<Artifact> apksUnderTest,
      ImmutableList<Artifact> additionalMergedManifests,
      Artifact proguardMapping,
      Artifact proguardDictionary,
      @Nullable Artifact oneVersionEnforcementArtifact)
      throws InterruptedException, RuleErrorException {

    ImmutableList<Artifact> proguardSpecs = ProguardHelper.collectTransitiveProguardSpecs(
        ruleContext, ImmutableList.of(resourceApk.getResourceProguardConfig()));

    boolean assumeMinSdkVersion =
        ruleContext.getFragment(AndroidConfiguration.class).assumeMinSdkVersion();
    if (!proguardSpecs.isEmpty() && assumeMinSdkVersion) {
      // NB: Order here is important. We're including generated Proguard specs before the user's
      // specs so that they can override values.
      proguardSpecs =
          ImmutableList.<Artifact>builder()
              .addAll(
                  androidSemantics.getProguardSpecsForManifest(
                      ruleContext, applicationManifest.getManifest()))
              .addAll(proguardSpecs)
              .build();
    }

    boolean rexEnabled =
        ruleContext.getFragment(AndroidConfiguration.class).useRexToCompressDexFiles()
        || (ruleContext.attributes().get("rewrite_dexes_with_rex", Type.BOOLEAN));

    // TODO(bazel-team): Verify that proguard spec files don't contain -printmapping directions
    // which this -printmapping command line flag will override.
    Artifact proguardOutputMap = null;
    if (ProguardHelper.genProguardMapping(ruleContext.attributes())
        || ProguardHelper.getJavaOptimizationMode(ruleContext).alwaysGenerateOutputMapping()
        || shrinkResources) {
      if (rexEnabled) {
        proguardOutputMap = ProguardHelper.getProguardTempArtifact(ruleContext,
            ProguardHelper.getJavaOptimizationMode(ruleContext).name().toLowerCase(),
            "proguard_output_for_rex.map");
      } else {
        proguardOutputMap =
            ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
      }
    }

    ProguardOutput proguardOutput =
        applyProguard(
            ruleContext,
            androidCommon,
            javaSemantics,
            binaryJar,
            proguardSpecs,
            proguardMapping,
            proguardDictionary,
            proguardOutputMap);

    if (shrinkResources) {
      resourceApk = shrinkResources(
          ruleContext,
          resourceApk,
          proguardSpecs,
          proguardOutput,
          filesBuilder);
    }

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

    NestedSet<Artifact> nativeLibsZips =
        AndroidCommon.collectTransitiveNativeLibsZips(ruleContext).build();

    Artifact finalDexes;
    Artifact finalProguardMap;
    if (rexEnabled) {
      finalDexes = getDxArtifact(ruleContext, "rexed_dexes.zip");
      SpawnAction.Builder rexActionBuilder = new SpawnAction.Builder();
      CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
      rexActionBuilder
          .useDefaultShellEnvironment()
          .setExecutable(ruleContext.getExecutablePrerequisite("$rex_wrapper", Mode.HOST))
          .setMnemonic("Rex")
          .setProgressMessage("Rexing dex files")
          .addInput(dexingOutput.classesDexZip)
          .addOutput(finalDexes);
      commandLine
          .addExecPath("--dex_input", dexingOutput.classesDexZip)
          .addExecPath("--dex_output", finalDexes);
      if (proguardOutput.getMapping() != null) {
        finalProguardMap =
            ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
        Artifact finalRexPackageMap =
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.REX_OUTPUT_PACKAGE_MAP);
        rexActionBuilder
            .addInput(proguardOutput.getMapping())
            .addOutput(finalProguardMap)
            .addOutput(finalRexPackageMap);
        filesBuilder.add(finalRexPackageMap);
        commandLine
            .addExecPath("--proguard_input_map", proguardOutput.getMapping())
            .addExecPath("--proguard_output_map", finalProguardMap)
            .addExecPath("--rex_output_package_map", finalRexPackageMap);
        if (ruleContext.attributes().isAttributeValueExplicitlySpecified("rex_package_map")) {
          Artifact rexPackageMap =
              ruleContext.getPrerequisiteArtifact("rex_package_map", Mode.TARGET);
          rexActionBuilder.addInput(rexPackageMap);
          commandLine.addExecPath("--rex_input_package_map", rexPackageMap);
        }
      } else {
        finalProguardMap = proguardOutput.getMapping();
      }
      // the Rex flag --keep-main-dex is used to support builds with API level below 21 that do not
      // support native multi-dex. This flag indicates to Rex to use the main_dex_list file which
      // can be provided by the user via the main_dex_list attribute or created automatically
      // when multidex mode is set to legacy.
      if (ruleContext.attributes().isAttributeValueExplicitlySpecified("main_dex_list")
          || getMultidexMode(ruleContext) == MultidexMode.LEGACY) {
        commandLine.add("--keep-main-dex");
      }
      // Pass rexopts to rex as a list of strings without validation
      if (ruleContext.attributes().isAttributeValueExplicitlySpecified("rexopts")) {
        List<String> rexopts = ruleContext.getExpander().withDataLocations().tokenized("rexopts");
        commandLine.addAll(rexopts);
      }
      rexActionBuilder.addCommandLine(commandLine.build());
      ruleContext.registerAction(rexActionBuilder.build(ruleContext));
    } else {
      finalDexes = dexingOutput.classesDexZip;
      finalProguardMap = proguardOutput.getMapping();
    }

    if (!proguardSpecs.isEmpty()) {
      proguardOutput.addAllToSet(filesBuilder, finalProguardMap);
    }

    Artifact unsignedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_UNSIGNED_APK);
    Artifact zipAlignedApk =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_APK);
    Artifact signingKey = AndroidCommon.getApkDebugSigningKey(ruleContext);
    FilesToRunProvider resourceExtractor =
        ruleContext.getExecutablePrerequisite("$resource_extractor", Mode.HOST);

    ApkActionsBuilder.create("apk")
        .setClassesDex(finalDexes)
        .addInputZip(resourceApk.getArtifact())
        .setJavaResourceZip(dexingOutput.javaResourceJar, resourceExtractor)
        .addInputZips(nativeLibsZips)
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

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);

    // If this is an instrumentation APK, create the provider for android_instrumentation_test.
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("instruments")) {
      Artifact targetApk =
          ruleContext
              .getPrerequisite("instruments", Mode.TARGET)
              .getProvider(ApkProvider.class)
              .getApk();
      Artifact instrumentationApk = zipAlignedApk;

      AndroidInstrumentationInfo instrumentationProvider =
          new AndroidInstrumentationInfo(targetApk, instrumentationApk);

      builder.addNativeDeclaredProvider(instrumentationProvider);
    }

    androidCommon.addTransitiveInfoProviders(
        builder,
        androidSemantics,
        null /* aar */,
        resourceApk,
        zipAlignedApk,
        apksUnderTest,
        nativeLibs,
        /* isResourcesOnly = */ false);

    if (proguardOutput.getMapping() != null) {
      builder.add(
          ProguardMappingProvider.class,
          ProguardMappingProvider.create(finalProguardMap));
    }

    if (oneVersionEnforcementArtifact != null) {
      builder.addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, oneVersionEnforcementArtifact);
    }

    if (mobileInstallResourceApks != null) {
      AndroidBinaryMobileInstall.addMobileInstall(
          ruleContext,
          builder,
          dexingOutput,
          javaSemantics,
          nativeLibs,
          resourceApk,
          mobileInstallResourceApks,
          resourceExtractor,
          nativeLibsZips,
          signingKey,
          additionalMergedManifests,
          applicationManifest);
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
        .addProvider(
            ApkProvider.class,
            ApkProvider.create(
                zipAlignedApk,
                unsignedApk,
                androidCommon.getInstrumentedJar(),
                applicationManifest.getManifest(),
                AndroidCommon.getApkDebugSigningKey(ruleContext)))
        .addProvider(AndroidPreDexJarProvider.class, AndroidPreDexJarProvider.create(jarToDex))
        .addProvider(
            AndroidFeatureFlagSetProvider.class,
            AndroidFeatureFlagSetProvider.create(
                AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext)))
        .addOutputGroup("android_deploy_info", deployInfo);
  }

  private static NestedSet<Artifact> getLibraryResourceJars(RuleContext ruleContext) {
    Iterable<AndroidLibraryResourceClassJarProvider> libraryResourceJarProviders =
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, Mode.TARGET, AndroidLibraryResourceClassJarProvider.class);

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
      JavaSemantics javaSemantics,
      Artifact deployJarArtifact,
      ImmutableList<Artifact> proguardSpecs,
      Artifact proguardMapping,
      Artifact proguardDictionary,
      @Nullable Artifact proguardOutputMap) throws InterruptedException {
    Artifact proguardOutputJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_BINARY_PROGUARD_JAR);

    // Proguard will be only used for binaries which specify a proguard_spec
    if (proguardSpecs.isEmpty()) {
      // Although normally the Proguard jar artifact is not needed for binaries which do not specify
      // proguard_specs, targets which use a select to provide an empty list to proguard_specs will
      // still have a Proguard jar implicit output, as it is impossible to tell what a select will
      // produce at the time of implicit output determination. As a result, this artifact must
      // always be created.
      return createEmptyProguardAction(ruleContext, javaSemantics, proguardOutputJar,
                                       deployJarArtifact, proguardOutputMap);
    }

    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    NestedSet<Artifact> libraryJars = NestedSetBuilder.<Artifact>naiveLinkOrder()
        .add(sdk.getAndroidJar())
        .addTransitive(common.getTransitiveNeverLinkLibraries())
        .build();
    Artifact proguardSeeds =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_SEEDS);
    Artifact proguardUsage =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_USAGE);
    ProguardOutput result = ProguardHelper.createOptimizationActions(
        ruleContext,
        sdk.getProguard(),
        deployJarArtifact,
        proguardSpecs,
        proguardSeeds,
        proguardUsage,
        proguardMapping,
        proguardDictionary,
        libraryJars,
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

  private static ProguardOutput createEmptyProguardAction(RuleContext ruleContext,
      JavaSemantics semantics, Artifact proguardOutputJar, Artifact deployJarArtifact,
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
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    ruleContext.registerAction(
        new FailAction(
            ruleContext.getActionOwner(),
            failures.build(),
            String.format("Can't run Proguard %s",
                optMode == JavaOptimizationMode.LEGACY
                    ? "without proguard_specs"
                    : "in optimization mode " + optMode)));
    return new ProguardOutput(deployJarArtifact, null, null, null, null, null, null);
  }

  /** Returns {@code true} if resource shrinking should be performed. */
  private static boolean shouldShrinkResources(RuleContext ruleContext) {
    TriState state = ruleContext.attributes().get("shrink_resources", BuildType.TRISTATE);
    if (state == TriState.AUTO) {
      boolean globalShrinkResources =
          ruleContext.getFragment(AndroidConfiguration.class).useAndroidResourceShrinking();
      state = (globalShrinkResources) ? TriState.YES : TriState.NO;
    }

    return (state == TriState.YES);
  }

  /** Returns {@code true} if resource shrinking should be performed. */
  private static boolean shouldShrinkResourceCycles(
      RuleContext ruleContext, boolean shrinkResources) throws RuleErrorException {
    boolean global =
        ruleContext.getFragment(AndroidConfiguration.class).useAndroidResourceCycleShrinking();
    if (global && !shrinkResources) {
      throw ruleContext.throwWithRuleError(
          "resource cycle shrinking can only be enabled when resource shrinking is enabled");
    }
    return global;
  }

  private static ResourceApk shrinkResources(
      RuleContext ruleContext,
      ResourceApk resourceApk,
      ImmutableList<Artifact> proguardSpecs,
      ProguardOutput proguardOutput,
      NestedSetBuilder<Artifact> filesBuilder) throws InterruptedException, RuleErrorException {

    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())
        && !proguardSpecs.isEmpty()) {

      Artifact apk =
          new ResourceShrinkerActionBuilder(ruleContext)
              .setResourceApkOut(
                  ruleContext.getImplicitOutputArtifact(
                      AndroidRuleClasses.ANDROID_RESOURCES_SHRUNK_APK))
              .setShrunkResourcesOut(
                  ruleContext.getImplicitOutputArtifact(
                      AndroidRuleClasses.ANDROID_RESOURCES_SHRUNK_ZIP))
              .setLogOut(
                  ruleContext.getImplicitOutputArtifact(
                      AndroidRuleClasses.ANDROID_RESOURCE_SHRINKER_LOG))
              .withResourceFiles(
                  ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP))
              .withShrunkJar(proguardOutput.getOutputJar())
              .withProguardMapping(proguardOutput.getMapping())
              .withPrimary(resourceApk.getPrimaryResource())
              .withDependencies(resourceApk.getResourceDependencies())
              .setTargetAaptVersion(AndroidAaptVersion.chooseTargetAaptVersion(ruleContext))
              .setResourceFilterFactory(ResourceFilterFactory.fromRuleContext(ruleContext))
              .setUncompressedExtensions(
                  ruleContext.getExpander().withDataLocations().tokenized("nocompress_extensions"))
              .build();
      filesBuilder.add(ruleContext.getImplicitOutputArtifact(
          AndroidRuleClasses.ANDROID_RESOURCE_SHRINKER_LOG));
      return new ResourceApk(apk,
          resourceApk.getResourceJavaSrcJar(),
          resourceApk.getResourceJavaClassJar(),
          resourceApk.getResourceDependencies(),
          resourceApk.getPrimaryResource(),
          resourceApk.getManifest(),
          resourceApk.getResourceProguardConfig(),
          resourceApk.getMainDexProguardConfig(),
          resourceApk.isLegacy());
    }
    return resourceApk;
  }

  @Immutable
  static final class DexingOutput {
    private final Artifact classesDexZip;
    final Artifact javaResourceJar;
    final ImmutableList<Artifact> shardDexZips;

    private DexingOutput(
        Artifact classesDexZip, Artifact javaResourceJar, Iterable<Artifact> shardDexZips) {
      this.classesDexZip = classesDexZip;
      this.javaResourceJar = javaResourceJar;
      this.shardDexZips = ImmutableList.copyOf(shardDexZips);
    }
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
      @Nullable  Artifact proguardOutputMap)
      throws InterruptedException, RuleErrorException {
    List<String> dexopts = ruleContext.getExpander().withDataLocations().tokenized("dexopts");
    MultidexMode multidexMode = getMultidexMode(ruleContext);
    if (!supportsMultidexMode(ruleContext, multidexMode)) {
      ruleContext.throwWithRuleError("Multidex mode \"" + multidexMode.getAttributeValue()
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

    Artifact mainDexList = ruleContext.getPrerequisiteArtifact("main_dex_list", Mode.TARGET);
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
        mainDexList = createMainDexListAction(
            ruleContext, androidSemantics, proguardedJar, mainDexProguardSpec, proguardOutputMap);
      }

      Artifact classesDex = getDxArtifact(ruleContext, "classes.dex.zip");
      if (dexShards > 1) {
        ImmutableList<Artifact> shards = makeShardArtifacts(
            ruleContext,
            dexShards,
            usesDexArchives ? ".jar.dex.zip" : ".jar");

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
                .setExecutable(ruleContext.getExecutablePrerequisite("$merge_dexzips", Mode.HOST))
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
          Artifact classesDexIntermediate = AndroidBinary.getDxArtifact(
              ruleContext, "intermediate_classes.dex.zip");
          // Have the dexer generate the intermediate file and the "cleaner" action consume this to
          // generate the final archive with only .dex files.
          AndroidCommon.createDexAction(ruleContext, proguardedJar,
              classesDexIntermediate, dexopts, /*multidex=*/ true, mainDexList);
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
            proguardedJar,
            DexArchiveAspect.topLevelDexbuilderDexopts(ruleContext, dexopts),
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
      Artifact shardsToMerge =
          createSharderAction(
              ruleContext,
              dexArchives,
              mainDexList,
              dexopts.contains(DX_MINIMAL_MAIN_DEX_OPTION),
              inclusionFilterJar);
      Artifact multidexShards =
          createTemplatedMergerActions(ruleContext, shardsToMerge, dexopts);
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
      ruleContext.attributeError("incremental_dexing",
          "target cannot be incrementally dexed because it uses Proguard");
      return false;
    }

    if (override == TriState.NO) {
      return false;
    }
    if (isBinaryProguarded) {
      // Require explicit opt-in for proguarded binaries, but no need to check dexopts since we'll
      // be able to create dexbuilder actions with the appropriate flags as part of this rule.
      return override == TriState.YES;
    }
    if (override == TriState.YES || config.useIncrementalDexing()) {
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
        ruleContext.attributeWarning("incremental_dexing",
            String.format("Using incremental dexing even though dexopts %s indicate this target "
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
  private static Artifact createSharderAction(
      RuleContext ruleContext,
      ImmutableList<Artifact> dexArchives,
      @Nullable Artifact mainDexList,
      boolean minimalMainDex,
      @Nullable Artifact inclusionFilterJar) {
    Artifact outputTree =
        ruleContext.getTreeArtifact(
            ruleContext.getUniqueDirectory("dexsplits"), ruleContext.getBinOrGenfilesDirectory());

    SpawnAction.Builder shardAction =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setMnemonic("ShardClassesToDex")
            .setProgressMessage("Assembling dex files for %s",
                ruleContext.getLabel().getCanonicalForm())
            .setExecutable(ruleContext.getExecutablePrerequisite("$dexsharder", Mode.HOST))
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
    if (minimalMainDex) {
      shardCommandLine.add(DX_MINIMAL_MAIN_DEX_OPTION);
    }
    if (inclusionFilterJar != null) {
      shardCommandLine.addExecPath("--inclusion_filter_jar", inclusionFilterJar);
      shardAction.addInput(inclusionFilterJar);
    }
    ruleContext.registerAction(
        shardAction.addCommandLine(shardCommandLine.build()).build(ruleContext));

    return outputTree;
  }

  /**
   * Sets up a monodex {@code $dexmerger} actions for each dex archive in the given tree artifact
   * and returns the output tree artifact.
   *
   * @return Tree artifact containing zips with final dex files named for inclusion in an APK.
   */
  private static Artifact createTemplatedMergerActions(
      RuleContext ruleContext,
      Artifact inputTree,
      Collection<String> dexopts) {
    Artifact outputTree =
        ruleContext.getTreeArtifact(
            ruleContext.getUniqueDirectory("dexfiles"), ruleContext.getBinOrGenfilesDirectory());
    SpawnActionTemplate.Builder dexmerger =
        new SpawnActionTemplate.Builder(inputTree, outputTree)
            .setExecutable(ruleContext.getExecutablePrerequisite("$dexmerger", Mode.HOST))
            .setMnemonics("DexShardsToMerge", "DexMerger")
            .setOutputPathMapper(input -> input.getParentRelativePath());
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
    CustomCommandLine args = CustomCommandLine.builder()
        .add("--exclude_build_data")
        .add("--dont_change_compression")
        .add("--sources")
        .addExpandedTreeArtifactExecPaths(inputTree)
        .addExecPath("--output", outputZip)
        .add("--no_duplicates")  // safety: expect distinct entry names in all inputs
        .build();
    // Must use params file as otherwise expanding the input tree artifact doesn't work
    Artifact paramFile =
        ruleContext.getDerivedArtifact(
            ParameterFile.derivePath(outputZip.getRootRelativePath()), outputZip.getRoot());
    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            ImmutableList.of(inputTree),
            paramFile,
            args,
            ParameterFile.ParameterFileType.SHELL_QUOTED,
            ISO_8859_1));
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
            .setExecutable(ruleContext.getExecutablePrerequisite("$dexmerger", Mode.HOST))
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
    dexmerger.addCommandLine(commandLine.build());
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
    for (Artifact jar : common.getJarsProducedForRuntime()) {
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
              attributes.getBootClassPath(),
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
          ruleContext.getPrerequisites(attr, Mode.TARGET, AndroidRuntimeJarProvider.class));
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
          ruleContext.getPrerequisites(attr, Mode.TARGET, DexArchiveProvider.class));
    }
    ImmutableSet<String> incrementalDexopts =
        DexArchiveAspect.incrementalDexopts(ruleContext, dexopts);
    for (Artifact jar : common.getJarsProducedForRuntime()) {
      // Create dex archives next to all Jars produced by AndroidCommon for this rule.  We need to
      // do this (instead of placing dex archives into the _dx subdirectory like DexArchiveAspect)
      // because for "legacy" ResourceApks, AndroidCommon produces Jars per resource dependency that
      // can theoretically have duplicate basenames, so they go into special directories, and we
      // piggyback on that naming scheme here by placing dex archives into the same directories.
      PathFragment jarPath = jar.getRootRelativePath();
      Artifact dexArchive =
          DexArchiveAspect.createDexArchiveAction(
              ruleContext,
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
            .setExecutable(ruleContext.getExecutablePrerequisite("$shuffle_jars", Mode.HOST))
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

    shardAction.addCommandLine(shardCommandLine.build());
    ruleContext.registerAction(shardAction.build(ruleContext));

    if (makeDexArchives && proguardedJar != null) {
      for (int i = 0; i < shards.size(); ++i) {
        checkState(shuffleOutputs.get(i) != shards.get(i));
        DexArchiveAspect.createDexArchiveAction(
            ruleContext,
            shuffleOutputs.get(i),
            DexArchiveAspect.topLevelDexbuilderDexopts(ruleContext, dexopts),
            shards.get(i));
      }
    }
    return javaResourceJar;
  }

  private static ImmutableList<Artifact> collectRuntimeJars(
      AndroidCommon common, JavaTargetAttributes attributes) {
    return ImmutableList.<Artifact>builder()
        .addAll(common.getRuntimeJars())
        .addAll(attributes.getRuntimeClassPathForArchive())
        .build();
  }

  private static ImmutableList<Artifact> toDexedClasspath(
      RuleContext ruleContext,
      ImmutableList<Artifact> classpath,
      Map<Artifact, Artifact> dexArchives)
      throws RuleErrorException {
    ImmutableList.Builder<Artifact> dexedClasspath = ImmutableList.builder();
    boolean reportMissing =
        AndroidCommon.getAndroidConfig(ruleContext).incrementalDexingErrorOnMissedJars();
    for (Artifact jar : classpath) {
      Artifact dexArchive = dexArchives.get(jar);
      if (reportMissing && dexArchive == null) {
        // Users can create this situation by directly depending on a .jar artifact (checked in
        // or coming from a genrule or similar, b/11285003).  This will also catch new  implicit
        // dependencies that incremental dexing would need to be extended to (b/34949364).
        // Typically the fix for the latter involves propagating DexArchiveAspect along the
        // attribute defining the new implicit dependency.
        ruleContext.throwWithAttributeError("deps", "Dependencies on .jar artifacts are not "
            + "allowed in Android binaries, please use a java_import to depend on "
            + jar.prettyPrint() + ". If this is an implicit dependency then the rule that "
            + "introduces it will need to be fixed to account for it correctly.");
      }
      dexedClasspath.add(dexArchive != null ? dexArchive : jar);
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
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext));
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
   * Creates an action that generates a list of classes to be passed to the dexer's
   * --main-dex-list flag (which specifies the classes that need to be directly in classes.dex).
   * Returns the file containing the list.
   */
  static Artifact createMainDexListAction(
      RuleContext ruleContext,
      AndroidSemantics androidSemantics,
      Artifact jar,
      @Nullable Artifact mainDexProguardSpec,
      @Nullable Artifact proguardOutputMap)
      throws InterruptedException {
    // Process the input jar through Proguard into an intermediate, streamlined jar.
    Artifact strippedJar = AndroidBinary.getDxArtifact(ruleContext, "main_dex_intermediate.jar");
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
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
            .add("-dontobfuscate")
            .add("-dontpreverify");

    List<Artifact> specs = new ArrayList<>();
    specs.addAll(
        ruleContext.getPrerequisiteArtifacts("main_dex_proguard_specs", Mode.TARGET).list());
    if (specs.isEmpty()) {
      specs.add(sdk.getMainDexClasses());
    }
    if (mainDexProguardSpec != null) {
      specs.add(mainDexProguardSpec);
    }

    for (Artifact spec : specs) {
      streamlinedBuilder.addInput(spec);
      streamlinedCommandLine.addExecPath("-include", spec);
    }

    androidSemantics.addMainDexListActionArguments(
        ruleContext, streamlinedBuilder, streamlinedCommandLine, proguardOutputMap);

    streamlinedBuilder.addCommandLine(streamlinedCommandLine.build());
    ruleContext.registerAction(streamlinedBuilder.build(ruleContext));

    // Create the main dex classes list.
    Artifact mainDexList = AndroidBinary.getDxArtifact(ruleContext, "main_dex_list.txt");
    SpawnAction.Builder builder = new SpawnAction.Builder()
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
                            .getExpander().withDataLocations().tokenized("main_dex_list_opts"))
                    .build())
            .build(ruleContext));
    return mainDexList;
  }

  private static Artifact createMainDexProguardSpec(RuleContext ruleContext) {
    return ProguardHelper.getProguardConfigArtifact(ruleContext, "main_dex");
  }

  /**
   * Tests if the resources need to be regenerated.
   *
   * <p>The resources should be regenerated (using aapt) if any of the following are true:
   * <ul>
   *    <li>There is more than one resource container
   *    <li>There are resource filters.
   *    <li>There are extensions that should be compressed.
   * </ul>
   */
  public static boolean shouldRegenerate(RuleContext ruleContext,
      ResourceDependencies resourceDeps) {
    return Iterables.size(resourceDeps.getResourceContainers()) > 1
        || ResourceFilterFactory.hasFilters(ruleContext)
        || ruleContext.attributes().isAttributeValueExplicitlySpecified("nocompress_extensions");
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

  /**
   * Returns an intermediate artifact used to support dex generation.
   */
  public static Artifact getDxArtifact(RuleContext ruleContext, String baseName) {
    return ruleContext.getUniqueDirectoryArtifact("_dx", baseName,
        ruleContext.getBinOrGenfilesDirectory());
  }
}
