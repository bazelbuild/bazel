// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DylibDependingRule.DYLIBS_ATTR_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.objc.AppleDebugOutputsInfo.OutputType;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.MultiArchBinarySupport.DependencySpecificConfiguration;
import java.util.Map;
import java.util.TreeMap;

/** Implementation for the "apple_binary" rule. */
public class AppleBinary implements RuleConfiguredTargetFactory {
  public static final String BINARY_TYPE_ATTR = "binary_type";
  public static final String BUNDLE_LOADER_ATTR_NAME = "bundle_loader";
  public static final String EXTENSION_SAFE_ATTR_NAME = "extension_safe";

  private final CppSemantics cppSemantics;

  protected AppleBinary(CppSemantics cppSemantics) {
    this.cppSemantics = cppSemantics;
  }

  /** Type of linked binary that apple_binary may create. */
  public enum BinaryType {

    /**
     * Binaries that can be loaded by other binaries at runtime, and which can't be directly
     * executed by the operating system. When linking, a bundle_loader binary may be passed which
     * signals the linker on where to look for unimplemented symbols, basically declaring that the
     * bundle should be loaded by that binary. Bundle binaries are usually found in Plugins, and one
     * common use case is tests. Tests are bundled into an .xctest bundle which contains the test
     * binary along with required resources. The test bundle is then loaded and run during test
     * execution.
     */
    LOADABLE_BUNDLE,

    /**
     * Binaries that can be run directly by the operating system. They implement the main method
     * that is the entry point to the program. In Apple apps, they are usually distributed in .app
     * bundles, which are directories that contain the executable along with required resources to
     * run.
     */
    EXECUTABLE,

    /**
     * Binaries meant to be loaded at load time (when the operating system is loading the binary
     * into memory), which cannot be unloaded. They are usually distributed in frameworks, which are
     * .framework bundles that contain the dylib as well as well as required resources to run.
     */
    DYLIB;

    @Override
    public String toString() {
      return name().toLowerCase();
    }

    /**
     * Returns the {@link BinaryType} with given name (case insensitive).
     *
     * @throws IllegalArgumentException if the name does not match a valid platform type.
     */
    public static BinaryType fromString(String name) {
      for (BinaryType binaryType : BinaryType.values()) {
        if (name.equalsIgnoreCase(binaryType.toString())) {
          return binaryType;
        }
      }
      throw new IllegalArgumentException(String.format("Unsupported binary type \"%s\"", name));
    }

    /** Returns the enum values as a list of strings for validation. */
    public static Iterable<String> getValues() {
      return Iterables.transform(ImmutableList.copyOf(values()), Functions.toStringFunction());
    }
  }

  @VisibleForTesting
  static final String BUNDLE_LOADER_NOT_IN_BUNDLE_ERROR =
      "Can only use bundle_loader when binary_type is bundle.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ObjcConfiguration objcConfig =
        ruleContext.getConfiguration().getFragment(ObjcConfiguration.class);
    if (objcConfig.disableNativeAppleBinaryRule()) {
      ruleContext.throwWithRuleError(
          "The native apple_binary rule is deprecated and will be deleted. Please use the Starlark"
              + " rule from https://github.com/bazelbuild/rules_apple.");
    }

    AppleLinkingOutputs linkingOutputs =
        linkMultiArchBinary(
            ruleContext,
            cppSemantics,
            /* avoidDeps= */ ImmutableList.of(),
            ImmutableList.of(),
            ImmutableList.of(),
            AnalysisUtils.isStampingEnabled(ruleContext),
            /* shouldLipo= */ true);

    return ruleConfiguredTargetFromLinkingOutputs(ruleContext, linkingOutputs);
  }

  /**
   * Links a (potentially multi-architecture) binary targeting Apple platforms.
   *
   * <p>This method comprises a bulk of the logic of the {@code apple_binary} rule, and is
   * statically available so that it may be referenced by Starlark APIs that replicate its
   * functionality.
   *
   * @param ruleContext the current rule context
   * @param cppSemantics the cpp semantics to use
   * @param avoidDeps a list of {@code TransitiveInfoColllection} that contain information about
   *     dependencies whose symbols are used by the linked binary but should not be linked into the
   *     binary itself
   * @param extraLinkopts extra linkopts to pass to the linker actions
   * @param extraLinkInputs extra input files to pass to the linker action
   * @param isStampingEnabled whether linkstamping is enabled
   * @param shouldLipo whether lipoing all binary slices as one output is desired
   * @return a tuple containing all necessary information about the linked binary
   */
  public static AppleLinkingOutputs linkMultiArchBinary(
      RuleContext ruleContext,
      CppSemantics cppSemantics,
      ImmutableList<TransitiveInfoCollection> avoidDeps,
      Iterable<String> extraLinkopts,
      Iterable<Artifact> extraLinkInputs,
      boolean isStampingEnabled,
      boolean shouldLipo)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ApplePlatform platform = null;

    if (shouldLipo) {
      MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
      PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

      AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

      try {
        platform = appleConfiguration.getMultiArchPlatform(platformType);
      } catch (IllegalArgumentException e) {
        ruleContext.throwWithRuleError(e);
      }

      avoidDeps =
          ImmutableList.<TransitiveInfoCollection>builder()
              .addAll(getDylibProviderTargets(ruleContext))
              .addAll(avoidDeps)
              .build();
    }

    ImmutableListMultimap<String, TransitiveInfoCollection> cpuToDepsCollectionMap =
        MultiArchBinarySupport.transformMap(ruleContext.getPrerequisitesByConfiguration("deps"));

    ImmutableMap<BuildConfiguration, CcToolchainProvider> childConfigurationsAndToolchains =
        MultiArchBinarySupport.getChildConfigurationsAndToolchains(ruleContext);
    MultiArchBinarySupport multiArchBinarySupport =
        new MultiArchBinarySupport(ruleContext, cppSemantics);

    ImmutableSet<DependencySpecificConfiguration> dependencySpecificConfigurations =
        multiArchBinarySupport.getDependencySpecificConfigurations(
            childConfigurationsAndToolchains, cpuToDepsCollectionMap, avoidDeps);

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();

    NestedSetBuilder<Artifact> binariesToLipo = null;
    ImmutableList.Builder<Artifact> allLinkInputs = ImmutableList.builder();
    ImmutableList.Builder<String> allLinkopts = ImmutableList.builder();
    if (shouldLipo) {
      binariesToLipo = NestedSetBuilder.stableOrder();
      allLinkInputs.addAll(getRequiredLinkInputs(ruleContext));
      allLinkopts.addAll(getRequiredLinkopts(ruleContext));
    }
    allLinkInputs.addAll(extraLinkInputs);
    allLinkopts.addAll(extraLinkopts);

    ImmutableListMultimap<BuildConfiguration, OutputGroupInfo> buildConfigToOutputGroupInfoMap =
        ruleContext.getPrerequisitesByConfiguration("deps", OutputGroupInfo.STARLARK_CONSTRUCTOR);
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();
    for (Map.Entry<BuildConfiguration, OutputGroupInfo> entry :
        buildConfigToOutputGroupInfoMap.entries()) {
      OutputGroupInfo dep = entry.getValue();
      headerTokens.addTransitive(dep.getOutputGroup(CcCompilationHelper.HIDDEN_HEADER_TOKENS));
    }
    outputGroupCollector.put(OutputGroupInfo.VALIDATION, headerTokens.build());

    ObjcProvider.Builder objcProviderBuilder =
        new ObjcProvider.Builder(ruleContext.getAnalysisEnvironment().getStarlarkSemantics());
    for (DependencySpecificConfiguration dependencySpecificConfiguration :
        dependencySpecificConfigurations) {
      objcProviderBuilder.addTransitiveAndPropagate(
          dependencySpecificConfiguration.objcProviderWithDylibSymbols());
    }

    AppleDebugOutputsInfo.Builder legacyDebugOutputsBuilder =
        AppleDebugOutputsInfo.Builder.create();
    AppleLinkingOutputs.Builder builder =
        new AppleLinkingOutputs.Builder().addOutputGroups(outputGroupCollector);

    for (DependencySpecificConfiguration dependencySpecificConfiguration :
        dependencySpecificConfigurations) {
      BuildConfiguration childConfig = dependencySpecificConfiguration.config();
      String configCpu = childConfig.getCpu();
      AppleConfiguration childAppleConfig = childConfig.getFragment(AppleConfiguration.class);
      CppConfiguration childCppConfig = childConfig.getFragment(CppConfiguration.class);
      ObjcConfiguration childObjcConfig = childConfig.getFragment(ObjcConfiguration.class);
      IntermediateArtifacts intermediateArtifacts =
          new IntermediateArtifacts(
              ruleContext, /*archiveFileNameSuffix*/ "", /*outputPrefix*/ "", childConfig);
      String arch = childAppleConfig.getSingleArchitecture();

      Artifact binaryArtifact =
          multiArchBinarySupport.registerConfigurationSpecificLinkActions(
              dependencySpecificConfiguration,
              new ExtraLinkArgs(allLinkopts.build()),
              allLinkInputs.build(),
              isStampingEnabled,
              cpuToDepsCollectionMap.get(configCpu),
              outputGroupCollector);
      if (shouldLipo) {
        binariesToLipo.add(binaryArtifact);
      }

      // TODO(b/177442911): Use the target platform from platform info coming from split
      // transition outputs instead of inferring this based on the target CPU.
      ApplePlatform cpuPlatform = ApplePlatform.forTargetCpu(configCpu);

      AppleLinkingOutputs.LinkingOutput.Builder outputBuilder =
          AppleLinkingOutputs.LinkingOutput.builder()
              .setPlatform(cpuPlatform.getTargetPlatform())
              .setArchitecture(arch)
              .setEnvironment(cpuPlatform.getTargetEnvironment())
              .setBinary(binaryArtifact);

      if (childCppConfig.getAppleBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
        Artifact bitcodeSymbols = intermediateArtifacts.bitcodeSymbolMap();
        outputBuilder.setBitcodeSymbols(bitcodeSymbols);
        legacyDebugOutputsBuilder.addOutput(arch, OutputType.BITCODE_SYMBOLS, bitcodeSymbols);
      }
      if (childCppConfig.appleGenerateDsym()) {
        Artifact dsymBinary =
            childObjcConfig.shouldStripBinary()
                ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
                : intermediateArtifacts.dsymSymbolForStrippedBinary();
        outputBuilder.setDsymBinary(dsymBinary);
        legacyDebugOutputsBuilder.addOutput(arch, OutputType.DSYM_BINARY, dsymBinary);
      }
      if (childObjcConfig.generateLinkmap()) {
        Artifact linkmap = intermediateArtifacts.linkmap();
        outputBuilder.setLinkmap(linkmap);
        legacyDebugOutputsBuilder.addOutput(arch, OutputType.LINKMAP, linkmap);
      }

      builder.addOutput(outputBuilder.build());
    }

    if (shouldLipo) {
      Artifact outputArtifact =
          ObjcRuleClasses.intermediateArtifacts(ruleContext).combinedArchitectureBinary();
      builder.setLegacyBinaryArtifact(outputArtifact, getBinaryType(ruleContext));
      new LipoSupport(ruleContext)
          .registerCombineArchitecturesAction(binariesToLipo.build(), outputArtifact, platform);
    }

    return builder
        .setDepsObjcProvider(objcProviderBuilder.build())
        .setLegacyDebugOutputsProvider(legacyDebugOutputsBuilder.build())
        .build();
  }

  private static ExtraLinkArgs getRequiredLinkopts(RuleContext ruleContext)
      throws RuleErrorException {
    BinaryType binaryType = getBinaryType(ruleContext);

    ImmutableList.Builder<String> extraLinkArgs = new ImmutableList.Builder<>();

    boolean didProvideBundleLoader =
        ruleContext.attributes().isAttributeValueExplicitlySpecified(BUNDLE_LOADER_ATTR_NAME);

    if (didProvideBundleLoader && binaryType != BinaryType.LOADABLE_BUNDLE) {
      ruleContext.throwWithRuleError(BUNDLE_LOADER_NOT_IN_BUNDLE_ERROR);
    }

    switch (binaryType) {
      case LOADABLE_BUNDLE:
        extraLinkArgs.add("-bundle");
        extraLinkArgs.add("-Wl,-rpath,@loader_path/Frameworks");
        if (didProvideBundleLoader) {
          AppleExecutableBinaryInfo executableProvider =
              ruleContext.getPrerequisite(
                  BUNDLE_LOADER_ATTR_NAME, AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR);
          extraLinkArgs.add(
              "-bundle_loader", executableProvider.getAppleExecutableBinary().getExecPathString());
        }
        break;
      case DYLIB:
        extraLinkArgs.add("-dynamiclib");
        break;
      case EXECUTABLE:
        break;
    }

    return new ExtraLinkArgs(extraLinkArgs.build());
  }

  private static ImmutableList<TransitiveInfoCollection> getDylibProviderTargets(
      RuleContext ruleContext) {
    return ImmutableList.<TransitiveInfoCollection>builder()
        .addAll(ruleContext.getPrerequisites(DYLIBS_ATTR_NAME))
        .addAll(ruleContext.getPrerequisites(BUNDLE_LOADER_ATTR_NAME))
        .build();
  }

  private static Iterable<Artifact> getRequiredLinkInputs(RuleContext ruleContext) {
    AppleExecutableBinaryInfo executableProvider =
        ruleContext.getPrerequisite(
            BUNDLE_LOADER_ATTR_NAME, AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR);
    if (executableProvider != null) {
      return ImmutableSet.<Artifact>of(executableProvider.getAppleExecutableBinary());
    }
    return ImmutableSet.<Artifact>of();
  }

  private static BinaryType getBinaryType(RuleContext ruleContext) {
    String binaryTypeString = ruleContext.attributes().get(BINARY_TYPE_ATTR, STRING);
    return BinaryType.fromString(binaryTypeString);
  }

  private static ConfiguredTarget ruleConfiguredTargetFromLinkingOutputs(
      RuleContext ruleContext, AppleLinkingOutputs linkingOutputs)
      throws RuleErrorException, ActionConflictException, InterruptedException {
    NativeInfo nativeInfo = linkingOutputs.getLegacyBinaryInfoProvider();
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    ObjcProvider objcProvider;
    Artifact outputArtifact;

    switch (getBinaryType(ruleContext)) {
      case EXECUTABLE:
        AppleExecutableBinaryInfo executableProvider = (AppleExecutableBinaryInfo) nativeInfo;
        objcProvider = executableProvider.getDepsObjcProvider();
        outputArtifact = executableProvider.getAppleExecutableBinary();
        break;
      case DYLIB:
        AppleDylibBinaryInfo dylibProvider = (AppleDylibBinaryInfo) nativeInfo;
        objcProvider = dylibProvider.getDepsObjcProvider();
        outputArtifact = dylibProvider.getAppleDylibBinary();
        break;
      case LOADABLE_BUNDLE:
        AppleLoadableBundleBinaryInfo loadableBundleProvider =
            (AppleLoadableBundleBinaryInfo) nativeInfo;
        objcProvider = loadableBundleProvider.getDepsObjcProvider();
        outputArtifact = loadableBundleProvider.getAppleLoadableBundleBinary();
        break;
      default:
        throw ruleContext.throwWithRuleError("Unhandled binary type " + getBinaryType(ruleContext));
    }

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().add(outputArtifact);

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    if (appleConfiguration.shouldLinkingRulesPropagateObjc() && objcProvider != null) {
      targetBuilder.addNativeDeclaredProvider(objcProvider);
      targetBuilder.addStarlarkTransitiveInfo(ObjcProvider.STARLARK_NAME, objcProvider);
    }

    InstrumentedFilesInfo instrumentedFilesProvider =
        InstrumentedFilesCollector.forward(ruleContext, "deps", "bundle_loader");

    return targetBuilder
        .addNativeDeclaredProvider(instrumentedFilesProvider)
        .addNativeDeclaredProvider(nativeInfo)
        .addNativeDeclaredProvider(linkingOutputs.getLegacyDebugOutputsProvider())
        .addOutputGroups(linkingOutputs.getOutputGroups())
        .build();
  }
}
