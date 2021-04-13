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
import static com.google.devtools.build.lib.rules.objc.AppleBinaryRule.BUNDLE_LOADER_ATTR_NAME;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_BINARIES;
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
import com.google.devtools.build.lib.rules.objc.AppleDebugOutputsInfo.OutputType;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.MultiArchBinarySupport.DependencySpecificConfiguration;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.Map;
import java.util.TreeMap;

/** Implementation for the "apple_binary" rule. */
public class AppleBinary implements RuleConfiguredTargetFactory {

  /** Type of linked binary that apple_binary may create. */
  enum BinaryType {

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
    static Iterable<String> getValues() {
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

    AppleBinaryOutput appleBinaryOutput =
        linkMultiArchBinary(
            ruleContext,
            ImmutableList.of(),
            ImmutableList.of(),
            AnalysisUtils.isStampingEnabled(ruleContext),
            /* shouldLipo= */ true);

    return ruleConfiguredTargetFromProvider(ruleContext, appleBinaryOutput);
  }

  /**
   * Links a (potentially multi-architecture) binary targeting Apple platforms.
   *
   * <p>This method comprises a bulk of the logic of the {@code apple_binary} rule, and is
   * statically available so that it may be referenced by Starlark APIs that replicate its
   * functionality.
   *
   * @param ruleContext the current rule context
   * @param extraLinkopts extra linkopts to pass to the linker actions
   * @param extraLinkInputs extra input files to pass to the linker action
   * @param isStampingEnabled whether linkstamping is enabled
   * @param shouldLipo whether lipoing all binary slices as one output is desired
   * @return a tuple containing all necessary information about the linked binary
   */
  public static AppleBinaryOutput linkMultiArchBinary(
      RuleContext ruleContext,
      Iterable<String> extraLinkopts,
      Iterable<Artifact> extraLinkInputs,
      boolean isStampingEnabled,
      boolean shouldLipo)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    ApplePlatform platform = null;
    try {
      platform = appleConfiguration.getMultiArchPlatform(platformType);
    } catch (IllegalArgumentException e) {
      ruleContext.throwWithRuleError(e);
    }
    ImmutableListMultimap<String, TransitiveInfoCollection> cpuToDepsCollectionMap =
        MultiArchBinarySupport.transformMap(ruleContext.getPrerequisitesByConfiguration("deps"));
    ImmutableListMultimap<String, ConfiguredTargetAndData> cpuToCTATDepsCollectionMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisiteCofiguredTargetAndTargetsByConfiguration("deps"));

    ImmutableMap<BuildConfiguration, CcToolchainProvider> childConfigurationsAndToolchains =
        MultiArchBinarySupport.getChildConfigurationsAndToolchains(ruleContext);
    Artifact outputArtifact = null;
    if (shouldLipo) {
      outputArtifact =
          ObjcRuleClasses.intermediateArtifacts(ruleContext).combinedArchitectureBinary();
    }

    MultiArchBinarySupport multiArchBinarySupport = new MultiArchBinarySupport(ruleContext);

    ImmutableSet<DependencySpecificConfiguration> dependencySpecificConfigurations =
        multiArchBinarySupport.getDependencySpecificConfigurations(
            childConfigurationsAndToolchains,
            cpuToDepsCollectionMap,
            cpuToCTATDepsCollectionMap,
            getDylibProviderTargets(ruleContext));

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();

    Iterable<Artifact> allLinkInputs =
        Iterables.concat(getRequiredLinkInputs(ruleContext), extraLinkInputs);
    ExtraLinkArgs allLinkopts =
        new ExtraLinkArgs(Iterables.concat(getRequiredLinkopts(ruleContext), extraLinkopts));

    ImmutableMap<String, Artifact> platformToBinariesMap =
        multiArchBinarySupport.registerActions(
            allLinkopts,
            dependencySpecificConfigurations,
            allLinkInputs,
            isStampingEnabled,
            cpuToDepsCollectionMap,
            outputGroupCollector,
            platform);

    if (shouldLipo) {
      NestedSetBuilder<Artifact> binariesToLipo = NestedSetBuilder.stableOrder();
      for (Map.Entry<String, Artifact> entry : platformToBinariesMap.entrySet()) {
        Artifact binaryToLipo = entry.getValue();
        binariesToLipo.add(binaryToLipo);
      }
      new LipoSupport(ruleContext)
          .registerCombineArchitecturesAction(binariesToLipo.build(), outputArtifact, platform);
    }

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
    if (shouldLipo) {
      objcProviderBuilder.add(MULTI_ARCH_LINKED_BINARIES, outputArtifact);
    }

    ObjcProvider objcProvider = objcProviderBuilder.build();
    NativeInfo binaryInfoProvider;

    switch (getBinaryType(ruleContext)) {
      case EXECUTABLE:
        binaryInfoProvider = new AppleExecutableBinaryInfo(outputArtifact, objcProvider);
        break;
      case DYLIB:
        binaryInfoProvider = new AppleDylibBinaryInfo(outputArtifact, objcProvider);
        break;
      case LOADABLE_BUNDLE:
        binaryInfoProvider = new AppleLoadableBundleBinaryInfo(outputArtifact, objcProvider);
        break;
      default:
        throw ruleContext.throwWithRuleError("Unhandled binary type " + getBinaryType(ruleContext));
    }

    AppleDebugOutputsInfo.Builder builder = AppleDebugOutputsInfo.Builder.create();

    for (DependencySpecificConfiguration dependencySpecificConfiguration :
        dependencySpecificConfigurations) {
      AppleConfiguration childAppleConfig =
          dependencySpecificConfiguration.config().getFragment(AppleConfiguration.class);
      CppConfiguration childCppConfig =
          dependencySpecificConfiguration.config().getFragment(CppConfiguration.class);
      ObjcConfiguration childObjcConfig =
          dependencySpecificConfiguration.config().getFragment(ObjcConfiguration.class);
      IntermediateArtifacts intermediateArtifacts =
          new IntermediateArtifacts(
              ruleContext, /*archiveFileNameSuffix*/
              "", /*outputPrefix*/
              "",
              dependencySpecificConfiguration.config());
      String arch = childAppleConfig.getSingleArchitecture();

      if (childCppConfig.getAppleBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
        Artifact bitcodeSymbol = intermediateArtifacts.bitcodeSymbolMap();
        builder.addOutput(arch, OutputType.BITCODE_SYMBOLS, bitcodeSymbol);
      }
      if (childCppConfig.appleGenerateDsym()) {
        Artifact dsymBinary =
            childObjcConfig.shouldStripBinary()
                ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
                : intermediateArtifacts.dsymSymbolForStrippedBinary();
        builder.addOutput(arch, OutputType.DSYM_BINARY, dsymBinary);
      }
      if (childObjcConfig.generateLinkmap()) {
        Artifact linkmap = intermediateArtifacts.linkmap();
        builder.addOutput(arch, OutputType.LINKMAP, linkmap);
      }
    }

    return new AppleBinaryOutput(
        binaryInfoProvider, builder.build(), outputGroupCollector, platformToBinariesMap);
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
    String binaryTypeString =
        ruleContext.attributes().get(AppleBinaryRule.BINARY_TYPE_ATTR, STRING);
    return BinaryType.fromString(binaryTypeString);
  }

  private static ConfiguredTarget ruleConfiguredTargetFromProvider(
      RuleContext ruleContext, AppleBinaryOutput appleBinaryOutput)
      throws RuleErrorException, ActionConflictException, InterruptedException {
    NativeInfo nativeInfo = appleBinaryOutput.getBinaryInfoProvider();
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
        .addNativeDeclaredProvider(appleBinaryOutput.getDebugOutputsProvider())
        .addOutputGroups(appleBinaryOutput.getOutputGroups())
        .build();
  }

  /** The set of rule outputs propagated by the {@code apple_binary} rule. */
  public static class AppleBinaryOutput {
    private final NativeInfo binaryInfoProvider;
    private final AppleDebugOutputsInfo debugOutputsProvider;
    private final Map<String, NestedSet<Artifact>> outputGroups;
    private final Map<String, Artifact> artifactByPlatform;

    private AppleBinaryOutput(
        NativeInfo binaryInfoProvider,
        AppleDebugOutputsInfo debugOutputsProvider,
        Map<String, NestedSet<Artifact>> outputGroups,
        Map<String, Artifact> artifactByPlatform) {
      this.binaryInfoProvider = binaryInfoProvider;
      this.debugOutputsProvider = debugOutputsProvider;
      this.outputGroups = outputGroups;
      this.artifactByPlatform = artifactByPlatform;
    }

    /**
     * Returns a {@link NativeInfo} possessing information about the linked binary. Depending on the
     * type of binary, this may be either a {@link AppleExecutableBinaryInfo}, a {@link
     * AppleDylibBinaryInfo}, or a {@link AppleLoadableBundleBinaryInfo}.
     */
    public NativeInfo getBinaryInfoProvider() {
      return binaryInfoProvider;
    }

    /**
     * Returns a {@link AppleDebugOutputsInfo} containing debug information about the linked binary.
     */
    public AppleDebugOutputsInfo getDebugOutputsProvider() {
      return debugOutputsProvider;
    }

    /**
     * Returns a map from output group name to set of artifacts belonging to this output group. This
     * should be added to configured target information using {@link
     * RuleConfiguredTargetBuilder#addOutputGroups(Map)}.
     */
    public Map<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }

    /**
     * Returns a map from cpu string with device type to each artifact representing a linked single
     * architecture binary before it is lipoed.
     */
    public Map<String, Artifact> getArtifactByPlatform() {
      return artifactByPlatform;
    }
  }
}
