// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.DYNAMIC_LINKING_MODE;
import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.STATIC_LINKING_MODE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.MapBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A ConfiguredTarget for <code>cc_binary</code> rules.
 */
public abstract class CcBinary implements RuleConfiguredTargetFactory {

  private final CppSemantics semantics;

  protected CcBinary(CppSemantics semantics) {
    this.semantics = semantics;
  }

  /**
   * The maximum number of inputs for any single .dwp generating action. For cases where
   * this value is exceeded, the action is split up into "batches" that fall under the limit.
   * See {@link #createDebugPackagerActions} for details.
   */
  @VisibleForTesting
  public static final int MAX_INPUTS_PER_DWP_ACTION = 100;

  /**
   * Intermediate dwps are written to this subdirectory under the main dwp's output path.
   */
  @VisibleForTesting
  public static final String INTERMEDIATE_DWP_DIR = "_dwps";

  /**
   * A string constant for the static_link_srcs feature.
   *
   * <p>Disabling this feature forces srcs of executables to be linked dynamically in
   * DYNAMIC_MODE=AUTO.
   */
  public static final String STATIC_LINK_SRCS = "static_link_srcs";

  /** Provider for native deps launchers. DO NOT USE. */
  @Deprecated
  public static class CcLauncherInfo extends NativeInfo {
    private static final String RESTRICTION_ERROR_MESSAGE =
        "This provider is restricted to native.java_binary, native.py_binary and native.java_test. "
            + "This is a ";
    public static final String PROVIDER_NAME = "CcLauncherInfo";
    public static final Provider PROVIDER = new Provider();

    private final CcCompilationOutputs ccCompilationOutputs;
    private final CcLinkParams staticModeParamsForExecutable;

    public CcLauncherInfo(
        CcLinkParams staticModeParamsForExecutable, CcCompilationOutputs ccCompilationOutputs) {
      super(PROVIDER);
      this.staticModeParamsForExecutable = staticModeParamsForExecutable;
      this.ccCompilationOutputs = ccCompilationOutputs;
    }

    public CcCompilationOutputs getCcCompilationOutputs(RuleContext ruleContext) {
      checkRestrictedUsage(ruleContext);
      return ccCompilationOutputs;
    }

    public CcLinkParams getStaticModeParamsForExecutable(RuleContext ruleContext) {
      checkRestrictedUsage(ruleContext);
      return staticModeParamsForExecutable;
    }

    private void checkRestrictedUsage(RuleContext ruleContext) {
      Rule rule = ruleContext.getRule();
      if (rule.getRuleClassObject().isSkylark()
          || (!rule.getRuleClass().equals("java_binary")
              && !rule.getRuleClass().equals("java_test")
              && !rule.getRuleClass().equals("py_binary")
              && !rule.getRuleClass().equals("py_test"))) {
        throw new IllegalStateException(RESTRICTION_ERROR_MESSAGE + rule.getRuleClass());
      }
    }

    /** Provider class for {@link CcLauncherInfo} objects. */
    public static class Provider extends BuiltinProvider<CcLauncherInfo> {
      private Provider() {
        super(PROVIDER_NAME, CcLauncherInfo.class);
      }
    }
  }

  private static Runfiles collectRunfiles(
      RuleContext ruleContext,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider toolchain,
      CcLinkingOutputs linkingOutputs,
      CcLinkingOutputs ccLibraryLinkingOutputs,
      CcCompilationContext ccCompilationContext,
      Link.LinkingMode linkingMode,
      NestedSet<Artifact> transitiveArtifacts,
      Iterable<Artifact> fakeLinkerInputs,
      boolean fake,
      ImmutableSet<CppSource> cAndCppSources,
      boolean linkCompileOutputSeparately) {
    Runfiles.Builder builder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());
    Function<TransitiveInfoCollection, Runfiles> runfilesMapping =
        CppHelper.runfilesFunction(ruleContext, linkingMode != Link.LinkingMode.DYNAMIC);
    builder.addTransitiveArtifacts(transitiveArtifacts);
    // Add the shared libraries to the runfiles. This adds any shared libraries that are in the
    // srcs of this target.
    builder.addArtifacts(linkingOutputs.getLibrariesForRunfiles(true));
    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    // TODO(plf): Why do we need .so files produced by cc_library in data dependencies of cc_binary?
    // This can probably be removed safely.
    for (TransitiveInfoCollection transitiveInfoCollection :
        ruleContext.getPrerequisites("data", Mode.DONT_CHECK)) {
      builder.merge(
          CppHelper.runfilesFunction(ruleContext, /* linkingStatically= */ true)
              .apply(transitiveInfoCollection));
      builder.merge(
          CppHelper.runfilesFunction(ruleContext, /* linkingStatically= */ false)
              .apply(transitiveInfoCollection));
    }
    builder.add(ruleContext, runfilesMapping);
    // Add the C++ runtime libraries if linking them dynamically.
    if (linkingMode == Link.LinkingMode.DYNAMIC) {
      builder.addTransitiveArtifacts(toolchain.getDynamicRuntimeLinkInputs(featureConfiguration));
    }
    if (linkCompileOutputSeparately) {
      builder.addArtifacts(
          LinkerInputs.toLibraryArtifacts(ccLibraryLinkingOutputs.getDynamicLibrariesForRuntime()));
    }
    // For cc_binary and cc_test rules, there is an implicit dependency on
    // the malloc library package, which is specified by the "malloc" attribute.
    // As the BUILD encyclopedia says, the "malloc" attribute should be ignored
    // if linkshared=1.
    boolean linkshared = isLinkShared(ruleContext);
    if (!linkshared) {
      TransitiveInfoCollection malloc = CppHelper.mallocForTarget(ruleContext);
      builder.addTarget(malloc, RunfilesProvider.DEFAULT_RUNFILES);
      builder.addTarget(malloc, runfilesMapping);
    }

    if (fake) {
      // Add the object files, libraries, and linker scripts that are used to
      // link this executable.
      builder.addSymlinksToArtifacts(Iterables.filter(fakeLinkerInputs, Artifact.MIDDLEMAN_FILTER));
      // The crosstool inputs for the link action are not sufficient; we also need the crosstool
      // inputs for compilation. Node that these cannot be middlemen because Runfiles does not
      // know how to expand them.
      builder.addTransitiveArtifacts(toolchain.getCrosstool());
      builder.addTransitiveArtifacts(toolchain.getLibcLink());
      // Add the sources files that are used to compile the object files.
      // We add the headers in the transitive closure and our own sources in the srcs
      // attribute. We do not provide the auxiliary inputs, because they are only used when we
      // do FDO compilation, and cc_fake_binary does not support FDO.
      ImmutableSet.Builder<Artifact> sourcesBuilder = ImmutableSet.<Artifact>builder();
      for (CppSource cppSource : cAndCppSources) {
        sourcesBuilder.add(cppSource.getSource());
      }
      builder.addSymlinksToArtifacts(sourcesBuilder.build());
      builder.addSymlinksToArtifacts(ccCompilationContext.getDeclaredIncludeSrcs());
      // Add additional files that are referenced from the compile command, like module maps
      // or header modules.
      builder.addSymlinksToArtifacts(ccCompilationContext.getAdditionalInputs());
      builder.addSymlinksToArtifacts(
          ccCompilationContext.getTransitiveModules(usePic(ruleContext, toolchain)));
    }
    return builder.build();
  }

  @Override
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    return CcBinary.init(semantics, context, /*fake =*/ false);
  }

  public static ConfiguredTarget init(CppSemantics semantics, RuleContext ruleContext, boolean fake)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CppHelper.checkAllowedDeps(ruleContext);

    ruleContext.checkSrcsSamePackage(true);

    CcCommon common = new CcCommon(ruleContext);
    CcToolchainProvider ccToolchain = common.getToolchain();

      ImmutableMap.Builder<String, String> toolchainMakeVariables = ImmutableMap.builder();
      ccToolchain.addGlobalMakeVariables(toolchainMakeVariables);
      ruleContext.initConfigurationMakeVariableContext(
          new MapBackedMakeVariableSupplier(toolchainMakeVariables.build()),
          new CcFlagsSupplier(ruleContext));

    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    LinkTargetType linkType =
        isLinkShared(ruleContext) ? LinkTargetType.DYNAMIC_LIBRARY : LinkTargetType.EXECUTABLE;

    semantics.validateAttributes(ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }

    // if cc_binary includes "linkshared=1", then gcc will be invoked with
    // linkopt "-shared", which causes the result of linking to be a shared
    // library. In this case, the name of the executable target should end
    // in ".so" or "dylib" or ".dll".
    Artifact binary;
    PathFragment binaryPath = PathFragment.create(ruleContext.getTarget().getName());
    if (!isLinkShared(ruleContext)) {
      binary =
          CppHelper.getLinkedArtifact(
              ruleContext, ccToolchain, ruleContext.getConfiguration(), LinkTargetType.EXECUTABLE);
    } else {
      binary = ruleContext.getBinArtifact(binaryPath);
    }

    if (isLinkShared(ruleContext)
        && !CppFileTypes.SHARED_LIBRARY.matches(binary.getFilename())
        && !CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(binary.getFilename())) {
      ruleContext.attributeError("linkshared", "'linkshared' used in non-shared library");
      return null;
    }

    LinkingMode linkingMode = getLinkStaticness(ruleContext, cppConfiguration);
    ImmutableSet.Builder<String> requestedFeaturesBuilder = new ImmutableSet.Builder<>();
    requestedFeaturesBuilder
        .addAll(ruleContext.getFeatures())
        .add(linkingMode == Link.LinkingMode.DYNAMIC ? DYNAMIC_LINKING_MODE : STATIC_LINKING_MODE);
    if (fake) {
      requestedFeaturesBuilder.add(CppRuleClasses.IS_CC_FAKE_BINARY);
    }

    FdoProvider fdoProvider = common.getFdoProvider();
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(
            ruleContext,
            requestedFeaturesBuilder.build(),
            /* unsupportedFeatures= */ ruleContext.getDisabledFeatures(),
            ccToolchain);

    CcCompilationHelper compilationHelper =
        new CcCompilationHelper(
                ruleContext, semantics, featureConfiguration, ccToolchain, fdoProvider)
            .fromCommon(common, /* additionalCopts= */ ImmutableList.of())
            .addPrivateHeaders(common.getPrivateHeaders())
            .addSources(common.getSources())
            .addDeps(ImmutableList.of(CppHelper.mallocForTarget(ruleContext)))
            .setFake(fake)
            .addPrecompiledFiles(precompiledFiles);
    CompilationInfo compilationInfo = compilationHelper.compile();
    CcCompilationContext ccCompilationContext = compilationInfo.getCcCompilationContext();
    CcCompilationOutputs ccCompilationOutputs = compilationInfo.getCcCompilationOutputs();

    // Allows the dynamic library generated for code of default dynamic mode targets to be linked
    // separately. The main use case for default dynamic mode is the cc_test rule. The same behavior
    // can also be enabled specifically for tests with an experimental flag.
    // TODO(meikeb): Retire the experimental flag by end of 2018.
    boolean linkCompileOutputSeparately =
        (ruleContext.isTestTarget()
                && cppConfiguration.getLinkCompileOutputSeparately()
                && linkingMode == LinkingMode.DYNAMIC)
            || (cppConfiguration.getDynamicModeFlag() == DynamicMode.DEFAULT
                && ruleContext.getDisabledFeatures().contains(STATIC_LINK_SRCS));
    // When linking the object files directly into the resulting binary, we do not need
    // library-level link outputs; thus, we do not let CcCompilationHelper produce link outputs
    // (either shared object files or archives) for a non-library link type [*], and add
    // the object files explicitly in determineLinkerArguments.
    //
    // When linking the object files into their own library, we want CcCompilationHelper to
    // take care of creating the library link outputs for us, so we need to set the link
    // type to STATIC_LIBRARY.
    //
    // [*] The only library link type is STATIC_LIBRARY. EXECUTABLE specifies a normal
    // cc_binary output, while DYNAMIC_LIBRARY is a cc_binary rules that produces an
    // output matching a shared object, for example cc_binary(name="foo.so", ...) on linux.
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (linkCompileOutputSeparately && !ccCompilationOutputs.isEmpty()) {
      CcLinkingHelper linkingHelper =
          new CcLinkingHelper(
                  ruleContext,
                  semantics,
                  featureConfiguration,
                  ccToolchain,
                  fdoProvider,
                  ruleContext.getConfiguration())
              .fromCommon(common)
              .addDeps(ImmutableList.of(CppHelper.mallocForTarget(ruleContext)))
              .emitInterfaceSharedObjects(true)
              .setAlwayslink(false);
      ccLinkingOutputs = linkingHelper.link(ccCompilationOutputs);
    }

    CcLinkParams linkParams =
        collectCcLinkParams(
            ruleContext,
            linkingMode != Link.LinkingMode.DYNAMIC,
            isLinkShared(ruleContext),
            common.getLinkopts());

    Artifact generatedDefFile = null;
    Artifact customDefFile = null;
    if (isLinkShared(ruleContext)) {
      if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        ImmutableList.Builder<Artifact> objectFiles = ImmutableList.builder();
        objectFiles.addAll(ccCompilationOutputs.getObjectFiles(false));
        for (LibraryToLink library : linkParams.getLibraries()) {
          if (library.containsObjectFiles()
              && library.getArtifactCategory() != ArtifactCategory.DYNAMIC_LIBRARY
              && library.getArtifactCategory() != ArtifactCategory.INTERFACE_LIBRARY) {
            objectFiles.addAll(library.getObjectFiles());
          }
        }
        generatedDefFile =
            CppHelper.createDefFileActions(
                ruleContext,
                ruleContext.getPrerequisiteArtifact("$def_parser", Mode.HOST),
                objectFiles.build(),
                binary.getFilename());
        customDefFile = common.getWinDefFile();
      }
    }

    boolean usePic = usePic(ruleContext, ccToolchain);

    // On Windows, if GENERATE_PDB_FILE feature is enabled
    // then a pdb file will be built along with the executable.
    Artifact pdbFile = null;
    if (featureConfiguration.isEnabled(CppRuleClasses.GENERATE_PDB_FILE)) {
      pdbFile = ruleContext.getRelatedArtifact(binary.getRootRelativePath(), ".pdb");
    }

    NestedSetBuilder<LibraryToLink> extraLinkTimeLibraries = NestedSetBuilder.linkOrder();
    NestedSetBuilder<Artifact> extraLinkTimeRuntimeLibraries = NestedSetBuilder.linkOrder();

    if (linkParams.getExtraLinkTimeLibraries() != null) {
      ExtraLinkTimeLibrary.BuildLibraryOutput extraLinkBuildLibraryOutput =
          linkParams
              .getExtraLinkTimeLibraries()
              .buildLibraries(
                  ruleContext, linkingMode != LinkingMode.DYNAMIC, isLinkShared(ruleContext));
      extraLinkTimeLibraries.addTransitive(extraLinkBuildLibraryOutput.getLibrariesToLink());
      extraLinkTimeRuntimeLibraries.addTransitive(
          extraLinkBuildLibraryOutput.getRuntimeLibraries());
    }

    Pair<CcLinkingOutputs, Pair<CcLinkingInfo, CcCompilationOutputs>>
        ccLinkingOutputsAndCcLinkingInfo =
            createTransitiveLinkingActions(
                ruleContext,
                ccToolchain,
                featureConfiguration,
                fdoProvider,
                common,
                precompiledFiles,
                ccCompilationOutputs,
                ccLinkingOutputs,
                ccCompilationContext,
                fake,
                binary,
                linkParams,
                extraLinkTimeLibraries.build(),
                linkCompileOutputSeparately,
                semantics,
                linkingMode,
                cppConfiguration,
                linkType,
                pdbFile,
                generatedDefFile,
                customDefFile);

    CcLinkingOutputs ccLinkingOutputsBinary = ccLinkingOutputsAndCcLinkingInfo.first;

    CcLauncherInfo ccLauncherInfo =
        new CcLauncherInfo(
            ccLinkingOutputsAndCcLinkingInfo.second.first.getStaticModeParamsForExecutable(),
            ccLinkingOutputsAndCcLinkingInfo.second.second);

    LibraryToLink outputLibrary = null;
    List<LibraryToLink> dynamicLibrariesForRuntime =
        ccLinkingOutputsBinary.getDynamicLibrariesForRuntime();
    if (!dynamicLibrariesForRuntime.isEmpty()) {
      Preconditions.checkState(dynamicLibrariesForRuntime.size() == 1);
      outputLibrary = dynamicLibrariesForRuntime.get(0);
    }
    Iterable<Artifact> fakeLinkerInputs =
        fake ? ccLinkingOutputsBinary.getLinkActionInputs() : ImmutableList.<Artifact>of();
    CcLinkingOutputs.Builder linkingOutputsBuilder = new CcLinkingOutputs.Builder();
    if (isLinkShared(ruleContext)) {
      linkingOutputsBuilder.addDynamicLibraryForLinking(outputLibrary);
      linkingOutputsBuilder.addDynamicLibraryForRuntime(outputLibrary);
    }
    // Also add all shared libraries from srcs.
    for (Artifact library : precompiledFiles.getSharedLibraries()) {
      Artifact symlink = common.getDynamicLibrarySymlink(library, true);
      LibraryToLink symlinkLibrary = LinkerInputs.solibLibraryToLink(
          symlink, library, CcLinkingOutputs.libraryIdentifierOf(library));
      linkingOutputsBuilder.addDynamicLibraryForLinking(symlinkLibrary);
      linkingOutputsBuilder.addDynamicLibraryForRuntime(symlinkLibrary);
    }
    CcLinkingOutputs linkingOutputs = linkingOutputsBuilder.build();
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER, binary);

    // Create the stripped binary, but don't add it to filesToBuild; it's only built when requested.
    Artifact strippedFile = ruleContext.getImplicitOutputArtifact(
        CppRuleClasses.CC_BINARY_STRIPPED);
    CppHelper.createStripAction(
        ruleContext, ccToolchain, cppConfiguration, binary, strippedFile, featureConfiguration);

    DwoArtifactsCollector dwoArtifacts =
        collectTransitiveDwoArtifacts(
            ruleContext,
            ccCompilationOutputs,
            linkingMode,
            ccToolchain.useFission(),
            usePic,
            ccLinkingOutputsBinary.getAllLtoArtifacts());
    Artifact dwpFile =
        ruleContext.getImplicitOutputArtifact(CppRuleClasses.CC_BINARY_DEBUG_PACKAGE);
    createDebugPackagerActions(ruleContext, ccToolchain, dwpFile, dwoArtifacts);

    // The debug package should include the dwp file only if it was explicitly requested.
    Artifact explicitDwpFile = dwpFile;
    if (!ccToolchain.useFission()) {
      explicitDwpFile = null;
    } else {
      // For cc_test rules, include the dwp in the runfiles if Fission is enabled and the test was
      // built statically.
      if (TargetUtils.isTestRule(ruleContext.getRule())
          && linkingMode != Link.LinkingMode.DYNAMIC
          && ccToolchain.useFission()
          && cppConfiguration.buildTestDwpIsActivated()) {
        filesToBuild = NestedSetBuilder.fromNestedSet(filesToBuild).add(dwpFile).build();
      }
    }

    // If the binary is linked dynamically and COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, collect
    // all the dynamic libraries we need at runtime. Then copy these libraries next to the binary.
    if (featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
      filesToBuild =
          NestedSetBuilder.fromNestedSet(filesToBuild)
              .addAll(
                  createDynamicLibrariesCopyActions(
                      ruleContext, linkParams.getDynamicLibrariesForRuntime()))
              .build();
    }

    // TODO(bazel-team): Do we need to put original shared libraries (along with
    // mangled symlinks) into the RunfilesSupport object? It does not seem
    // logical since all symlinked libraries will be linked anyway and would
    // not require manual loading but if we do, then we would need to collect
    // their names and use a different constructor below.
    Runfiles runfiles =
        collectRunfiles(
            ruleContext,
            featureConfiguration,
            ccToolchain,
            linkingOutputs,
            ccLinkingOutputs,
            ccCompilationContext,
            linkingMode,
            NestedSetBuilder.<Artifact>stableOrder()
                .addTransitive(filesToBuild)
                .addTransitive(extraLinkTimeRuntimeLibraries.build())
                .build(),
            fakeLinkerInputs,
            fake,
            compilationHelper.getCompilationUnitSources(),
            linkCompileOutputSeparately);
    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(ruleContext, runfiles, binary);

    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    addTransitiveInfoProviders(
        ruleContext,
        ccToolchain,
        cppConfiguration,
        common,
        ruleBuilder,
        filesToBuild,
        ccCompilationOutputs,
        ccCompilationContext,
        linkingOutputs,
        dwoArtifacts,
        fake);

    // Support test execution on darwin.
    if (ApplePlatform.isApplePlatform(ccToolchain.getTargetCpu())
        && TargetUtils.isTestRule(ruleContext.getRule())) {
      ruleBuilder.addNativeDeclaredProvider(
          new ExecutionInfo(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, "")));
    }

    // If PDB file is generated by the link action, we add it to pdb_file output group
    if (pdbFile != null) {
      ruleBuilder.addOutputGroup("pdb_file", pdbFile);
    }

    if (generatedDefFile != null) {
      ruleBuilder.addOutputGroup("def_file", generatedDefFile);
    }

    List<LibraryToLink> dynamicLibraryForLinking =
        ccLinkingOutputsBinary.getDynamicLibrariesForLinking();
    if (!dynamicLibraryForLinking.isEmpty()) {
      Preconditions.checkState(dynamicLibraryForLinking.size() == 1);
      ruleBuilder.addOutputGroup(
          "interface_library", dynamicLibraryForLinking.get(0).getOriginalLibraryArtifact());
    }

    return ruleBuilder
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .addProvider(
            DebugPackageProvider.class,
            new DebugPackageProvider(ruleContext.getLabel(), strippedFile, binary, explicitDwpFile))
        .setRunfilesSupport(runfilesSupport, binary)
        .addNativeDeclaredProvider(ccLauncherInfo)
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .build();
  }

  public static Pair<CcLinkingOutputs, Pair<CcLinkingInfo, CcCompilationOutputs>>
      createTransitiveLinkingActions(
          RuleContext ruleContext,
          CcToolchainProvider ccToolchain,
          FeatureConfiguration featureConfiguration,
          FdoProvider fdoProvider,
          CcCommon common,
          PrecompiledFiles precompiledFiles,
          CcCompilationOutputs ccCompilationOutputs,
          CcLinkingOutputs ccLinkingOutputs,
          CcCompilationContext ccCompilationContext,
          boolean fake,
          Artifact binary,
          CcLinkParams linkParams,
          NestedSet<LibraryToLink> extraLinkTimeLibraries,
          boolean linkCompileOutputSeparately,
          CppSemantics cppSemantics,
          LinkingMode linkingMode,
          CppConfiguration cppConfiguration,
          LinkTargetType linkType,
          Artifact pdbFile,
          Artifact generatedDefFile,
          Artifact customDefFile)
          throws InterruptedException, RuleErrorException {
    CcCompilationOutputs.Builder ccCompilationOutputsBuilder =
        new CcCompilationOutputs.Builder()
            .addPicObjectFiles(ccCompilationOutputs.getObjectFiles(/* usePic= */ true))
            .addObjectFiles(ccCompilationOutputs.getObjectFiles(/* usePic= */ false));
    for (Map.Entry<Artifact, Artifact> entry :
        ccCompilationOutputs.getLtoBitcodeFiles().entrySet()) {
      ccCompilationOutputsBuilder.addLtoBitcodeFile(entry.getKey(), entry.getValue());
    }
    CcCompilationOutputs ccCompilationOutputsWithOnlyObjects = ccCompilationOutputsBuilder.build();
    CcLinkingHelper ccLinkingHelper =
        new CcLinkingHelper(
            ruleContext,
            cppSemantics,
            featureConfiguration,
            ccToolchain,
            fdoProvider,
            ruleContext.getConfiguration());

    CcLinkParams.Builder ccLinkParamsBuilder = CcLinkParams.builder();
    ccLinkParamsBuilder.addTransitiveArgs(linkParams);

    if (linkCompileOutputSeparately) {
      for (LibraryToLink library : ccLinkingOutputs.getDynamicLibrariesForLinking()) {
        ccLinkParamsBuilder.addLibrary(library);
      }
      ccCompilationOutputsWithOnlyObjects = new CcCompilationOutputs.Builder().build();
    }

    // Determine the libraries to link in.
    // First libraries from srcs. Shared library artifacts here are substituted with mangled symlink
    // artifacts generated by getDynamicLibraryLink(). This is done to minimize number of -rpath
    // entries during linking process.
    for (Artifact library : precompiledFiles.getLibraries()) {
      if (Link.SHARED_LIBRARY_FILETYPES.matches(library.getFilename())) {
        ccLinkParamsBuilder.addLibrary(
            LinkerInputs.solibLibraryToLink(
                common.getDynamicLibrarySymlink(library, true),
                library,
                CcLinkingOutputs.libraryIdentifierOf(library)));
      } else if (Link.LINK_LIBRARY_FILETYPES.matches(library.getFilename())) {
        ccLinkParamsBuilder.addLibrary(
            LinkerInputs.precompiledLibraryToLink(
                library, ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY));
      } else if (Link.ARCHIVE_FILETYPES.matches(library.getFilename())) {
        ccLinkParamsBuilder.addLibrary(
            LinkerInputs.precompiledLibraryToLink(library, ArtifactCategory.STATIC_LIBRARY));
      } else {
        throw new IllegalStateException();
      }
    }

    if (linkingMode != Link.LinkingMode.DYNAMIC
        && !cppConfiguration.disableEmittingStaticLibgcc()) {
      // Only force a static link of libgcc if static runtime linking is enabled (which
      // can't be true if runtimeInputs is empty).
      // TODO(bazel-team): Move this to CcToolchain.
      if (!ccToolchain.getStaticRuntimeLinkInputs(featureConfiguration).isEmpty()) {
        ccLinkParamsBuilder.addLinkOpts(ImmutableList.of("-static-libgcc"));
      }
    }

    ccLinkParamsBuilder
        .addNonCodeInputs(ccCompilationContext.getTransitiveCompilationPrerequisites())
        .addNonCodeInputs(common.getLinkerScripts());

    CcLinkParams ccLinkParamsForLauncher = ccLinkParamsBuilder.build();
    CcLinkParams ccLinkParams =
        CcLinkParams.builder()
            .addTransitiveArgs(ccLinkParamsForLauncher)
            .addLibraries(extraLinkTimeLibraries)
            .build();

    CcLinkingInfo.Builder ccLinkingInfo = CcLinkingInfo.Builder.create();
    ccLinkingInfo.setStaticModeParamsForDynamicLibrary(ccLinkParams);
    ccLinkingInfo.setStaticModeParamsForExecutable(ccLinkParams);
    ccLinkingInfo.setDynamicModeParamsForDynamicLibrary(ccLinkParams);
    ccLinkingInfo.setDynamicModeParamsForExecutable(ccLinkParams);

    CcLinkingInfo.Builder ccLinkingInfoForLauncher = CcLinkingInfo.Builder.create();
    ccLinkingInfoForLauncher.setStaticModeParamsForDynamicLibrary(ccLinkParamsForLauncher);
    ccLinkingInfoForLauncher.setStaticModeParamsForExecutable(ccLinkParamsForLauncher);
    ccLinkingInfoForLauncher.setDynamicModeParamsForDynamicLibrary(ccLinkParamsForLauncher);
    ccLinkingInfoForLauncher.setDynamicModeParamsForExecutable(ccLinkParamsForLauncher);

    ccLinkingHelper
        .addCcLinkingInfos(ImmutableList.of(ccLinkingInfo.build()))
        .setUseTestOnlyFlags(ruleContext.isTestTarget())
        .setShouldCreateStaticLibraries(false)
        .setLinkingMode(linkingMode)
        .setDynamicLinkType(linkType)
        .setLinkerOutputArtifact(binary)
        .setNeverLink(true)
        .emitInterfaceSharedObjects(
            isLinkShared(ruleContext)
                && featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)
                && CppHelper.useInterfaceSharedObjects(cppConfiguration, ccToolchain))
        .setPdbFile(pdbFile)
        .setFake(fake);

    if (customDefFile != null) {
      ccLinkingHelper.setDefFile(customDefFile);
    } else if (CppHelper.shouldUseGeneratedDefFile(ruleContext, featureConfiguration)) {
      ccLinkingHelper.setDefFile(generatedDefFile);
    }

    return Pair.of(
        ccLinkingHelper.link(ccCompilationOutputsWithOnlyObjects),
        Pair.of(ccLinkingInfoForLauncher.build(), ccCompilationOutputsWithOnlyObjects));
  }

  /**
   * Returns "true" if the {@code linkshared} attribute exists and is set.
   */
  private static final boolean isLinkShared(RuleContext context) {
    return context.attributes().has("linkshared", Type.BOOLEAN)
        && context.attributes().get("linkshared", Type.BOOLEAN);
  }

  private static final LinkingMode getLinkStaticness(
      RuleContext context, CppConfiguration cppConfiguration) {
    if (cppConfiguration.getDynamicModeFlag() == DynamicMode.FULLY) {
      return LinkingMode.DYNAMIC;
    } else if (cppConfiguration.getDynamicModeFlag() == DynamicMode.OFF
        || context.attributes().get("linkstatic", Type.BOOLEAN)) {
      return LinkingMode.STATIC;
    } else {
      return LinkingMode.DYNAMIC;
    }
  }

  /**
   * Collects .dwo artifacts either transitively or directly, depending on the link type.
   *
   * <p>For a cc_binary, we only include the .dwo files corresponding to the .o files that are
   * passed into the link. For static linking, this includes all transitive dependencies. But for
   * dynamic linking, dependencies are separately linked into their own shared libraries, so we
   * don't need them here.
   */
  private static DwoArtifactsCollector collectTransitiveDwoArtifacts(
      RuleContext context,
      CcCompilationOutputs compilationOutputs,
      Link.LinkingMode linkingMode,
      boolean generateDwo,
      boolean ltoBackendArtifactsUsePic,
      Iterable<LtoBackendArtifacts> ltoBackendArtifacts) {
    if (linkingMode == LinkingMode.DYNAMIC) {
      return DwoArtifactsCollector.directCollector(
          compilationOutputs, generateDwo, ltoBackendArtifactsUsePic, ltoBackendArtifacts);
    } else {
      return CcCommon.collectTransitiveDwoArtifacts(
          context, compilationOutputs, generateDwo, ltoBackendArtifactsUsePic, ltoBackendArtifacts);
    }
  }

  @VisibleForTesting
  public static Iterable<Artifact> getDwpInputs(
      RuleContext context,
      CcToolchainProvider toolchain,
      NestedSet<Artifact> picDwoArtifacts,
      NestedSet<Artifact> dwoArtifacts) {
    return usePic(context, toolchain) ? picDwoArtifacts : dwoArtifacts;
  }

  /**
   * Creates the actions needed to generate this target's "debug info package" (i.e. its .dwp file).
   */
  private static void createDebugPackagerActions(
      RuleContext context,
      CcToolchainProvider toolchain,
      Artifact dwpOutput,
      DwoArtifactsCollector dwoArtifactsCollector) {
    Iterable<Artifact> allInputs =
        getDwpInputs(
            context,
            toolchain,
            dwoArtifactsCollector.getPicDwoArtifacts(),
            dwoArtifactsCollector.getDwoArtifacts());

    // No inputs? Just generate a trivially empty .dwp.
    //
    // Note this condition automatically triggers for any build where fission is disabled.
    // Because rules referencing .dwp targets may be invoked with or without fission, we need
    // to support .dwp generation even when fission is disabled. Since no actual functionality
    // is expected then, an empty file is appropriate.
    if (Iterables.isEmpty(allInputs)) {
      context.registerAction(FileWriteAction.create(context, dwpOutput, "", false));
      return;
    }

    // Get the tool inputs necessary to run the dwp command.
    NestedSet<Artifact> dwpTools = toolchain.getDwp();
    Preconditions.checkState(!dwpTools.isEmpty());

    // We apply a hierarchical action structure to limit the maximum number of inputs to any
    // single action.
    //
    // While the dwp tool consumes .dwo files, it can also consume intermediate .dwp files,
    // allowing us to split a large input set into smaller batches of arbitrary size and order.
    // Aside from the parallelism performance benefits this offers, this also reduces input
    // size requirements: if a.dwo, b.dwo, c.dwo, and e.dwo are each 1 KB files, we can apply
    // two intermediate actions DWP(a.dwo, b.dwo) --> i1.dwp and DWP(c.dwo, e.dwo) --> i2.dwp.
    // When we then apply the final action DWP(i1.dwp, i2.dwp) --> finalOutput.dwp, the inputs
    // to this action will usually total far less than 4 KB.
    //
    // The actions form an n-ary tree with n == MAX_INPUTS_PER_DWP_ACTION. The tree is fuller
    // at the leaves than the root, but that both increases parallelism and reduces the final
    // action's input size.
    Packager packager =
        createIntermediateDwpPackagers(context, dwpOutput, toolchain, dwpTools, allInputs, 1);
    packager.spawnAction.setMnemonic("CcGenerateDwp").addOutput(dwpOutput);
    packager.commandLine.addExecPath("-o", dwpOutput);
    context.registerAction(packager.build(context));
  }

  private static class Packager {
    SpawnAction.Builder spawnAction = new SpawnAction.Builder();
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();

    Action[] build(RuleContext context) {
      spawnAction.addCommandLine(
          commandLine.build(), ParamFileInfo.builder(ParameterFileType.UNQUOTED).build());
      return spawnAction.build(context);
    }
  }

  /**
   * Creates the intermediate actions needed to generate this target's "debug info package" (i.e.
   * its .dwp file).
   */
  private static Packager createIntermediateDwpPackagers(
      RuleContext context,
      Artifact dwpOutput,
      CcToolchainProvider toolchain,
      NestedSet<Artifact> dwpTools,
      Iterable<Artifact> inputs,
      int intermediateDwpCount) {
    List<Packager> packagers = new ArrayList<>();

    // Step 1: generate our batches. We currently break into arbitrary batches of fixed maximum
    // input counts, but we can always apply more intelligent heuristics if the need arises.
    Packager currentPackager = newDwpAction(toolchain, dwpTools);
    int inputsForCurrentPackager = 0;

    for (Artifact dwoInput : inputs) {
      if (inputsForCurrentPackager == MAX_INPUTS_PER_DWP_ACTION) {
        packagers.add(currentPackager);
        currentPackager = newDwpAction(toolchain, dwpTools);
        inputsForCurrentPackager = 0;
      }
      currentPackager.spawnAction.addInput(dwoInput);
      currentPackager.commandLine.addExecPath(dwoInput);
      inputsForCurrentPackager++;
    }
    packagers.add(currentPackager);
    // Step 2: given the batches, create the actions.
    if (packagers.size() > 1) {
      // If we have multiple batches, make them all intermediate actions, then pipe their outputs
      // into an additional level.
      List<Artifact> intermediateOutputs = new ArrayList<>();

      for (Packager packager : packagers) {
        Artifact intermediateOutput =
            getIntermediateDwpFile(context, dwpOutput, intermediateDwpCount++);
        packager.spawnAction.setMnemonic("CcGenerateIntermediateDwp").addOutput(intermediateOutput);
        packager.commandLine.addExecPath("-o", intermediateOutput);
        context.registerAction(packager.build(context));
        intermediateOutputs.add(intermediateOutput);
      }
      return createIntermediateDwpPackagers(
          context, dwpOutput, toolchain, dwpTools, intermediateOutputs, intermediateDwpCount);
    }
    return Iterables.getOnlyElement(packagers);
  }

  /**
   * Create the actions to symlink/copy execution dynamic libraries to binary directory so that they
   * are available at runtime.
   *
   * @param dynamicLibrariesForRuntime The libraries to be copied.
   * @return The result artifacts of the copies.
   */
  private static ImmutableList<Artifact> createDynamicLibrariesCopyActions(
      RuleContext ruleContext, NestedSet<Artifact> dynamicLibrariesForRuntime) {
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (Artifact target : dynamicLibrariesForRuntime) {
      if (!ruleContext.getLabel().getPackageName().equals(target.getOwner().getPackageName())) {
        // SymlinkAction on file is actually copy on Windows.
        Artifact copy = ruleContext.getBinArtifact(target.getFilename());
        ruleContext.registerAction(SymlinkAction.toArtifact(
            ruleContext.getActionOwner(), target, copy, "Copying Execution Dynamic Library"));
        result.add(copy);
      }
    }
    return result.build();
  }

  /**
   * Returns a new SpawnAction builder for generating dwp files, pre-initialized with standard
   * settings.
   */
  private static Packager newDwpAction(
      CcToolchainProvider toolchain, NestedSet<Artifact> dwpTools) {
    Packager packager = new Packager();
    packager
        .spawnAction
        .addTransitiveInputs(dwpTools)
        .setExecutable(toolchain.getToolPathFragment(Tool.DWP));
    return packager;
  }

  /**
   * Creates an intermediate dwp file keyed off the name and path of the final output.
   */
  private static Artifact getIntermediateDwpFile(RuleContext ruleContext, Artifact dwpOutput,
      int orderNumber) {
    PathFragment outputPath = dwpOutput.getRootRelativePath();
    PathFragment intermediatePath =
        FileSystemUtils.appendWithoutExtension(outputPath, "-" + orderNumber);
    return ruleContext.getPackageRelativeArtifact(
        PathFragment.create(INTERMEDIATE_DWP_DIR + "/" + intermediatePath.getPathString()),
        dwpOutput.getRoot());
  }

  /**
   * Collect link parameters from the transitive closure.
   */
  private static CcLinkParams collectCcLinkParams(RuleContext context,
      boolean linkingStatically, boolean linkShared, List<String> linkopts) {
    CcLinkParams.Builder builder = CcLinkParams.builder(linkingStatically, linkShared);

    builder.addCcLibrary(context);
    if (!isLinkShared(context)) {
      builder.addTransitiveTarget(CppHelper.mallocForTarget(context));
    }
    builder.addLinkOpts(linkopts);
    return builder.build();
  }

  private static void addTransitiveInfoProviders(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      CppConfiguration cppConfiguration,
      CcCommon common,
      RuleConfiguredTargetBuilder builder,
      NestedSet<Artifact> filesToBuild,
      CcCompilationOutputs ccCompilationOutputs,
      CcCompilationContext ccCompilationContext,
      CcLinkingOutputs linkingOutputs,
      DwoArtifactsCollector dwoArtifacts,
      boolean fake) {
    List<Artifact> instrumentedObjectFiles = new ArrayList<>();
    instrumentedObjectFiles.addAll(ccCompilationOutputs.getObjectFiles(false));
    instrumentedObjectFiles.addAll(ccCompilationOutputs.getObjectFiles(true));
    InstrumentedFilesInfo instrumentedFilesProvider =
        common.getInstrumentedFilesProvider(
            instrumentedObjectFiles,
            !TargetUtils.isTestRule(ruleContext.getRule()) && !fake,
            ccCompilationContext.getVirtualToOriginalHeaders());

    NestedSet<Artifact> headerTokens =
        CcCompilationHelper.collectHeaderTokens(ruleContext, ccCompilationOutputs);
    NestedSet<Artifact> filesToCompile =
        ccCompilationOutputs.getFilesToCompile(
            cppConfiguration.processHeadersInDependencies(), toolchain.usePicForDynamicLibraries());

    builder
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(
            CcInfo.builder().setCcCompilationContext(ccCompilationContext).build())
        .addProvider(
            CcNativeLibraryProvider.class,
            new CcNativeLibraryProvider(
                collectTransitiveCcNativeLibraries(
                    ruleContext, linkingOutputs.getDynamicLibrariesForLinking())))
        .addNativeDeclaredProvider(instrumentedFilesProvider)
        .addProvider(
            CppDebugFileProvider.class,
            new CppDebugFileProvider(
                dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()))
        .addOutputGroup(OutputGroupInfo.TEMP_FILES, ccCompilationOutputs.getTemps())
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, filesToCompile)
        // For CcBinary targets, we only want to ensure that we process headers in dependencies and
        // thus only add header tokens to HIDDEN_TOP_LEVEL. If we add all HIDDEN_TOP_LEVEL artifacts
        // from dependent CcLibrary targets, we'd be building .pic.o files in nopic builds.
        .addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, headerTokens)
        .addOutputGroup(
            OutputGroupInfo.COMPILATION_PREREQUISITES,
            CcCommon.collectCompilationPrerequisites(ruleContext, ccCompilationContext));

    CppHelper.maybeAddStaticLinkMarkerProvider(builder, ruleContext);
  }

  private static NestedSet<LinkerInput> collectTransitiveCcNativeLibraries(
      RuleContext ruleContext,
      List<? extends LinkerInput> dynamicLibraries) {
    NestedSetBuilder<LinkerInput> builder = NestedSetBuilder.linkOrder();
    builder.addAll(dynamicLibraries);
    for (CcNativeLibraryProvider dep :
      ruleContext.getPrerequisites("deps", Mode.TARGET, CcNativeLibraryProvider.class)) {
      builder.addTransitive(dep.getTransitiveCcNativeLibraries());
    }
    return builder.build();
  }

  private static boolean usePic(RuleContext ruleContext, CcToolchainProvider ccToolchainProvider) {
    if (isLinkShared(ruleContext)) {
      return ccToolchainProvider.usePicForDynamicLibraries();
    } else {
      return CppHelper.usePicForBinaries(ruleContext, ccToolchainProvider);
    }
  }
}
