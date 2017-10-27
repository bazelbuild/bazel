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

import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINK_INPUTS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_FILE;
import static java.util.Comparator.naturalOrder;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper.Info;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CollidingProvidesException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.FdoSupportProvider;
import com.google.devtools.build.lib.rules.cpp.FeatureSpecification;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.NoProcessing;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ObjcVariablesExtension.VariableCategory;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Constructs command lines for objc compilation, archiving, and linking. Uses the crosstool
 * infrastructure to register {@link CppCompileAction} and {@link CppLinkAction} instances, making
 * use of a provided toolchain.
 */
// TODO(b/65163377): Remove LegacyCompilationSupport and use only this implementation.
public class CrosstoolCompilationSupport extends CompilationSupport {

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final String NO_ENABLE_MODULES_FEATURE_NAME = "no_enable_modules";
  private static final String DEAD_STRIP_FEATURE_NAME = "dead_strip";
  /**
   * Enabled if this target's rule is not a test rule.  Binary stripping should not be applied in
   * the link step. TODO(b/36562173): Replace this behavior with a condition on bundle creation.
   *
   * <p>Note that the crosstool does not support feature negation in FlagSet.with_feature, which
   * is the mechanism used to condition linker arguments here.  Therefore, we expose
   * "is_not_test_target" instead of the more intuitive "is_test_target".
   */
  private static final String IS_NOT_TEST_TARGET_FEATURE_NAME = "is_not_test_target";
  /** Enabled if this target generates debug symbols in a dSYM file. */
  private static final String GENERATE_DSYM_FILE_FEATURE_NAME = "generate_dsym_file";
  /**
   * Enabled if this target does not generate debug symbols.
   *
   * <p>Note that the crosstool does not support feature negation in FlagSet.with_feature, which is
   * the mechanism used to condition linker arguments here. Therefore, we expose
   * "no_generate_debug_symbols" in addition to "generate_dsym_file"
   */
  private static final String NO_GENERATE_DEBUG_SYMBOLS_FEATURE_NAME = "no_generate_debug_symbols";

  private static final String GENERATE_LINKMAP_FEATURE_NAME = "generate_linkmap";

  /** Enabled if this target has objc sources in its transitive closure. */
  private static final String CONTAINS_OBJC = "contains_objc_sources";

  private static final ImmutableList<String> ACTIVATED_ACTIONS =
      ImmutableList.of(
          "objc-compile",
          "objc++-compile",
          "objc-archive",
          "objc-fully-link",
          "objc-executable",
          "objc++-executable",
          "assemble",
          "preprocess-assemble",
          "c-compile",
          "c++-compile");

  /**
   * Creates a new CompilationSupport instance that uses the c++ rule backend
   *
   * @param ruleContext the RuleContext for the calling target
   * @param outputGroupCollector a map that will be updated with output groups produced by compile
   *     action generation.
   */
  public CrosstoolCompilationSupport(
      RuleContext ruleContext, Map<String, NestedSet<Artifact>> outputGroupCollector) {
    this(
        ruleContext,
        ruleContext.getConfiguration(),
        ObjcRuleClasses.intermediateArtifacts(ruleContext),
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build(),
        /*useDeps=*/ true,
        outputGroupCollector,
        null,
        /*isTestRule=*/ false,
        /*usePch=*/ true);
  }

  /**
   * Creates a new CompilationSupport instance that uses the c++ rule backend
   *
   * @param ruleContext the RuleContext for the calling target
   * @param buildConfiguration the configuration for the calling target
   * @param intermediateArtifacts IntermediateArtifacts for deriving artifact paths
   * @param compilationAttributes attributes of the calling target
   * @param useDeps true if deps should be used
   * @param toolchain if not null overrides the default toolchain from the ruleContext.
   * @param usePch true if pch should be used
   */
  public CrosstoolCompilationSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      boolean useDeps,
      Map<String, NestedSet<Artifact>> outputGroupCollector,
      CcToolchainProvider toolchain,
      boolean isTestRule,
      boolean usePch) {
    super(
        ruleContext,
        buildConfiguration,
        intermediateArtifacts,
        compilationAttributes,
        useDeps,
        outputGroupCollector,
        toolchain,
        isTestRule,
        usePch);
  }

  @Override
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider, ExtraCompileArgs extraCompileArgs,
      Iterable<PathFragment> priorityHeaders,
      @Nullable CcToolchainProvider ccToolchain,
      @Nullable FdoSupportProvider fdoSupport) throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(ccToolchain);
    Preconditions.checkNotNull(fdoSupport);
    ObjcVariablesExtension.Builder extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setCompilationArtifacts(compilationArtifacts)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setConfiguration(buildConfiguration);
    CcLibraryHelper helper;

    if (compilationArtifacts.getArchive().isPresent()) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
      registerObjFilelistAction(getObjFiles(compilationArtifacts, intermediateArtifacts), objList);

      extension.addVariableCategory(VariableCategory.ARCHIVE_VARIABLES);

      helper =
          createCcLibraryHelper(
                  objcProvider,
                  compilationArtifacts,
                  extension.build(),
                  extraCompileArgs,
                  ccToolchain,
                  fdoSupport,
                  priorityHeaders)
              .setLinkType(LinkTargetType.OBJC_ARCHIVE)
              .addLinkActionInput(objList);
    } else {
      helper =
          createCcLibraryHelper(
              objcProvider,
              compilationArtifacts,
              extension.build(),
              extraCompileArgs,
              ccToolchain,
              fdoSupport,
              priorityHeaders);
    }

    Info info = helper.build();
    outputGroupCollector.putAll(info.getOutputGroups());

    registerHeaderScanningActions(info, objcProvider, compilationArtifacts);

    return this;
  }

  @Override
  protected CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Iterable<Artifact> inputArtifacts, Artifact outputArchive,
      @Nullable CcToolchainProvider ccToolchain, @Nullable FdoSupportProvider fdoSupport)
      throws InterruptedException {
    Preconditions.checkNotNull(ccToolchain);
    Preconditions.checkNotNull(fdoSupport);
    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();
    ObjcVariablesExtension extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setConfiguration(buildConfiguration)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setFullyLinkArchive(outputArchive)
            .addVariableCategory(VariableCategory.FULLY_LINK_VARIABLES)
            .build();
    CppLinkAction fullyLinkAction =
        new CppLinkActionBuilder(
                ruleContext,
                outputArchive,
                ccToolchain,
                fdoSupport,
                getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration, objcProvider),
                createObjcCppSemantics(
                    objcProvider, /* privateHdrs= */ ImmutableList.of(), /* pchHdr= */ null))
            .addActionInputs(objcProvider.getObjcLibraries())
            .addActionInputs(objcProvider.getCcLibraries())
            .addActionInputs(objcProvider.get(IMPORTED_LIBRARY).toSet())
            .setCrosstoolInputs(ccToolchain.getLink())
            .setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtension(extension)
            .build();
    ruleContext.registerAction(fullyLinkAction);

    return this;
  }

  private StrippingType getStrippingType(ExtraLinkArgs extraLinkArgs) {
    return Iterables.contains(extraLinkArgs, "-dynamiclib")
        ? StrippingType.DYNAMIC_LIB
        : StrippingType.DEFAULT;
  }

  @Override
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType,
      CcToolchainProvider toolchain)
      throws InterruptedException {
    Iterable<Artifact> prunedJ2ObjcArchives =
        computeAndStripPrunedJ2ObjcArchives(
            j2ObjcEntryClassProvider, j2ObjcMappingFileProvider, objcProvider);
    ImmutableList<Artifact> bazelBuiltLibraries =
        Iterables.isEmpty(prunedJ2ObjcArchives)
            ? objcProvider.getObjcLibraries()
            : substituteJ2ObjcPrunedLibraries(objcProvider);

    Artifact inputFileList = intermediateArtifacts.linkerObjList();
    ImmutableSet<Artifact> forceLinkArtifacts = getForceLoadArtifacts(objcProvider);

    Iterable<Artifact> objFiles =
        Iterables.concat(
            bazelBuiltLibraries, objcProvider.get(IMPORTED_LIBRARY), objcProvider.getCcLibraries());
    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    objFiles = Iterables.filter(objFiles, Predicates.not(Predicates.in(forceLinkArtifacts)));

    registerObjFilelistAction(objFiles, inputFileList);

    LinkTargetType linkType = (objcProvider.is(Flag.USES_CPP))
        ? LinkTargetType.OBJCPP_EXECUTABLE
        : LinkTargetType.OBJC_EXECUTABLE;

    ObjcVariablesExtension.Builder extensionBuilder =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setConfiguration(buildConfiguration)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setFrameworkNames(frameworkNames(objcProvider))
            .setLibraryNames(libraryNames(objcProvider))
            .setForceLoadArtifacts(getForceLoadArtifacts(objcProvider))
            .setAttributeLinkopts(attributes.linkopts())
            .addVariableCategory(VariableCategory.EXECUTABLE_LINKING_VARIABLES);

    Artifact binaryToLink = getBinaryToLink();
    FdoSupportProvider fdoSupport =
        CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
    CppLinkActionBuilder executableLinkAction =
        new CppLinkActionBuilder(
                ruleContext,
                binaryToLink,
                toolchain,
                fdoSupport,
                getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, objcProvider),
                createObjcCppSemantics(
                    objcProvider, /* privateHdrs= */ ImmutableList.of(), /* pchHdr= */ null))
            .setMnemonic("ObjcLink")
            .addActionInputs(bazelBuiltLibraries)
            .addActionInputs(objcProvider.getCcLibraries())
            .addTransitiveActionInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveActionInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveActionInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .addTransitiveActionInputs(objcProvider.get(LINK_INPUTS))
            .setCrosstoolInputs(toolchain.getLink())
            .addActionInputs(prunedJ2ObjcArchives)
            .addActionInputs(extraLinkInputs)
            .addActionInput(inputFileList)
            .setLinkType(linkType)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .addLinkopts(ImmutableList.copyOf(extraLinkArgs));

    if (objcConfiguration.generateDsym()) {
      Artifact dsymBundleZip = intermediateArtifacts.tempDsymBundleZip(dsymOutputType);
      extensionBuilder
          .setDsymBundleZip(dsymBundleZip)
          .addVariableCategory(VariableCategory.DSYM_VARIABLES);
      registerDsymActions(dsymOutputType);
      executableLinkAction.addActionOutput(dsymBundleZip);
    }

    if (objcConfiguration.generateLinkmap()) {
      Artifact linkmap = intermediateArtifacts.linkmap();
      extensionBuilder
          .setLinkmap(linkmap)
          .addVariableCategory(VariableCategory.LINKMAP_VARIABLES);
      executableLinkAction.addActionOutput(linkmap);
    }

    if (appleConfiguration.getBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
      Artifact bitcodeSymbolMap = intermediateArtifacts.bitcodeSymbolMap();
      extensionBuilder
          .setBitcodeSymbolMap(bitcodeSymbolMap)
          .addVariableCategory(VariableCategory.BITCODE_VARIABLES);
      executableLinkAction.addActionOutput(bitcodeSymbolMap);
    }

    executableLinkAction.addVariablesExtension(extensionBuilder.build());
    ruleContext.registerAction(executableLinkAction.build());

    if (objcConfiguration.shouldStripBinary()) {
      registerBinaryStripAction(binaryToLink, getStrippingType(extraLinkArgs));
    }

    return this;
  }

  private IncludeProcessing createIncludeProcessing(
      Iterable<Artifact> privateHdrs, ObjcProvider objcProvider, @Nullable Artifact pchHdr) {
    if (isHeaderThinningEnabled()) {
      Iterable<Artifact> potentialInputs =
          Iterables.concat(
              privateHdrs,
              objcProvider.get(HEADER),
              objcProvider.get(STATIC_FRAMEWORK_FILE),
              objcProvider.get(DYNAMIC_FRAMEWORK_FILE));
      if (pchHdr != null) {
        potentialInputs = Iterables.concat(potentialInputs, ImmutableList.of(pchHdr));
      }
      return new HeaderThinning(potentialInputs);
    } else {
      return new NoProcessing();
    }
  }

  private CcLibraryHelper createCcLibraryHelper(
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      VariablesExtension extension,
      ExtraCompileArgs extraCompileArgs,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      Iterable<PathFragment> priorityHeaders) {
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    Collection<Artifact> arcSources = ImmutableSortedSet.copyOf(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources =
        ImmutableSortedSet.copyOf(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs =
        ImmutableSortedSet.copyOf(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs =
        Stream.concat(
                Streams.stream(attributes.hdrs()),
                Streams.stream(compilationArtifacts.getAdditionalHdrs()))
            .collect(toImmutableSortedSet(naturalOrder()));
    Artifact pchHdr = getPchFile().orNull();
    ObjcCppSemantics semantics = createObjcCppSemantics(objcProvider, privateHdrs, pchHdr);
    CcLibraryHelper result =
        new CcLibraryHelper(
                ruleContext,
                semantics,
                getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration, objcProvider),
                CcLibraryHelper.SourceCategory.CC_AND_OBJC,
                ccToolchain,
                fdoSupport,
                buildConfiguration)
            .addSources(arcSources, ImmutableMap.of("objc_arc", ""))
            .addSources(nonArcSources, ImmutableMap.of("no_objc_arc", ""))
            .addSources(privateHdrs)
            .addDefines(objcProvider.get(DEFINE))
            .enableCompileProviders()
            .addPublicHeaders(publicHdrs)
            .addPrecompiledFiles(precompiledFiles)
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            // Not all our dependencies need to export cpp information.
            // For example, objc_proto_library can depend on a proto_library rule that does not
            // generate C++ protos.
            .setCheckDepsGenerateCpp(false)
            .setCopts(
                ImmutableList.<String>builder()
                    .addAll(getCompileRuleCopts())
                    .addAll(
                        ruleContext
                            .getFragment(ObjcConfiguration.class)
                            .getCoptsForCompilationMode())
                    .addAll(extraCompileArgs)
                    .build())
            .addIncludeDirs(priorityHeaders)
            .addIncludeDirs(objcProvider.get(INCLUDE))
            .addSystemIncludeDirs(objcProvider.get(INCLUDE_SYSTEM))
            .setCppModuleMap(intermediateArtifacts.moduleMap())
            .setLinkedArtifactNameSuffix(intermediateArtifacts.archiveFileNameSuffix())
            .setPropagateModuleMapToCompileAction(false)
            .setNeverLink(true)
            .addVariableExtension(extension);

    if (pchHdr != null) {
      result.addNonModuleMapHeader(pchHdr);
    }
    if (!useDeps) {
      result.doNotUseDeps();
    }
    if (getCustomModuleMap(ruleContext).isPresent()) {
      result.doNotGenerateModuleMap();
    }
    return result;
  }

  private ObjcCppSemantics createObjcCppSemantics(
      ObjcProvider objcProvider, Collection<Artifact> privateHdrs, Artifact pchHdr) {
    return new ObjcCppSemantics(
        objcProvider,
        createIncludeProcessing(privateHdrs, objcProvider, pchHdr),
        ruleContext.getFragment(ObjcConfiguration.class),
        isHeaderThinningEnabled(),
        intermediateArtifacts,
        buildConfiguration);
  }

  private FeatureConfiguration getFeatureConfiguration(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration configuration,
      ObjcProvider objcProvider) {
    boolean isHost = ruleContext.getConfiguration().isHostConfiguration();
    ImmutableSet.Builder<String> activatedCrosstoolSelectables =
        ImmutableSet.<String>builder()
            .addAll(ccToolchain.getFeatures().getDefaultFeatures())
            .addAll(ACTIVATED_ACTIONS)
            .addAll(
                ruleContext
                    .getFragment(AppleConfiguration.class)
                    .getBitcodeMode()
                    .getFeatureNames())
            // We create a module map by default to allow for Swift interop.
            .add(CppRuleClasses.MODULE_MAPS)
            .add(CppRuleClasses.COMPILE_ALL_MODULES)
            .add(CppRuleClasses.EXCLUDE_PRIVATE_HEADERS_IN_MODULE_MAPS)
            .add(CppRuleClasses.ONLY_DOTH_HEADERS_IN_MODULE_MAPS)
            .add(CppRuleClasses.COMPILE_ACTION_FLAGS_IN_FLAG_SET)
            .add(CppRuleClasses.DEPENDENCY_FILE)
            .add(CppRuleClasses.INCLUDE_PATHS)
            .add(isHost ? "host" : "nonhost")
            .add(configuration.getCompilationMode().toString());

    if (configuration.getFragment(ObjcConfiguration.class).moduleMapsEnabled()
        && !getCustomModuleMap(ruleContext).isPresent()) {
      activatedCrosstoolSelectables.add(OBJC_MODULE_FEATURE_NAME);
    }
    if (!CompilationAttributes.Builder.fromRuleContext(ruleContext).build().enableModules()) {
      activatedCrosstoolSelectables.add(NO_ENABLE_MODULES_FEATURE_NAME);
    }
    if (configuration.getFragment(ObjcConfiguration.class).shouldStripBinary()) {
      activatedCrosstoolSelectables.add(DEAD_STRIP_FEATURE_NAME);
    }
    if (getPchFile().isPresent()) {
      activatedCrosstoolSelectables.add("pch");
    }
    if (!isTestRule) {
      activatedCrosstoolSelectables.add(IS_NOT_TEST_TARGET_FEATURE_NAME);
    }
    if (configuration.getFragment(ObjcConfiguration.class).generateDsym()) {
      activatedCrosstoolSelectables.add(GENERATE_DSYM_FILE_FEATURE_NAME);
    } else {
      activatedCrosstoolSelectables.add(NO_GENERATE_DEBUG_SYMBOLS_FEATURE_NAME);
    }
    if (configuration.getFragment(ObjcConfiguration.class).generateLinkmap()) {
      activatedCrosstoolSelectables.add(GENERATE_LINKMAP_FEATURE_NAME);
    }
    AppleBitcodeMode bitcodeMode =
        configuration.getFragment(AppleConfiguration.class).getBitcodeMode();
    if (bitcodeMode != AppleBitcodeMode.NONE) {
      activatedCrosstoolSelectables.addAll(bitcodeMode.getFeatureNames());
    }
    if (objcProvider.is(Flag.USES_OBJC)) {
      activatedCrosstoolSelectables.add(CONTAINS_OBJC);
    }

    activatedCrosstoolSelectables.addAll(ruleContext.getFeatures());
    try {
      return toolchain
          .getFeatures()
          .getFeatureConfiguration(
              FeatureSpecification.create(
                  activatedCrosstoolSelectables.build(), ImmutableSet.<String>of()));
    } catch (CollidingProvidesException e) {
      ruleContext.ruleError(e.getMessage());
      return FeatureConfiguration.EMPTY;
    }
  }

  private static ImmutableList<Artifact> getObjFiles(
      CompilationArtifacts compilationArtifacts, IntermediateArtifacts intermediateArtifacts) {
    ImmutableList.Builder<Artifact> result = new ImmutableList.Builder<>();
    for (Artifact sourceFile : compilationArtifacts.getSrcs()) {
      result.add(intermediateArtifacts.objFile(sourceFile));
    }
    for (Artifact nonArcSourceFile : compilationArtifacts.getNonArcSrcs()) {
      result.add(intermediateArtifacts.objFile(nonArcSourceFile));
    }
    result.addAll(compilationArtifacts.getPrecompiledSrcs());
    return result.build();
  }

  private void registerHeaderScanningActions(
      Info info, ObjcProvider objcProvider, CompilationArtifacts compilationArtifacts) {
    // PIC is not used for Obj-C builds, if that changes this method will need to change
    if (!isHeaderThinningEnabled()
        || info.getCcCompilationOutputs().getObjectFiles(false).isEmpty()) {
      return;
    }

    ImmutableList.Builder<ObjcHeaderThinningInfo> headerThinningInfos = ImmutableList.builder();
    AnalysisEnvironment analysisEnvironment = ruleContext.getAnalysisEnvironment();
    for (Artifact objectFile : info.getCcCompilationOutputs().getObjectFiles(false)) {
      ActionAnalysisMetadata generatingAction =
          analysisEnvironment.getLocalGeneratingAction(objectFile);
      if (generatingAction instanceof CppCompileAction) {
        CppCompileAction action = (CppCompileAction) generatingAction;
        Artifact sourceFile = action.getSourceFile();
        if (!sourceFile.isTreeArtifact()
            && SOURCES_FOR_HEADER_THINNING.matches(sourceFile.getFilename())) {
          headerThinningInfos.add(
              new ObjcHeaderThinningInfo(
                  sourceFile,
                  intermediateArtifacts.headersListFile(sourceFile),
                  action.getCompilerOptions()));
        }
      }
    }
    registerHeaderScanningActions(headerThinningInfos.build(), objcProvider, compilationArtifacts);
  }
}
