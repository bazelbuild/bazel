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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.cpp.Link.LINK_LIBRARY_FILETYPES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINK_INPUTS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HEADERS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PRECOMPILED_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.STRIP;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Comparator.naturalOrder;
import static java.util.stream.Collectors.toCollection;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CollidingProvidesException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.FdoContext;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.rules.cpp.IncludeScanning;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.NoProcessing;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.cpp.UmbrellaHeaderAction;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ObjcVariablesExtension.VariableCategory;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * Support for rules that compile sources. Provides ways to determine files that should be output,
 * registering Xcode settings and generating the various actions that might be needed for
 * compilation.
 *
 * <p>A subclass should express a particular strategy for compile and link action registration.
 * Subclasses should implement the API without adding new visible methods - rule implementations
 * should be able to use a {@link CompilationSupport} instance to compile and link source without
 * knowing the subclass being used.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
public class CompilationSupport {

  @VisibleForTesting static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";

  @VisibleForTesting
  static final String MODULES_CACHE_PATH_WARNING =
      "setting '-fmodules-cache-path' manually in copts is unsupported";

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

  // Flags for clang 6.1(xcode 6.4)
  @VisibleForTesting
  static final ImmutableList<String> CLANG_GCOV_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-arcs", "-ftest-coverage");

  @VisibleForTesting
  static final ImmutableList<String> CLANG_LLVM_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-instr-generate", "-fcoverage-mapping");

  // These are added by Xcode when building, because the simulator is built on OSX
  // frameworks so we aim compile to match the OSX objc runtime.
  @VisibleForTesting
  static final ImmutableList<String> SIMULATOR_COMPILE_FLAGS =
      ImmutableList.of(
          "-fexceptions", "-fasm-blocks", "-fobjc-abi-version=2", "-fobjc-legacy-dispatch");

  /**
   * Frameworks implicitly linked to iOS, watchOS, and tvOS binaries when using legacy compilation.
   */
  @VisibleForTesting
  static final NestedSet<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS =
      NestedSetBuilder.create(
          Order.STABLE_ORDER, new SdkFramework("Foundation"), new SdkFramework("UIKit"));

  /** Selects cc libraries that have alwayslink=1. */
  private static final Predicate<Artifact> ALWAYS_LINKED_CC_LIBRARY =
      input -> LINK_LIBRARY_FILETYPES.matches(input.getFilename());

  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final String NO_ENABLE_MODULES_FEATURE_NAME = "no_enable_modules";
  private static final String DEAD_STRIP_FEATURE_NAME = "dead_strip";

  /**
   * Enabled if this target's rule is not a test rule. Binary stripping should not be applied in the
   * link step. TODO(b/36562173): Replace this behavior with a condition on bundle creation.
   *
   * <p>Note that the crosstool does not support feature negation in FlagSet.with_feature, which is
   * the mechanism used to condition linker arguments here. Therefore, we expose
   * "is_not_test_target" instead of the more intuitive "is_test_target".
   */
  private static final String IS_NOT_TEST_TARGET_FEATURE_NAME = "is_not_test_target";

  private static final String GENERATE_LINKMAP_FEATURE_NAME = "generate_linkmap";

  private static final String XCODE_VERSION_FEATURE_NAME_PREFIX = "xcode_";

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

  /** The kind of include processing to use. */
  enum IncludeProcessingType {
    INCLUDE_SCANNING,
    NO_PROCESSING;
  }

  /** Returns the location of the xcrunwrapper tool. */
  public static final FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$xcrunwrapper", TransitionMode.HOST);
  }

  /** Returns the location of the libtool tool. */
  public static final FilesToRunProvider libtool(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite(
        ObjcRuleClasses.LIBTOOL_ATTRIBUTE, TransitionMode.HOST);
  }

  /**
   * Files which can be instrumented along with the attributes in which they may occur and the
   * attributes along which they are propagated from dependencies (via {@link
   * InstrumentedFilesInfo}).
   */
  private static final InstrumentationSpec INSTRUMENTATION_SPEC =
      new InstrumentationSpec(
              FileTypeSet.of(ObjcRuleClasses.NON_CPP_SOURCES, ObjcRuleClasses.CPP_SOURCES, HEADERS))
          .withSourceAttributes("srcs", "non_arc_srcs", "hdrs")
          .withDependencyAttributes("deps", "data", "binary", "xctest_app");

  /** Defines a library that contains the transitive closure of dependencies. */
  public static final SafeImplicitOutputsFunction FULLY_LINKED_LIB =
      fromTemplates("%{name}_fully_linked.a");

  /** Create and return the include processing to be used. */
  private IncludeProcessing createIncludeProcessing() {
    switch (includeProcessingType) {
      case INCLUDE_SCANNING:
        return IncludeScanning.INSTANCE;
      default:
        return NoProcessing.INSTANCE;
    }
  }

  private static ImmutableList<String> pathsToIncludeArgs(Iterable<PathFragment> paths) {
    ImmutableList.Builder<String> builder = ImmutableList.<String>builder();
    for (PathFragment path : paths) {
      builder.add("-I" + path);
    }
    return builder.build();
  }

  private CompilationInfo compile(
      ObjcCompilationContext objcCompilationContext,
      VariablesExtension extension,
      ExtraCompileArgs extraCompileArgs,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      List<PathFragment> priorityHeaders,
      Collection<Artifact> sources,
      Collection<Artifact> privateHdrs,
      Collection<Artifact> publicHdrs,
      Artifact pchHdr,
      ObjcCppSemantics semantics,
      String purpose,
      boolean generateModuleMap,
      boolean shouldProcessHeaders)
      throws RuleErrorException, InterruptedException {
    CcCompilationHelper result =
        new CcCompilationHelper(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                CppHelper.getGrepIncludes(ruleContext),
                semantics,
                getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration),
                CcCompilationHelper.SourceCategory.CC_AND_OBJC,
                ccToolchain,
                fdoContext,
                buildConfiguration,
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()),
                shouldProcessHeaders)
            .addSources(sources)
            .addPublicHeaders(publicHdrs)
            .addPublicTextualHeaders(objcCompilationContext.getPublicTextualHeaders())
            .addPrivateHeaders(privateHdrs)
            .addDefines(
                NestedSetBuilder.wrap(Order.LINK_ORDER, objcCompilationContext.getDefines()))
            .addIncludeDirs(priorityHeaders)
            .addIncludeDirs(objcCompilationContext.getIncludes())
            .addSystemIncludeDirs(objcCompilationContext.getSystemIncludes())
            .addQuoteIncludeDirs(objcCompilationContext.getQuoteIncludes())
            .addCcCompilationContexts(objcCompilationContext.getDepCcCompilationContexts())
            .setCopts(
                ImmutableList.<String>builder()
                    .addAll(getCompileRuleCopts())
                    .addAll(
                        ruleContext
                            .getFragment(ObjcConfiguration.class)
                            .getCoptsForCompilationMode())
                    .addAll(extraCompileArgs)
                    .addAll(
                        pathsToIncludeArgs(objcCompilationContext.getStrictDependencyIncludes()))
                    .build())
            .setCppModuleMap(intermediateArtifacts.moduleMap())
            .setPropagateModuleMapToCompileAction(false)
            .addVariableExtension(extension)
            .setPurpose(purpose)
            .setCodeCoverageEnabled(CcCompilationHelper.isCodeCoverageEnabled(ruleContext))
            .setHeadersCheckingMode(semantics.determineHeadersCheckingMode(ruleContext));

    if (pchHdr != null) {
      result.addPublicTextualHeaders(ImmutableList.of(pchHdr));
    }

    if (getCustomModuleMap(ruleContext).isPresent() || !generateModuleMap) {
      result.doNotGenerateModuleMap();
    }

    return result.compile();
  }

  private static class CompilationResult {
    private final CcCompilationContext ccCompilationContext;
    private final CcCompilationOutputs ccCompilationOutputs;
    private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;

    public CompilationResult(
        CcCompilationContext ccCompilationContext,
        CcCompilationOutputs ccCompilationOutputs,
        ImmutableMap<String, NestedSet<Artifact>> outputGroups) {
      this.ccCompilationContext = ccCompilationContext;
      this.ccCompilationOutputs = ccCompilationOutputs;
      this.outputGroups = outputGroups;
    }

    public CcCompilationContext getCcCompilationContext() {
      return ccCompilationContext;
    }

    public CcCompilationOutputs getCcCompilationOutputs() {
      return ccCompilationOutputs;
    }

    public ImmutableMap<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }
  }

  private CompilationResult ccCompileAndLink(
      ObjcCompilationContext objcCompilationContext,
      CompilationArtifacts compilationArtifacts,
      ObjcVariablesExtension.Builder extensionBuilder,
      ExtraCompileArgs extraCompileArgs,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      List<PathFragment> priorityHeaders,
      LinkTargetType linkType,
      Artifact linkActionInput)
      throws RuleErrorException, InterruptedException {
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    ImmutableSortedSet<Artifact> arcSources =
        ImmutableSortedSet.copyOf(compilationArtifacts.getSrcs());
    ImmutableSortedSet<Artifact> nonArcSources =
        ImmutableSortedSet.copyOf(compilationArtifacts.getNonArcSrcs());
    ImmutableSortedSet<Artifact> privateHdrs =
        ImmutableSortedSet.copyOf(compilationArtifacts.getPrivateHdrs());
    ImmutableSortedSet<Artifact> publicHdrs =
        Stream.concat(
                attributes.hdrs().toList().stream(),
                compilationArtifacts.getAdditionalHdrs().toList().stream())
            .collect(toImmutableSortedSet(naturalOrder()));
    Artifact pchHdr = getPchFile().orNull();
    ObjcCppSemantics semantics = createObjcCppSemantics();

    String purpose = String.format("%s_objc_arc", semantics.getPurpose());
    extensionBuilder.setArcEnabled(true);
    CompilationInfo objcArcCompilationInfo =
        compile(
            objcCompilationContext,
            extensionBuilder.build(),
            extraCompileArgs,
            ccToolchain,
            fdoContext,
            priorityHeaders,
            arcSources,
            privateHdrs,
            publicHdrs,
            pchHdr,
            semantics,
            purpose,
            /* generateModuleMap= */ true,
            /* shouldProcessHeaders= */ true);

    purpose = String.format("%s_non_objc_arc", semantics.getPurpose());
    extensionBuilder.setArcEnabled(false);
    CompilationInfo nonObjcArcCompilationInfo =
        compile(
            objcCompilationContext,
            extensionBuilder.build(),
            extraCompileArgs,
            ccToolchain,
            fdoContext,
            priorityHeaders,
            nonArcSources,
            privateHdrs,
            publicHdrs,
            pchHdr,
            semantics,
            purpose,
            // Only generate the module map once (see above) and re-use it here.
            /* generateModuleMap= */ false,
            // We only need to validate headers once, in arc compilation above.
            /* shouldProcessHeaders= */ false);

    FeatureConfiguration featureConfiguration =
        getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration);
    CcLinkingHelper resultLink =
        new CcLinkingHelper(
                ruleContext,
                ruleContext.getLabel(),
                ruleContext,
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                fdoContext,
                buildConfiguration,
                ruleContext.getFragment(CppConfiguration.class),
                ruleContext.getSymbolGenerator(),
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
            .setTestOrTestOnlyTarget(ruleContext.isTestTarget() || ruleContext.isTestOnlyTarget())
            .addCcLinkingContexts(
                CppHelper.getLinkingContextsFromDeps(
                    ImmutableList.copyOf(
                        ruleContext.getPrerequisites("deps", TransitionMode.TARGET))))
            .setLinkedArtifactNameSuffix(intermediateArtifacts.archiveFileNameSuffix())
            .setNeverLink(true)
            .addVariableExtension(extensionBuilder.build());

    if (linkType != null) {
      resultLink.setStaticLinkType(linkType);
    }

    if (linkActionInput != null) {
      resultLink.addLinkActionInput(linkActionInput);
    }

    CcCompilationContext.Builder ccCompilationContextBuilder =
        CcCompilationContext.builder(
            ruleContext, ruleContext.getConfiguration(), ruleContext.getLabel());
    // Do a re-exporting merge of the ARC and non-ARC contexts so that the direct headers are
    // preserved in the unified context.
    ccCompilationContextBuilder.mergeDependentCcCompilationContexts(
        Arrays.asList(
            objcArcCompilationInfo.getCcCompilationContext(),
            nonObjcArcCompilationInfo.getCcCompilationContext()),
        ImmutableList.of());
    ccCompilationContextBuilder.setPurpose(
        String.format("%s_merged_arc_non_arc_objc", semantics.getPurpose()));

    CcCompilationOutputs precompiledFilesObjects =
        CcCompilationOutputs.builder()
            .addObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ false))
            .addPicObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ true))
            .build();

    CcCompilationOutputs compilationOutputs =
        CcCompilationOutputs.builder()
            .merge(objcArcCompilationInfo.getCcCompilationOutputs())
            .merge(nonObjcArcCompilationInfo.getCcCompilationOutputs())
            .merge(precompiledFilesObjects)
            .build();

    if (!compilationOutputs.isEmpty()) {
      resultLink.link(compilationOutputs);
    }

    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    Map<String, NestedSet<Artifact>> arcOutputGroups =
        CcCompilationHelper.buildOutputGroupsForEmittingCompileProviders(
            objcArcCompilationInfo.getCcCompilationOutputs(),
            objcArcCompilationInfo.getCcCompilationContext(),
            cppConfiguration,
            ccToolchain,
            featureConfiguration,
            ruleContext);

    Map<String, NestedSet<Artifact>> nonArcOutputGroups =
        CcCompilationHelper.buildOutputGroupsForEmittingCompileProviders(
            nonObjcArcCompilationInfo.getCcCompilationOutputs(),
            nonObjcArcCompilationInfo.getCcCompilationContext(),
            cppConfiguration,
            ccToolchain,
            featureConfiguration,
            ruleContext);

    Map<String, NestedSet<Artifact>> mergedOutputGroups =
        CcCommon.mergeOutputGroups(ImmutableList.of(arcOutputGroups, nonArcOutputGroups));

    return new CompilationResult(
        ccCompilationContextBuilder.build(),
        compilationOutputs,
        ImmutableMap.copyOf(mergedOutputGroups));
  }

  ObjcCppSemantics createObjcCppSemantics() {
    return new ObjcCppSemantics(
        includeProcessingType,
        createIncludeProcessing(),
        ruleContext.getFragment(ObjcConfiguration.class),
        intermediateArtifacts,
        buildConfiguration,
        attributes.enableModules());
  }

  private FeatureConfiguration getFeatureConfiguration(
      RuleContext ruleContext, CcToolchainProvider ccToolchain, BuildConfiguration configuration) {
    boolean isTool = ruleContext.getConfiguration().isToolConfiguration();
    ImmutableSet.Builder<String> activatedCrosstoolSelectables =
        ImmutableSet.<String>builder()
            .addAll(ccToolchain.getFeatures().getDefaultFeaturesAndActionConfigs())
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
            .add(CppRuleClasses.DEPENDENCY_FILE)
            .add(CppRuleClasses.INCLUDE_PATHS)
            .add(isTool ? "host" : "nonhost")
            .add(configuration.getCompilationMode().toString());

    if (configuration.getFragment(ObjcConfiguration.class).moduleMapsEnabled()
        && !getCustomModuleMap(ruleContext).isPresent()) {
      activatedCrosstoolSelectables.add(OBJC_MODULE_FEATURE_NAME);
    }
    if (!attributes.enableModules()) {
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
    if (objcConfiguration.generateDsym()) {
      activatedCrosstoolSelectables.add(CppRuleClasses.GENERATE_DSYM_FILE_FEATURE_NAME);
    } else {
      activatedCrosstoolSelectables.add(CppRuleClasses.NO_GENERATE_DEBUG_SYMBOLS_FEATURE_NAME);
    }
    if (configuration.getFragment(ObjcConfiguration.class).generateLinkmap()) {
      activatedCrosstoolSelectables.add(GENERATE_LINKMAP_FEATURE_NAME);
    }
    AppleBitcodeMode bitcodeMode =
        configuration.getFragment(AppleConfiguration.class).getBitcodeMode();
    if (bitcodeMode != AppleBitcodeMode.NONE) {
      activatedCrosstoolSelectables.addAll(bitcodeMode.getFeatureNames());
    }
    // Add a feature identifying the Xcode version so CROSSTOOL authors can enable flags for
    // particular versions of Xcode. To ensure consistency across platforms, use exactly two
    // components in the version number.
    activatedCrosstoolSelectables.add(
        XCODE_VERSION_FEATURE_NAME_PREFIX
            + XcodeConfig.getXcodeConfigInfo(ruleContext)
                .getXcodeVersion()
                .toStringWithComponents(2));

    activatedCrosstoolSelectables.addAll(ruleContext.getFeatures());

    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    activatedCrosstoolSelectables.addAll(CcCommon.getCoverageFeatures(cppConfiguration));

    try {
      ImmutableSet<String> activatedCrosstoolSelectablesSet;
      if (!ccToolchain.supportsHeaderParsing()) {
        // TODO(b/159096411): Remove once supports_header_parsing has been removed from the
        // cc_toolchain rule.
        activatedCrosstoolSelectablesSet =
            activatedCrosstoolSelectables.build().stream()
                .filter(feature -> !feature.equals(CppRuleClasses.PARSE_HEADERS))
                .collect(toImmutableSet());
      } else {
        activatedCrosstoolSelectablesSet = activatedCrosstoolSelectables.build();
      }
      return ccToolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectablesSet);
    } catch (CollidingProvidesException e) {
      ruleContext.ruleError(e.getMessage());
      return FeatureConfiguration.EMPTY;
    }
  }

  /** Iterable wrapper providing strong type safety for arguments to binary linking. */
  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(String... args) {
      super(args);
    }

    ExtraLinkArgs(Iterable<String> args) {
      super(args);
    }
  }

  /** Iterable wrapper providing strong type safety for extra compile flags. */
  static final class ExtraCompileArgs extends IterableWrapper<String> {
    static final ExtraCompileArgs NONE = new ExtraCompileArgs();

    ExtraCompileArgs(String... args) {
      super(args);
    }
  }

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT = "File '%s' is in both srcs and hdrs.";

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT =
      "File '%s' is present in both srcs and non_arc_srcs which is forbidden.";

  @VisibleForTesting
  static final String BOTH_MODULE_NAME_AND_MODULE_MAP_SPECIFIED =
      "Specifying both module_name and module_map is invalid, please remove one of them.";

  static final ImmutableList<String> DEFAULT_COMPILER_FLAGS = ImmutableList.of("-DOS_IOS");

  /** Returns information about the given rule's compilation artifacts. */
  // TODO(bazel-team): Remove this information from ObjcCommon and move it internal to this class.
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext) {
    return compilationArtifacts(ruleContext, ObjcRuleClasses.intermediateArtifacts(ruleContext));
  }

  /**
   * Returns information about the given rule's compilation artifacts. Dependencies specified in the
   * current rule's attributes are obtained via {@code ruleContext}. Output locations are determined
   * using the given {@code intermediateArtifacts} object. The fact that these are distinct objects
   * allows the caller to generate compilation actions pertaining to a configuration separate from
   * the current rule's configuration.
   */
  static CompilationArtifacts compilationArtifacts(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts) {
    PrerequisiteArtifacts srcs =
        ruleContext
            .getPrerequisiteArtifacts("srcs", TransitionMode.TARGET)
            .errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(
            ruleContext
                .getPrerequisiteArtifacts("non_arc_srcs", TransitionMode.TARGET)
                .errorsForNonMatching(NON_ARC_SRCS_TYPE)
                .list())
        .addPrivateHdrs(srcs.filter(HEADERS).list())
        .addPrecompiledSrcs(srcs.filter(PRECOMPILED_SRCS_TYPE).list())
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  /** Returns a list of framework library search paths. */
  static ImmutableList<String> frameworkLibrarySearchPaths(ObjcProvider provider) {
    ImmutableList.Builder<String> searchPaths = new ImmutableList.Builder<>();
    return searchPaths
        // Add library search paths corresponding to custom (non-SDK) frameworks. For each framework
        // foo/bar.framework, include "foo" as a search path.
        .addAll(provider.staticFrameworkPaths().toList())
        .addAll(provider.dynamicFrameworkPaths().toList())
        .build();
  }

  private final RuleContext ruleContext;
  private final BuildConfiguration buildConfiguration;
  private final ObjcConfiguration objcConfiguration;
  private final AppleConfiguration appleConfiguration;
  private final CompilationAttributes attributes;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Map<String, NestedSet<Artifact>> outputGroupCollector;
  private final ImmutableList.Builder<Artifact> objectFilesCollector;
  private final CcToolchainProvider toolchain;
  private final boolean isTestRule;
  private final boolean usePch;
  private final IncludeProcessingType includeProcessingType;
  private Optional<ObjcProvider> objcProvider;

  private void setObjcProvider(ObjcProvider objcProvider) {
    checkState(!this.objcProvider.isPresent());
    this.objcProvider = Optional.of(objcProvider);
  }

  public ObjcProvider getObjcProvider() {
    checkState(objcProvider.isPresent());
    return objcProvider.get();
  }

  /**
   * Creates a new compilation support for the given rule and build configuration.
   *
   * <p>All actions will be created under the given build configuration, which may be different than
   * the current rule context configuration.
   *
   * <p>The compilation and linking flags will be retrieved from the given compilation attributes.
   * The names of the generated artifacts will be retrieved from the given intermediate artifacts.
   *
   * <p>By instantiating multiple compilation supports for the same rule but with intermediate
   * artifacts with different output prefixes, multiple archives can be compiled for the same rule
   * context.
   */
  private CompilationSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      Map<String, NestedSet<Artifact>> outputGroupCollector,
      ImmutableList.Builder<Artifact> objectFilesCollector,
      CcToolchainProvider toolchain,
      boolean isTestRule,
      boolean usePch)
      throws InterruptedException {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
    this.attributes = compilationAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
    this.isTestRule = isTestRule;
    this.outputGroupCollector = outputGroupCollector;
    this.objectFilesCollector = objectFilesCollector;
    this.objcProvider = Optional.absent();
    this.usePch = usePch;
    if (toolchain == null
        && ruleContext
            .attributes()
            .has(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, BuildType.LABEL)) {
      toolchain = CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    }

    this.toolchain = toolchain;

    if (objcConfiguration.shouldScanIncludes()) {
      includeProcessingType = IncludeProcessingType.INCLUDE_SCANNING;
    } else {
      includeProcessingType = IncludeProcessingType.NO_PROCESSING;
    }
  }

  /** Builder for {@link CompilationSupport} */
  public static class Builder {
    private RuleContext ruleContext;
    private BuildConfiguration buildConfiguration;
    private IntermediateArtifacts intermediateArtifacts;
    private CompilationAttributes compilationAttributes;
    private Map<String, NestedSet<Artifact>> outputGroupCollector;
    private ImmutableList.Builder<Artifact> objectFilesCollector;
    private CcToolchainProvider toolchain;
    private boolean isTestRule = false;
    private boolean usePch = true;

    /** Sets the {@link RuleContext} for the calling target. */
    public Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
      return this;
    }

    /** Sets the {@link BuildConfiguration} for the calling target. */
    public Builder setConfig(BuildConfiguration buildConfiguration) {
      this.buildConfiguration = buildConfiguration;
      return this;
    }

    /** Sets {@link IntermediateArtifacts} for deriving artifact paths. */
    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    /** Sets {@link CompilationAttributes} for the calling target. */
    public Builder setCompilationAttributes(CompilationAttributes compilationAttributes) {
      this.compilationAttributes = compilationAttributes;
      return this;
    }

    /**
     * Sets that this {@link CompilationSupport} will not use the pch from the rule context in
     * determining compilation actions.
     */
    public Builder doNotUsePch() {
      this.usePch = false;
      return this;
    }

    /** Indicates that this CompilationSupport is for use in a test rule. */
    public Builder setIsTestRule() {
      this.isTestRule = true;
      return this;
    }

    /**
     * Causes the provided map to be updated with output groups produced by compile action
     * registration.
     *
     * <p>This map is intended to be mutated by {@link
     * CompilationSupport#registerCompileAndArchiveActions}. The added output groups should be
     * exported by the calling rule class implementation.
     */
    public Builder setOutputGroupCollector(Map<String, NestedSet<Artifact>> outputGroupCollector) {
      this.outputGroupCollector = outputGroupCollector;
      return this;
    }

    /**
     * Set a collector for the object files produced by compile action registration.
     *
     * <p>The object files are intended to be added by {@link
     * CompilationSupport#registerCompileAndArchiveActions}.
     */
    public Builder setObjectFilesCollector(ImmutableList.Builder<Artifact> objectFilesCollector) {
      this.objectFilesCollector = objectFilesCollector;
      return this;
    }

    /**
     * Sets {@link CcToolchainProvider} for the calling target.
     *
     * <p>This is needed if it can't correctly be inferred directly from the rule context. Setting
     * to null causes the default to be used as if this was never called.
     */
    public Builder setToolchainProvider(CcToolchainProvider toolchain) {
      this.toolchain = toolchain;
      return this;
    }

    /** Returns a {@link CompilationSupport} instance. */
    public CompilationSupport build() throws InterruptedException {
      checkNotNull(ruleContext, "CompilationSupport is missing RuleContext");

      if (buildConfiguration == null) {
        buildConfiguration = ruleContext.getConfiguration();
      }

      if (intermediateArtifacts == null) {
        intermediateArtifacts =
            ObjcRuleClasses.intermediateArtifacts(ruleContext, buildConfiguration);
      }

      if (compilationAttributes == null) {
        compilationAttributes = CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
      }

      if (outputGroupCollector == null) {
        outputGroupCollector = new TreeMap<>();
      }

      if (objectFilesCollector == null) {
        objectFilesCollector = ImmutableList.builder();
      }

      return new CompilationSupport(
          ruleContext,
          buildConfiguration,
          intermediateArtifacts,
          compilationAttributes,
          outputGroupCollector,
          objectFilesCollector,
          toolchain,
          isTestRule,
          usePch);
    }
  }

  /**
   * Returns a provider that collects this target's instrumented sources as well as those of its
   * dependencies.
   *
   * @param objectFiles the object files generated by this target
   * @return an instrumented files provider
   */
  public InstrumentedFilesInfo getInstrumentedFilesProvider(ImmutableList<Artifact> objectFiles) {
    return InstrumentedFilesCollector.collect(
        ruleContext,
        INSTRUMENTATION_SPEC,
        new ObjcCoverageMetadataCollector(),
        objectFiles,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        // The COVERAGE_GCOV_PATH environment variable is added in TestSupport#getExtraProviders()
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
        !isTestRule,
        /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));
  }

  /**
   * Validates compilation-related attributes on this rule.
   *
   * @return this compilation support
   * @throws RuleErrorException if there are attribute errors
   */
  CompilationSupport validateAttributes() throws RuleErrorException {
    for (PathFragment absoluteInclude :
        Iterables.filter(attributes.includes().toList(), PathFragment::isAbsolute)) {
      ruleContext.attributeError(
          "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
    }

    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      ImmutableSet<Artifact> hdrsSet = attributes.hdrs().toSet();
      ImmutableSet<Artifact> srcsSet =
          ImmutableSet.copyOf(
              ruleContext.getPrerequisiteArtifacts("srcs", TransitionMode.TARGET).list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      ImmutableSet<Artifact> nonArcSrcsSet =
          ImmutableSet.copyOf(
              ruleContext.getPrerequisiteArtifacts("non_arc_srcs", TransitionMode.TARGET).list());
      for (Artifact conflict : Sets.intersection(nonArcSrcsSet, srcsSet)) {
        String path = conflict.getRootRelativePath().toString();
        ruleContext.attributeError(
            "srcs", String.format(FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT, path));
      }
    }

    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("module_name")
        && ruleContext.attributes().isAttributeValueExplicitlySpecified("module_map")) {
      ruleContext.attributeError("module_name", BOTH_MODULE_NAME_AND_MODULE_MAP_SPECIFIED);
    }

    ruleContext.assertNoErrors();
    return this;
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcCompilationContext provides the compiling information to register these actions
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts, ObjcCompilationContext objcCompilationContext)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        compilationArtifacts,
        objcCompilationContext,
        Optional.absent(),
        ExtraCompileArgs.NONE,
        ImmutableList.<PathFragment>of());
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(ObjcCommon common)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        common, ExtraCompileArgs.NONE, ImmutableList.<PathFragment>of());
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, List<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(common, ExtraCompileArgs.NONE, priorityHeaders);
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcCompilationContext provides the compiling information to register these actions
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  private CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcCompilationContext objcCompilationContext,
      Optional<ObjcCommon> objcCommon,
      ExtraCompileArgs extraCompileArgs,
      List<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    checkNotNull(toolchain);
    checkNotNull(toolchain.getFdoContext());
    ObjcVariablesExtension.Builder extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setCompilationArtifacts(compilationArtifacts)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setConfiguration(buildConfiguration);

    CompilationResult compilationResult;

    if (compilationArtifacts.getArchive().isPresent()) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      extension.addVariableCategory(VariableCategory.ARCHIVE_VARIABLES);

      compilationResult =
          ccCompileAndLink(
              objcCompilationContext,
              compilationArtifacts,
              extension,
              extraCompileArgs,
              toolchain,
              toolchain.getFdoContext(),
              priorityHeaders,
              LinkTargetType.OBJC_ARCHIVE,
              objList);

      // TODO(b/30783125): Signal the need for this action in the CROSSTOOL.
      registerObjFilelistAction(
          ImmutableSet.copyOf(
              compilationResult.getCcCompilationOutputs().getObjectFiles(/* usePic= */ false)),
          objList);
    } else {
      compilationResult =
          ccCompileAndLink(
              objcCompilationContext,
              compilationArtifacts,
              extension,
              extraCompileArgs,
              toolchain,
              toolchain.getFdoContext(),
              priorityHeaders,
              /* linkType */ null,
              /* linkActionInput */ null);
    }

    objectFilesCollector.addAll(
        compilationResult.getCcCompilationOutputs().getObjectFiles(/* usePic= */ false));
    outputGroupCollector.putAll(compilationResult.getOutputGroups());

    if (objcCommon.isPresent()
        && objcCommon.get().getPurpose() == ObjcCommon.Purpose.COMPILE_AND_LINK) {

      ObjcProvider.NativeBuilder objcProviderBuilder = objcCommon.get().getObjcProviderBuilder();
      ObjcProvider objcProvider =
          objcProviderBuilder
              .setCcCompilationContext(compilationResult.getCcCompilationContext())
              .build();

      setObjcProvider(objcProvider);
    }

    return this;
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, ExtraCompileArgs extraCompileArgs, List<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    if (common.getCompilationArtifacts().isPresent()) {
      registerCompileAndArchiveActions(
          common.getCompilationArtifacts().get(),
          common.getObjcCompilationContext(),
          Optional.of(common),
          extraCompileArgs,
          priorityHeaders);
    }
    return this;
  }

  private StrippingType getStrippingType(ExtraLinkArgs extraLinkArgs) {
    if (Iterables.contains(extraLinkArgs, "-dynamiclib")) {
      return StrippingType.DYNAMIC_LIB;
    }
    if (Iterables.contains(extraLinkArgs, "-kext")) {
      return StrippingType.KERNEL_EXTENSION;
    }
    return StrippingType.DEFAULT;
  }

  /**
   * Registers any actions necessary to link this rule and its dependencies. Automatically infers
   * the toolchain from the configuration of this CompilationSupport.
   *
   * <p>Dsym bundle is generated if {@link ObjcConfiguration#generateDsym()} is set.
   *
   * <p>When Bazel flags {@code --compilation_mode=opt} and {@code --objc_enable_binary_stripping}
   * are specified, additional optimizations will be performed on the linked binary: all-symbol
   * stripping (using {@code /usr/bin/strip}) and dead-code stripping (using linker flags: {@code
   * -dead_strip} and {@code -no_dead_strip_inits_and_terms}).
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   * @param j2ObjcMappingFileProvider contains mapping files for j2objc transpilation
   * @param j2ObjcEntryClassProvider contains j2objc entry class information for dead code removal
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   * @return this compilation support
   */
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs)
      throws InterruptedException, RuleErrorException {
    Iterable<Artifact> prunedJ2ObjcArchives =
        computeAndStripPrunedJ2ObjcArchives(
            j2ObjcEntryClassProvider, j2ObjcMappingFileProvider, objcProvider);
    ImmutableList<Artifact> bazelBuiltLibraries =
        Iterables.isEmpty(prunedJ2ObjcArchives)
            ? objcProvider.getObjcLibraries()
            : substituteJ2ObjcPrunedLibraries(objcProvider);

    // Passing large numbers of inputs on the command line triggers a bug in Apple's Clang
    // (b/29094356), so we'll create an input list manually and pass -filelist path/to/input/list.
    // We can't populate this list yet--it needs to contain any linkstamp objects, which we won't
    // know about until we actually create the CppLinkAction--but it needs to go into the
    // CppLinkAction too, so create it now.
    Artifact inputFileList = intermediateArtifacts.linkerObjList();

    ImmutableSet<Artifact> forceLinkArtifacts = getForceLoadArtifacts(objcProvider);

    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    ImmutableSet<Artifact> objFiles =
        ImmutableSet.copyOf(
            Iterables.filter(
                Iterables.concat(
                    bazelBuiltLibraries,
                    objcProvider.get(IMPORTED_LIBRARY).toList(),
                    objcProvider.getCcLibraries()),
                Predicates.not(Predicates.in(forceLinkArtifacts))));

    LinkTargetType linkType =
        objcProvider.is(Flag.USES_CPP)
            ? LinkTargetType.OBJCPP_EXECUTABLE
            : LinkTargetType.OBJC_EXECUTABLE;

    ObjcVariablesExtension.Builder extensionBuilder =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setConfiguration(buildConfiguration)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setFrameworkNames(frameworkNames(objcProvider))
            .setFrameworkSearchPath(frameworkLibrarySearchPaths(objcProvider))
            .setLibraryNames(libraryNames(objcProvider))
            .setForceLoadArtifacts(getForceLoadArtifacts(objcProvider))
            .setAttributeLinkopts(attributes.linkopts())
            .addVariableCategory(VariableCategory.EXECUTABLE_LINKING_VARIABLES);

    Artifact binaryToLink = getBinaryToLink();
    CppLinkActionBuilder executableLinkActionBuilder =
        new CppLinkActionBuilder(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                binaryToLink,
                ruleContext.getConfiguration(),
                toolchain,
                toolchain.getFdoContext(),
                getFeatureConfiguration(ruleContext, toolchain, buildConfiguration),
                createObjcCppSemantics())
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
            .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget())
            .setMnemonic("ObjcLink")
            .addActionInputs(bazelBuiltLibraries)
            .addActionInputs(objcProvider.getCcLibraries())
            .addTransitiveActionInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveActionInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveActionInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .addTransitiveActionInputs(objcProvider.get(LINK_INPUTS))
            .setLinkerFiles(toolchain.getLinkerFiles())
            .addActionInputs(prunedJ2ObjcArchives)
            .addActionInputs(extraLinkInputs)
            .addActionInput(inputFileList)
            .setLinkType(linkType)
            .setLinkingMode(LinkingMode.STATIC)
            .addLinkopts(ImmutableList.copyOf(extraLinkArgs));

    if (objcConfiguration.generateDsym()) {
      Artifact dsymSymbol =
          objcConfiguration.shouldStripBinary()
              ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
              : intermediateArtifacts.dsymSymbolForStrippedBinary();
      extensionBuilder
          .setDsymSymbol(dsymSymbol)
          .addVariableCategory(VariableCategory.DSYM_VARIABLES);
      executableLinkActionBuilder.addActionOutput(dsymSymbol);
    }

    if (objcConfiguration.generateLinkmap()) {
      Artifact linkmap = intermediateArtifacts.linkmap();
      extensionBuilder.setLinkmap(linkmap).addVariableCategory(VariableCategory.LINKMAP_VARIABLES);
      executableLinkActionBuilder.addActionOutput(linkmap);
    }

    if (appleConfiguration.getBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
      Artifact bitcodeSymbolMap = intermediateArtifacts.bitcodeSymbolMap();
      extensionBuilder
          .setBitcodeSymbolMap(bitcodeSymbolMap)
          .addVariableCategory(VariableCategory.BITCODE_VARIABLES);
      executableLinkActionBuilder.addActionOutput(bitcodeSymbolMap);
    }

    executableLinkActionBuilder.addVariablesExtension(extensionBuilder.build());

    for (CcLinkingContext context :
        CppHelper.getLinkingContextsFromDeps(
            ImmutableList.copyOf(ruleContext.getPrerequisites("deps", TransitionMode.TARGET)))) {
      executableLinkActionBuilder.addLinkstamps(context.getLinkstamps().toList());
    }

    CppLinkAction executableLinkAction = executableLinkActionBuilder.build();

    // Populate the input file list with both the compiled object files and any linkstamp object
    // files.
    registerObjFilelistAction(
        ImmutableSet.<Artifact>builder()
            .addAll(objFiles)
            .addAll(executableLinkAction.getLinkstampObjectFileInputs())
            .build(),
        inputFileList);

    ruleContext.registerAction(executableLinkAction);

    if (objcConfiguration.shouldStripBinary()) {
      registerBinaryStripAction(binaryToLink, getStrippingType(extraLinkArgs));
    }

    return this;
  }

  /**
   * Returns the copts for the compile action in the current rule context (using a combination of
   * the rule's "copts" attribute as well as the current configuration copts).
   */
  private Iterable<String> getCompileRuleCopts() {
    List<String> copts =
        Stream.concat(objcConfiguration.getCopts().stream(), attributes.copts().stream())
            .collect(toCollection(ArrayList::new));

    for (String copt : copts) {
      if (copt.contains("-fmodules-cache-path")) {
        // Bazel decides on the cache path location.
        ruleContext.ruleWarning(MODULES_CACHE_PATH_WARNING);
      }
    }

    if (attributes.enableModules() && !getCustomModuleMap(ruleContext).isPresent()) {
      copts.add("-fmodules");
    }
    if (copts.contains("-fmodules")) {
      // If modules are enabled, clang caches module information. If unspecified, this is a
      // system-wide cache directory, which is a problem for remote executors which may run
      // multiple actions with different source trees that can't share this cache.
      // We thus set its path to the root of the genfiles directory.
      // Unfortunately, this cache contains non-hermetic information, thus we avoid declaring it as
      // an implicit output (as outputs must be hermetic).
      String cachePath =
          buildConfiguration.getGenfilesFragment() + "/" + OBJC_MODULE_CACHE_DIR_NAME;
      copts.add("-fmodules-cache-path=" + cachePath);
    }
    return copts;
  }

  /**
   * Registers an action that writes given set of object files to the given objList. This objList is
   * suitable to signal symbols to archive in a libtool archiving invocation.
   */
  // TODO(ulfjack): Use NestedSet for objFiles.
  private CompilationSupport registerObjFilelistAction(
      ImmutableSet<Artifact> objFiles, Artifact objList) {
    CustomCommandLine.Builder objFilesToLinkParam = new CustomCommandLine.Builder();
    NestedSetBuilder<Artifact> treeObjFiles = NestedSetBuilder.stableOrder();

    for (Artifact objFile : objFiles) {
      // If the obj file is a tree artifact, we need to expand it into the contained individual
      // files properly.
      if (objFile.isTreeArtifact()) {
        treeObjFiles.add(objFile);
        objFilesToLinkParam.addExpandedTreeArtifactExecPaths(objFile);
      } else {
        objFilesToLinkParam.addPath(objFile.getExecPath());
      }
    }

    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            treeObjFiles.build(),
            objList,
            objFilesToLinkParam.build(),
            ParameterFile.ParameterFileType.UNQUOTED));
    return this;
  }

  /**
   * Registers an action to create an archive artifact by fully (statically) linking all transitive
   * dependencies of this rule.
   *
   * @param objcProvider provides all compiling and linking information to create this artifact
   * @param outputArchive the output artifact for this action
   * @return this {@link CompilationSupport} instance
   */
  CompilationSupport registerFullyLinkAction(ObjcProvider objcProvider, Artifact outputArchive)
      throws InterruptedException, RuleErrorException {
    checkNotNull(toolchain);
    checkNotNull(toolchain.getFdoContext());
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
                ruleContext,
                ruleContext.getLabel(),
                outputArchive,
                ruleContext.getConfiguration(),
                toolchain,
                toolchain.getFdoContext(),
                getFeatureConfiguration(ruleContext, toolchain, buildConfiguration),
                createObjcCppSemantics())
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
            .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget())
            .addActionInputs(objcProvider.getObjcLibraries())
            .addActionInputs(objcProvider.getCcLibraries())
            .addActionInputs(objcProvider.get(IMPORTED_LIBRARY).toSet())
            .setLinkerFiles(toolchain.getLinkerFiles())
            .setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
            .setLinkingMode(LinkingMode.STATIC)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtension(extension)
            .build();
    ruleContext.registerAction(fullyLinkAction);

    return this;
  }

  /**
   * Returns all framework names to pass to the linker using {@code -framework} flags. For a
   * framework in the directory foo/bar.framework, the name is "bar". Each framework is found
   * without using the full path by means of the framework search paths. Search paths are added by
   * {@link#commonLinkAndCompileFlagsForClang(ObjcProvider, ObjcConfiguration, AppleConfiguration)})
   *
   * <p>It's awful that we can't pass the full path to the framework and avoid framework search
   * paths, but this is imposed on us by clang. clang does not support passing the full path to the
   * framework, so Bazel cannot do it either.
   */
  private Set<String> frameworkNames(ObjcProvider provider) {
    Set<String> names = new LinkedHashSet<>();
    names.addAll(SdkFramework.names(provider.get(SDK_FRAMEWORK)));
    names.addAll(provider.staticFrameworkNames().toList());
    names.addAll(provider.dynamicFrameworkNames().toList());
    return names;
  }

  /** Returns libraries that should be passed to the linker. */
  private ImmutableList<String> libraryNames(ObjcProvider objcProvider) {
    ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
    for (String dylib : objcProvider.get(SDK_DYLIB).toList()) {
      if (dylib.startsWith("lib")) {
        // remove lib prefix if it exists which is standard
        // for libraries (libxml.dylib -> -lxml).
        dylib = dylib.substring(3);
      }
      args.add(dylib);
    }
    return args.build();
  }

  /** Returns libraries that should be passed into the linker with {@code -force_load}. */
  private ImmutableSet<Artifact> getForceLoadArtifacts(ObjcProvider objcProvider) {
    List<Artifact> ccLibraries = objcProvider.getCcLibraries();
    Iterable<Artifact> ccLibrariesToForceLoad =
        Iterables.filter(ccLibraries, ALWAYS_LINKED_CC_LIBRARY);

    return ImmutableSet.<Artifact>builder()
        .addAll(objcProvider.get(FORCE_LOAD_LIBRARY).toList())
        .addAll(ccLibrariesToForceLoad)
        .build();
  }

  /** Returns pruned J2Objc archives for this target. */
  private ImmutableList<Artifact> j2objcPrunedLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> j2objcPrunedLibraryBuilder = ImmutableList.builder();
    for (Artifact j2objcLibrary : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toList()) {
      j2objcPrunedLibraryBuilder.add(intermediateArtifacts.j2objcPrunedArchive(j2objcLibrary));
    }
    return j2objcPrunedLibraryBuilder.build();
  }

  /** Returns true if this build should strip J2Objc dead code. */
  private boolean stripJ2ObjcDeadCode(J2ObjcEntryClassProvider j2ObjcEntryClassProvider) {
    J2ObjcConfiguration j2objcConfiguration =
        buildConfiguration.getFragment(J2ObjcConfiguration.class);
    // Only perform J2ObjC dead code stripping if flag --j2objc_dead_code_removal is specified and
    // users have specified entry classes.
    return j2objcConfiguration.removeDeadCode()
        && !j2ObjcEntryClassProvider.getEntryClasses().isEmpty();
  }

  /** Registers actions to perform J2Objc dead code removal. */
  private void registerJ2ObjcDeadCodeRemovalActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider) {
    NestedSet<String> entryClasses = j2ObjcEntryClassProvider.getEntryClasses();
    Artifact pruner =
        ruleContext.getPrerequisiteArtifact("$j2objc_dead_code_pruner", TransitionMode.HOST);
    NestedSet<Artifact> j2ObjcDependencyMappingFiles =
        j2ObjcMappingFileProvider.getDependencyMappingFiles();
    NestedSet<Artifact> j2ObjcHeaderMappingFiles =
        j2ObjcMappingFileProvider.getHeaderMappingFiles();
    NestedSet<Artifact> j2ObjcArchiveSourceMappingFiles =
        j2ObjcMappingFileProvider.getArchiveSourceMappingFiles();

    for (Artifact j2objcArchive : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toList()) {
      Artifact prunedJ2ObjcArchive = intermediateArtifacts.j2objcPrunedArchive(j2objcArchive);
      Artifact dummyArchive =
          ruleContext
              .getPrerequisite(
                  "$dummy_lib", TransitionMode.TARGET, ObjcProvider.STARLARK_CONSTRUCTOR)
              .get(LIBRARY)
              .getSingleton();

      CustomCommandLine commandLine =
          CustomCommandLine.builder()
              .addExecPath("--input_archive", j2objcArchive)
              .addExecPath("--output_archive", prunedJ2ObjcArchive)
              .addExecPath("--dummy_archive", dummyArchive)
              .addExecPath("--xcrunwrapper", xcrunwrapper(ruleContext).getExecutable())
              .addExecPaths(
                  "--dependency_mapping_files",
                  VectorArg.join(",").each(j2ObjcDependencyMappingFiles))
              .addExecPaths(
                  "--header_mapping_files", VectorArg.join(",").each(j2ObjcHeaderMappingFiles))
              .addExecPaths(
                  "--archive_source_mapping_files",
                  VectorArg.join(",").each(j2ObjcArchiveSourceMappingFiles))
              .add("--entry_classes")
              .addAll(VectorArg.join(",").each(entryClasses))
              .build();

      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(
                  XcodeConfigInfo.fromRuleContext(ruleContext),
                  appleConfiguration.getSingleArchPlatform())
              .setMnemonic("DummyPruner")
              .setExecutable(pruner)
              .addInput(dummyArchive)
              .addInput(pruner)
              .addInput(j2objcArchive)
              .addInput(xcrunwrapper(ruleContext).getExecutable())
              .addTransitiveInputs(j2ObjcDependencyMappingFiles)
              .addTransitiveInputs(j2ObjcHeaderMappingFiles)
              .addTransitiveInputs(j2ObjcArchiveSourceMappingFiles)
              .addCommandLine(
                  commandLine,
                  ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
                      .setCharset(ISO_8859_1)
                      .setUseAlways(true)
                      .build())
              .addOutput(prunedJ2ObjcArchive)
              .build(ruleContext));
    }
  }

  /** Returns archives arising from j2objc transpilation after dead code removal. */
  private Iterable<Artifact> computeAndStripPrunedJ2ObjcArchives(
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      ObjcProvider objcProvider) {
    Iterable<Artifact> prunedJ2ObjcArchives = ImmutableList.<Artifact>of();
    if (stripJ2ObjcDeadCode(j2ObjcEntryClassProvider)) {
      registerJ2ObjcDeadCodeRemovalActions(
          objcProvider, j2ObjcMappingFileProvider, j2ObjcEntryClassProvider);
      prunedJ2ObjcArchives = j2objcPrunedLibraries(objcProvider);
    }
    return prunedJ2ObjcArchives;
  }

  /**
   * Returns a nested set of Bazel-built ObjC libraries with all unpruned J2ObjC libraries
   * substituted with pruned ones.
   */
  private ImmutableList<Artifact> substituteJ2ObjcPrunedLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> libraries = new ImmutableList.Builder<>();

    Set<Artifact> unprunedJ2ObjcLibs = objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toSet();
    for (Artifact library : objcProvider.getObjcLibraries()) {
      // If we match an unpruned J2ObjC library, add the pruned version of the J2ObjC static library
      // instead.
      if (unprunedJ2ObjcLibs.contains(library)) {
        libraries.add(intermediateArtifacts.j2objcPrunedArchive(library));
      } else {
        libraries.add(library);
      }
    }
    return libraries.build();
  }

  /** Returns the artifact that should be the outcome of this build's link action */
  private Artifact getBinaryToLink() {

    // When compilation_mode=opt and objc_enable_binary_stripping are specified, the unstripped
    // binary containing debug symbols is generated by the linker, which also needs the debug
    // symbols for dead-code removal. The binary is also used to generate dSYM bundle if
    // --apple_generate_dsym is specified. A symbol strip action is later registered to strip
    // the symbol table from the unstripped binary.
    return objcConfiguration.shouldStripBinary()
        ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
        : intermediateArtifacts.strippedSingleArchitectureBinary();
  }

  private static CommandLine symbolStripCommandLine(
      ImmutableList<String> extraFlags, Artifact unstrippedArtifact, Artifact strippedArtifact) {
    return CustomCommandLine.builder()
        .add(STRIP)
        .addAll(extraFlags)
        .addExecPath("-o", strippedArtifact)
        .addPath(unstrippedArtifact.getExecPath())
        .build();
  }

  /** Signals if stripping should include options for dynamic libraries. */
  private enum StrippingType {
    DEFAULT,
    DYNAMIC_LIB,
    KERNEL_EXTENSION
  }

  /**
   * Registers an action that uses the 'strip' tool to perform binary stripping on the given binary
   * subject to the given {@link StrippingType}.
   */
  private void registerBinaryStripAction(Artifact binaryToLink, StrippingType strippingType) {
    final ImmutableList<String> stripArgs;
    if (isTestRule) {
      // For test targets, only debug symbols are stripped off, since /usr/bin/strip is not able
      // to strip off all symbols in XCTest bundle.
      stripArgs = ImmutableList.of("-S");
    } else {
      switch (strippingType) {
        case DYNAMIC_LIB:
        case KERNEL_EXTENSION:
          // For dylibs and kexts, must strip only local symbols.
          stripArgs = ImmutableList.of("-x");
          break;
        case DEFAULT:
          stripArgs = ImmutableList.<String>of();
          break;
        default:
          throw new IllegalArgumentException("Unsupported stripping type " + strippingType);
      }
    }

    Artifact strippedBinary = intermediateArtifacts.strippedSingleArchitectureBinary();

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                XcodeConfigInfo.fromRuleContext(ruleContext),
                appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcBinarySymbolStrip")
            .setExecutable(xcrunwrapper(ruleContext))
            .addCommandLine(symbolStripCommandLine(stripArgs, binaryToLink, strippedBinary))
            .addOutput(strippedBinary)
            .addInput(binaryToLink)
            .build(ruleContext));
  }

  private CompilationSupport registerGenerateUmbrellaHeaderAction(
      Artifact umbrellaHeader, NestedSet<Artifact> publicHeaders) {
    ruleContext.registerAction(
        new UmbrellaHeaderAction(
            ruleContext.getActionOwner(),
            umbrellaHeader,
            publicHeaders,
            ImmutableList.<PathFragment>of()));

    return this;
  }

  private Optional<Artifact> getPchFile() {
    if (!usePch) {
      return Optional.absent();
    }
    Artifact pchHdr = null;
    if (ruleContext.attributes().has("pch", BuildType.LABEL)) {
      pchHdr = ruleContext.getPrerequisiteArtifact("pch", TransitionMode.TARGET);
    }
    return Optional.fromNullable(pchHdr);
  }

  /**
   * Registers an action that will generate a clang module map for this target, using the hdrs
   * attribute of this rule.
   */
  CompilationSupport registerGenerateModuleMapAction(CompilationArtifacts compilationArtifacts) {
    // TODO(bazel-team): Include textual headers in the module map when Xcode 6 support is
    // dropped.
    // TODO(b/32225593): Include private headers in the module map.
    // Both registerGenerateModuleMapAction and registerGenerateUmbrellaHeaderAction make a copy,
    // so flattening eagerly here using toList() is acceptable.
    NestedSet<Artifact> publicHeaders =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(attributes.hdrs())
            .addTransitive(compilationArtifacts.getAdditionalHdrs())
            .build();
    CppModuleMap moduleMap = intermediateArtifacts.moduleMap();
    registerGenerateModuleMapAction(moduleMap, publicHeaders);

    Optional<Artifact> umbrellaHeader = moduleMap.getUmbrellaHeader();
    if (umbrellaHeader.isPresent()) {
      registerGenerateUmbrellaHeaderAction(umbrellaHeader.get(), publicHeaders);
    }

    return this;
  }

  /**
   * Registers an action that will generate a clang module map.
   *
   * @param moduleMap the module map to generate
   * @param publicHeaders the headers that should be directly accessible by dependers
   * @return this compilation support
   */
  public CompilationSupport registerGenerateModuleMapAction(
      CppModuleMap moduleMap, NestedSet<Artifact> publicHeaders) {
    return registerGenerateModuleMapAction(moduleMap, publicHeaders.toList());
  }

  /**
   * Registers an action that will generate a clang module map.
   *
   * @param moduleMap the module map to generate
   * @param publicHeaders the headers that should be directly accessible by dependers
   * @return this compilation support
   */
  public CompilationSupport registerGenerateModuleMapAction(
      CppModuleMap moduleMap, Iterable<Artifact> publicHeaders) {
    publicHeaders = Iterables.filter(publicHeaders, CppFileTypes.MODULE_MAP_HEADER);
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            ImmutableList.<Artifact>of(),
            publicHeaders,
            attributes.moduleMapsForDirectDeps().toList(),
            ImmutableList.<PathFragment>of(),
            /*compiledModule=*/ true,
            /*moduleMapHomeIsCwd=*/ false,
            /* generateSubmodules= */ false,
            /*externDependencies=*/ true));

    return this;
  }

  /**
   * Collector that, given a list of output artifacts, finds and registers coverage notes metadata
   * for any compilation action.
   */
  private static class ObjcCoverageMetadataCollector extends LocalMetadataCollector {

    @Override
    public void collectMetadataArtifacts(
        Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment,
        NestedSetBuilder<Artifact> metadataFilesBuilder) {
      for (Artifact artifact : artifacts) {
        ActionAnalysisMetadata action = analysisEnvironment.getLocalGeneratingAction(artifact);
        if (action.getMnemonic().equals("ObjcCompile")) {
          addOutputs(metadataFilesBuilder, action, ObjcRuleClasses.COVERAGE_NOTES);
        }
      }
    }
  }

  public static Optional<Artifact> getCustomModuleMap(RuleContext ruleContext) {
    if (ruleContext.attributes().has("module_map", BuildType.LABEL)) {
      return Optional.fromNullable(
          ruleContext.getPrerequisiteArtifact("module_map", TransitionMode.TARGET));
    }
    return Optional.absent();
  }
}
