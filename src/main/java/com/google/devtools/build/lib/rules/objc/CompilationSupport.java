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

import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.cpp.Link.LINK_LIBRARY_FILETYPES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_SEARCH_PATHS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
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
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CollidingProvidesException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
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
  static final ImmutableList<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS =
      ImmutableList.of(new SdkFramework("Foundation"), new SdkFramework("UIKit"));

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

  private static final String XCODE_VERSION_FEATURE_NAME_PREFIX = "xcode_";

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

  /** The kind of include processing to use. */
  enum IncludeProcessingType {
    HEADER_THINNING,
    INCLUDE_SCANNING,
    NO_PROCESSING;
  }

  /** Returns the location of the xcrunwrapper tool. */
  public static final FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$xcrunwrapper", Mode.HOST);
  }

  /** Returns the location of the libtool tool. */
  public static final FilesToRunProvider libtool(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite(ObjcRuleClasses.LIBTOOL_ATTRIBUTE, Mode.HOST);
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

  /**
   * Returns additional inputs to include processing, outside of the headers provided by
   * ObjProvider.
   */
  private Iterable<Artifact> getExtraIncludeProcessingInputs(
      Collection<Artifact> privateHdrs, Artifact pchHdr) {
    Iterable<Artifact> extraInputs = privateHdrs;
    if (pchHdr != null) {
      extraInputs = Iterables.concat(extraInputs, ImmutableList.of(pchHdr));
    }
    return extraInputs;
  }

  /**
   * Create and return the include processing to be used. Only HeaderThinning uses potentialInputs.
   */
  private IncludeProcessing createIncludeProcessing(Iterable<Artifact> potentialInputs) {
    switch (includeProcessingType) {
      case HEADER_THINNING:
        return new HeaderThinning(potentialInputs);
      case INCLUDE_SCANNING:
        return IncludeScanning.INSTANCE;
      default:
        return NoProcessing.INSTANCE;
    }
  }

  private CompilationInfo compile(
      ObjcProvider objcProvider,
      VariablesExtension extension,
      ExtraCompileArgs extraCompileArgs,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      Iterable<PathFragment> priorityHeaders,
      Collection<Artifact> sources,
      Collection<Artifact> privateHdrs,
      Collection<Artifact> publicHdrs,
      Collection<Artifact> dependentGeneratedHdrs,
      Artifact pchHdr,
      // TODO(b/70777494): Find out how deps get used and remove if not needed.
      Iterable<? extends TransitiveInfoCollection> deps,
      ObjcCppSemantics semantics,
      String purpose,
      boolean generateModuleMap)
      throws RuleErrorException, InterruptedException {
    CcCompilationHelper result =
        new CcCompilationHelper(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                CppHelper.getGrepIncludes(ruleContext),
                semantics,
                getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration, objcProvider),
                CcCompilationHelper.SourceCategory.CC_AND_OBJC,
                ccToolchain,
                fdoContext,
                buildConfiguration,
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .addSources(sources)
            .addPrivateHeaders(privateHdrs)
            .addDefines(objcProvider.get(DEFINE))
            .addPublicHeaders(publicHdrs)
            .addPrivateHeadersUnchecked(dependentGeneratedHdrs)
            .addCcCompilationContexts(
                Streams.stream(AnalysisUtils.getProviders(deps, CcInfo.PROVIDER))
                    .map(CcInfo::getCcCompilationContext)
                    .collect(ImmutableList.toImmutableList()))
            .setCopts(
                ImmutableList.<String>builder()
                    .addAll(getCompileRuleCopts())
                    .addAll(
                        ruleContext
                            .getFragment(ObjcConfiguration.class)
                            .getCoptsForCompilationMode())
                    .addAll(extraCompileArgs)
                    .build())
            .addFrameworkIncludeDirs(frameworkHeaderSearchPathFragments(objcProvider))
            .addIncludeDirs(priorityHeaders)
            .addIncludeDirs(objcProvider.get(INCLUDE))
            .addSystemIncludeDirs(objcProvider.get(INCLUDE_SYSTEM))
            .setCppModuleMap(intermediateArtifacts.moduleMap())
            .setPropagateModuleMapToCompileAction(false)
            .addVariableExtension(extension)
            .setPurpose(purpose)
            .addQuoteIncludeDirs(
                ObjcCommon.userHeaderSearchPaths(objcProvider, ruleContext.getConfiguration()))
            .setCodeCoverageEnabled(CcCompilationHelper.isCodeCoverageEnabled(ruleContext))
            .setHeadersCheckingMode(semantics.determineHeadersCheckingMode(ruleContext));

    if (pchHdr != null) {
      result.addAdditionalInputs(ImmutableList.of(pchHdr));
    }

    if (getCustomModuleMap(ruleContext).isPresent() || !generateModuleMap) {
      result.doNotGenerateModuleMap();
    }

    return result.compile();
  }

  private Pair<CcCompilationOutputs, ImmutableMap<String, NestedSet<Artifact>>> ccCompileAndLink(
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      ObjcVariablesExtension.Builder extensionBuilder,
      ExtraCompileArgs extraCompileArgs,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      Iterable<PathFragment> priorityHeaders,
      LinkTargetType linkType,
      Artifact linkActionInput)
      throws RuleErrorException, InterruptedException {
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
    // This is a hack to inject generated headers into the action graph for include scanning.  This
    // is supposed to be done via the compilation prerequisite middleman artifact of dependent
    // CcCompilationContexts, but ObjcProvider does not propagate that.  This issue will go away
    // when we finish migrating the compile info in ObjcProvider to CcCompilationContext.
    //
    // To limit the extra work we're adding, we only add what is required, i.e. the
    // generated headers.
    Collection<Artifact> dependentGeneratedHdrs =
        (includeProcessingType == IncludeProcessingType.INCLUDE_SCANNING)
            ? objcProvider.getGeneratedHeaderList()
            : ImmutableList.of();
    Artifact pchHdr = getPchFile().orNull();
    Iterable<? extends TransitiveInfoCollection> deps =
        ruleContext.getPrerequisites("deps", Mode.TARGET);
    ObjcCppSemantics semantics = createObjcCppSemantics(objcProvider, privateHdrs, pchHdr);

    String purpose = String.format("%s_objc_arc", semantics.getPurpose());
    extensionBuilder.setArcEnabled(true);
    CompilationInfo objcArcCompilationInfo =
        compile(
            objcProvider,
            extensionBuilder.build(),
            extraCompileArgs,
            ccToolchain,
            fdoContext,
            priorityHeaders,
            arcSources,
            privateHdrs,
            publicHdrs,
            dependentGeneratedHdrs,
            pchHdr,
            deps,
            semantics,
            purpose,
            /* generateModuleMap= */ true);

    purpose = String.format("%s_non_objc_arc", semantics.getPurpose());
    extensionBuilder.setArcEnabled(false);
    CompilationInfo nonObjcArcCompilationInfo =
        compile(
            objcProvider,
            extensionBuilder.build(),
            extraCompileArgs,
            ccToolchain,
            fdoContext,
            priorityHeaders,
            nonArcSources,
            privateHdrs,
            publicHdrs,
            dependentGeneratedHdrs,
            pchHdr,
            deps,
            semantics,
            purpose,
            // Only generate the module map once (see above) and re-use it here.
            /* generateModuleMap= */ false);

    FeatureConfiguration featureConfiguration =
        getFeatureConfiguration(ruleContext, ccToolchain, buildConfiguration, objcProvider);
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
                    ImmutableList.copyOf(ruleContext.getPrerequisites("deps", Mode.TARGET))))
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
    ccCompilationContextBuilder.mergeDependentCcCompilationContexts(
        Arrays.asList(
            objcArcCompilationInfo.getCcCompilationContext(),
            nonObjcArcCompilationInfo.getCcCompilationContext()));
    ccCompilationContextBuilder.setPurpose(
        String.format("%s_merged_arc_non_arc_objc", semantics.getPurpose()));
    ccCompilationContextBuilder.addQuoteIncludeDirs(
        ObjcCommon.userHeaderSearchPaths(objcProvider, ruleContext.getConfiguration()));

    CcCompilationOutputs precompiledFilesObjects =
        CcCompilationOutputs.builder()
            .addObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ false))
            .addPicObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ true))
            .build();

    CcCompilationOutputs.Builder compilationOutputsBuilder =
        CcCompilationOutputs.builder()
            .merge(objcArcCompilationInfo.getCcCompilationOutputs())
            .merge(nonObjcArcCompilationInfo.getCcCompilationOutputs())
            .merge(precompiledFilesObjects);
    compilationOutputsBuilder.merge(objcArcCompilationInfo.getCcCompilationOutputs());
    compilationOutputsBuilder.merge(nonObjcArcCompilationInfo.getCcCompilationOutputs());
    CcCompilationOutputs compilationOutputs = compilationOutputsBuilder.build();

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

    return new Pair<>(compilationOutputsBuilder.build(), ImmutableMap.copyOf(mergedOutputGroups));
  }

  private ObjcCppSemantics createObjcCppSemantics(
      ObjcProvider objcProvider, Collection<Artifact> privateHdrs, Artifact pchHdr) {
    Iterable<Artifact> extraInputs = getExtraIncludeProcessingInputs(privateHdrs, pchHdr);
    return new ObjcCppSemantics(
        objcProvider,
        includeProcessingType,
        createIncludeProcessing(Iterables.concat(extraInputs, objcProvider.get(HEADER))),
        extraInputs,
        ruleContext.getFragment(ObjcConfiguration.class),
        intermediateArtifacts,
        buildConfiguration,
        attributes.enableModules());
  }

  private FeatureConfiguration getFeatureConfiguration(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration configuration,
      ObjcProvider objcProvider) {
    boolean isHost = ruleContext.getConfiguration().isHostConfiguration();
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
            .add(isHost ? "host" : "nonhost")
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
      return ccToolchain
          .getFeatures()
          .getFeatureConfiguration(activatedCrosstoolSelectables.build());
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

  /**
   * Set of {@link com.google.devtools.build.lib.util.FileType} of source artifacts that are
   * compatible with header thinning.
   */
  private static final FileTypeSet SOURCES_FOR_HEADER_THINNING =
      FileTypeSet.of(
          CppFileTypes.OBJC_SOURCE,
          CppFileTypes.OBJCPP_SOURCE,
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.C_SOURCE);

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
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(
            ruleContext
                .getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET)
                .errorsForNonMatching(NON_ARC_SRCS_TYPE)
                .list())
        .addPrivateHdrs(srcs.filter(HEADERS).list())
        .addPrecompiledSrcs(srcs.filter(PRECOMPILED_SRCS_TYPE).list())
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  /** Returns a list of framework header search path fragments. */
  static ImmutableList<PathFragment> frameworkHeaderSearchPathFragments(ObjcProvider provider)
      throws InterruptedException {
    ImmutableList.Builder<PathFragment> searchPaths = new ImmutableList.Builder<>();
    return searchPaths
        .addAll(uniqueParentDirectories(provider.get(FRAMEWORK_SEARCH_PATHS)))
        .build();
  }

  /** Returns a list of framework header search paths. */
  static ImmutableList<String> frameworkHeaderSearchPaths(ObjcProvider provider)
      throws InterruptedException {
    ImmutableList.Builder<String> searchPaths = new ImmutableList.Builder<>();
    return searchPaths
        .addAll(
            Iterables.transform(
                frameworkHeaderSearchPathFragments(provider), PathFragment::getSafePathString))
        .build();
  }

  /** Returns a list of framework library search paths. */
  static ImmutableList<String> frameworkLibrarySearchPaths(ObjcProvider provider)
      throws InterruptedException {
    ImmutableList.Builder<String> searchPaths = new ImmutableList.Builder<>();
    return searchPaths
        // Add library search paths corresponding to custom (non-SDK) frameworks. For each framework
        // foo/bar.framework, include "foo" as a search path.
        .addAll(provider.staticFrameworkPaths())
        .addAll(provider.dynamicFrameworkPaths())
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
    } else if (isHeaderThinningEnabled()) {
      includeProcessingType = IncludeProcessingType.HEADER_THINNING;
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
      Preconditions.checkNotNull(ruleContext, "CompilationSupport is missing RuleContext");

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
        Iterables.filter(attributes.includes(), PathFragment::isAbsolute)) {
      ruleContext.attributeError(
          "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
    }

    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      ImmutableSet<Artifact> hdrsSet = ImmutableSet.copyOf(attributes.hdrs());
      ImmutableSet<Artifact> srcsSet =
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      ImmutableSet<Artifact> nonArcSrcsSet =
          ImmutableSet.copyOf(
              ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET).list());
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
   * @param objcProvider provides all compiling and linking information to register these actions
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts, ObjcProvider objcProvider)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        compilationArtifacts,
        objcProvider,
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
      ObjcCommon common, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(common, ExtraCompileArgs.NONE, priorityHeaders);
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcProvider provides all compiling and linking information to register these actions
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  private CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider,
      ExtraCompileArgs extraCompileArgs,
      Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(toolchain);
    Preconditions.checkNotNull(toolchain.getFdoContext());
    ObjcVariablesExtension.Builder extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setCompilationArtifacts(compilationArtifacts)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setConfiguration(buildConfiguration)
            .setFrameworkSearchPath(frameworkHeaderSearchPaths(objcProvider));

    Pair<CcCompilationOutputs, ImmutableMap<String, NestedSet<Artifact>>> compilationInfo;

    if (compilationArtifacts.getArchive().isPresent()) {
      Artifact objList = intermediateArtifacts.archiveObjList();

      extension.addVariableCategory(VariableCategory.ARCHIVE_VARIABLES);

      compilationInfo =
          ccCompileAndLink(
              objcProvider,
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
          compilationInfo.getFirst().getObjectFiles(/* usePic= */ false), objList);
    } else {
      compilationInfo =
          ccCompileAndLink(
              objcProvider,
              compilationArtifacts,
              extension,
              extraCompileArgs,
              toolchain,
              toolchain.getFdoContext(),
              priorityHeaders,
              /* linkType */ null,
              /* linkActionInput */ null);
    }

    objectFilesCollector.addAll(compilationInfo.getFirst().getObjectFiles(/* usePic= */ false));
    outputGroupCollector.putAll(compilationInfo.getSecond());

    registerHeaderScanningActions(compilationInfo.getFirst(), objcProvider, compilationArtifacts);

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
      ObjcCommon common, ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    if (common.getCompilationArtifacts().isPresent()) {
      registerCompileAndArchiveActions(
          common.getCompilationArtifacts().get(),
          common.getObjcProvider(),
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

    Artifact inputFileList = intermediateArtifacts.linkerObjList();
    ImmutableSet<Artifact> forceLinkArtifacts = getForceLoadArtifacts(objcProvider);

    Iterable<Artifact> objFiles =
        Iterables.concat(
            bazelBuiltLibraries, objcProvider.get(IMPORTED_LIBRARY), objcProvider.getCcLibraries());
    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    objFiles = Iterables.filter(objFiles, Predicates.not(Predicates.in(forceLinkArtifacts)));

    registerObjFilelistAction(objFiles, inputFileList);

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
    CppLinkActionBuilder executableLinkAction =
        new CppLinkActionBuilder(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                binaryToLink,
                ruleContext.getConfiguration(),
                toolchain,
                toolchain.getFdoContext(),
                getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, objcProvider),
                createObjcCppSemantics(
                    objcProvider, /* privateHdrs= */ ImmutableList.of(), /* pchHdr= */ null))
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
      executableLinkAction.addActionOutput(dsymSymbol);
    }

    if (objcConfiguration.generateLinkmap()) {
      Artifact linkmap = intermediateArtifacts.linkmap();
      extensionBuilder.setLinkmap(linkmap).addVariableCategory(VariableCategory.LINKMAP_VARIABLES);
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
  private CompilationSupport registerObjFilelistAction(
      Iterable<Artifact> objFiles, Artifact objList) {
    ImmutableSet<Artifact> dedupedObjFiles = ImmutableSet.copyOf(objFiles);
    CustomCommandLine.Builder objFilesToLinkParam = new CustomCommandLine.Builder();
    ImmutableList.Builder<Artifact> treeObjFiles = new ImmutableList.Builder<>();

    for (Artifact objFile : dedupedObjFiles) {
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
            ParameterFile.ParameterFileType.UNQUOTED,
            ISO_8859_1));
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
    Preconditions.checkNotNull(toolchain);
    Preconditions.checkNotNull(toolchain.getFdoContext());
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
            .setFrameworkSearchPath(frameworkHeaderSearchPaths(objcProvider))
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
                getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, objcProvider),
                createObjcCppSemantics(
                    objcProvider, /* privateHdrs= */ ImmutableList.of(), /* pchHdr= */ null))
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
    Iterables.addAll(names, SdkFramework.names(provider.get(SDK_FRAMEWORK)));
    Iterables.addAll(names, provider.staticFrameworkNames());
    Iterables.addAll(names, provider.dynamicFrameworkNames());
    return names;
  }

  /** Returns libraries that should be passed to the linker. */
  private ImmutableList<String> libraryNames(ObjcProvider objcProvider) {
    ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
    for (String dylib : objcProvider.get(SDK_DYLIB)) {
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
        .addAll(objcProvider.get(FORCE_LOAD_LIBRARY))
        .addAll(ccLibrariesToForceLoad)
        .build();
  }

  /** Returns pruned J2Objc archives for this target. */
  private ImmutableList<Artifact> j2objcPrunedLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> j2objcPrunedLibraryBuilder = ImmutableList.builder();
    for (Artifact j2objcLibrary : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY)) {
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
    Artifact pruner = ruleContext.getPrerequisiteArtifact("$j2objc_dead_code_pruner", Mode.HOST);
    NestedSet<Artifact> j2ObjcDependencyMappingFiles =
        j2ObjcMappingFileProvider.getDependencyMappingFiles();
    NestedSet<Artifact> j2ObjcHeaderMappingFiles =
        j2ObjcMappingFileProvider.getHeaderMappingFiles();
    NestedSet<Artifact> j2ObjcArchiveSourceMappingFiles =
        j2ObjcMappingFileProvider.getArchiveSourceMappingFiles();

    for (Artifact j2objcArchive : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY)) {
      Artifact prunedJ2ObjcArchive = intermediateArtifacts.j2objcPrunedArchive(j2objcArchive);
      Artifact dummyArchive =
          Iterables.getOnlyElement(
              ruleContext
                  .getPrerequisite("$dummy_lib", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR)
                  .get(LIBRARY));

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
      Artifact umbrellaHeader, Iterable<Artifact> publicHeaders) {
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
      pchHdr = ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET);
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
    Iterable<Artifact> publicHeaders = attributes.hdrs();
    publicHeaders = Iterables.concat(publicHeaders, compilationArtifacts.getAdditionalHdrs());
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
      CppModuleMap moduleMap, Iterable<Artifact> publicHeaders) {
    publicHeaders = Iterables.filter(publicHeaders, CppFileTypes.MODULE_MAP_HEADER);
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            ImmutableList.<Artifact>of(),
            publicHeaders,
            attributes.moduleMapsForDirectDeps(),
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

  private static Iterable<PathFragment> uniqueParentDirectories(Iterable<PathFragment> paths) {
    ImmutableSet.Builder<PathFragment> parents = new ImmutableSet.Builder<>();
    for (PathFragment path : paths) {
      parents.add(path.getParentDirectory());
    }
    return parents.build();
  }

  /** Holds information about Objective-C compile actions that require header thinning. */
  private static final class ObjcHeaderThinningInfo {
    /** Source file for compile action. */
    public final Artifact sourceFile;
    /** headers_list file for compile action. */
    public final Artifact headersListFile;
    /** Command line arguments for compile action execution. */
    public final ImmutableList<String> arguments;

    public ObjcHeaderThinningInfo(
        Artifact sourceFile, Artifact headersListFile, ImmutableList<String> arguments) {
      this.sourceFile = Preconditions.checkNotNull(sourceFile);
      this.headersListFile = Preconditions.checkNotNull(headersListFile);
      this.arguments = Preconditions.checkNotNull(arguments);
    }

    public ObjcHeaderThinningInfo(
        Artifact sourceFile, Artifact headersListFile, Iterable<String> arguments) {
      this(sourceFile, headersListFile, ImmutableList.copyOf(arguments));
    }
  }

  /**
   * Returns true when ObjC header thinning is enabled via configuration and an a valid
   * header_scanner executable target is provided.
   */
  private boolean isHeaderThinningEnabled() {
    if (objcConfiguration.useExperimentalHeaderThinning()
        && ruleContext.isAttrDefined(ObjcRuleClasses.HEADER_SCANNER_ATTRIBUTE, BuildType.LABEL)) {
      FilesToRunProvider tool = getHeaderThinningToolExecutable();
      // Additional here to ensure that an Executable Artifact exists to disable where the tool
      // is an empty filegroup
      return tool != null && tool.getExecutable() != null;
    }
    return false;
  }

  private FilesToRunProvider getHeaderThinningToolExecutable() {
    return ruleContext
        .getPrerequisite(ObjcRuleClasses.HEADER_SCANNER_ATTRIBUTE, Mode.HOST)
        .getProvider(FilesToRunProvider.class);
  }

  private void registerHeaderScanningActions(
      CcCompilationOutputs ccCompilationOutputs,
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts)
      throws RuleErrorException {
    // PIC is not used for Obj-C builds, if that changes this method will need to change
    if (includeProcessingType != IncludeProcessingType.HEADER_THINNING
        || ccCompilationOutputs.getObjectFiles(false).isEmpty()) {
      return;
    }

    try {
      ImmutableList.Builder<ObjcHeaderThinningInfo> headerThinningInfos = ImmutableList.builder();
      AnalysisEnvironment analysisEnvironment = ruleContext.getAnalysisEnvironment();
      for (Artifact objectFile : ccCompilationOutputs.getObjectFiles(false)) {
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
                    intermediateArtifacts.headersListFile(objectFile),
                    action.getCompilerOptions()));
          }
        }
      }
      registerHeaderScanningActions(
          headerThinningInfos.build(), objcProvider, compilationArtifacts);
    } catch (CommandLineExpansionException e) {
      throw ruleContext.throwWithRuleError(e.getMessage());
    }
  }

  /**
   * Creates and registers ObjcHeaderScanning {@link SpawnAction}. Groups all the actions by their
   * compilation command line arguments and creates a ObjcHeaderScanning action for each unique one.
   *
   * <p>The number of sources to scan per actions are bounded so that targets with a high number of
   * sources are not penalized. A large number of sources may require a lot of processing
   * particularly when the headers required for different sources vary greatly and the caching
   * mechanism in the tool is largely useless. In these instances these actions would benefit by
   * being distributed so they don't contribute to the critical path. The partition size is
   * configurable so that it can be tuned.
   */
  private void registerHeaderScanningActions(
      ImmutableList<ObjcHeaderThinningInfo> headerThinningInfo,
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts) {
    if (headerThinningInfo.isEmpty()) {
      return;
    }

    ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
        objcHeaderThinningInfoByCommandLine = groupActionsByCommandLine(headerThinningInfo);
    // Register a header scanning spawn action for each unique set of command line arguments
    for (ImmutableList<String> args : objcHeaderThinningInfoByCommandLine.keySet()) {
      // As infos is in insertion order we should reliably get the same sublists below
      for (List<ObjcHeaderThinningInfo> partition :
          Lists.partition(
              objcHeaderThinningInfoByCommandLine.get(args),
              objcConfiguration.objcHeaderThinningPartitionSize())) {
        registerHeaderScanningAction(objcProvider, compilationArtifacts, args, partition);
      }
    }
  }

  private void registerHeaderScanningAction(
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      ImmutableList<String> args,
      List<ObjcHeaderThinningInfo> infos) {
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setMnemonic("ObjcHeaderScanning")
            .setExecutable(getHeaderThinningToolExecutable())
            .addInputs(
                ruleContext
                    .getPrerequisiteArtifacts(ObjcRuleClasses.APPLE_SDK_ATTRIBUTE, Mode.TARGET)
                    .list());
    CustomCommandLine.Builder cmdLine =
        CustomCommandLine.builder()
            .add("--arch", appleConfiguration.getSingleArchitecture().toLowerCase())
            .add("--platform", appleConfiguration.getSingleArchPlatform().getLowerCaseNameInPlist())
            .add(
                "--sdk_version",
                XcodeConfig.getXcodeConfigInfo(ruleContext)
                    .getSdkVersionForPlatform(appleConfiguration.getSingleArchPlatform())
                    .toStringWithMinimumComponents(2))
            .add(
                "--xcode_version",
                XcodeConfig.getXcodeConfigInfo(ruleContext)
                    .getXcodeVersion()
                    .toStringWithMinimumComponents(2))
            .add("--");
    for (ObjcHeaderThinningInfo info : infos) {
      cmdLine.addFormatted(
          "%s:%s", info.sourceFile.getExecPath(), info.headersListFile.getExecPath());
      builder.addInput(info.sourceFile).addOutput(info.headersListFile);
    }
    ruleContext.registerAction(
        builder
            .addCommandLine(cmdLine.add("--").addAll(args).build())
            .addInputs(compilationArtifacts.getPrivateHdrs())
            .addTransitiveInputs(attributes.hdrs())
            .addTransitiveInputs(objcProvider.get(ObjcProvider.HEADER))
            .addInputs(getPchFile().asSet())
            .build(ruleContext));
  }

  /**
   * Groups {@link ObjcHeaderThinningInfo} objects based on the command line arguments of the
   * ObjcCompile action.
   *
   * <p>Grouping by command line arguments allows {@link
   * #registerHeaderScanningActions(ImmutableList, ObjcProvider, CompilationArtifacts)} to create a
   * {@link SpawnAction} based on the compiler command line flags that may cause a difference in
   * behaviour by the preprocessor. Some of the command line arguments must be filtered out as they
   * change with every source {@link Artifact}; for example the object file (-o) and dotd filenames
   * (-MF). These arguments are known not to change the preprocessor behaviour.
   *
   * @param headerThinningInfos information for compile actions that require header thinning
   * @return values in {@code headerThinningInfos} grouped by compile action command line arguments
   */
  private static ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
      groupActionsByCommandLine(ImmutableList<ObjcHeaderThinningInfo> headerThinningInfos) {
    // Maintain insertion order so that iteration in #registerHeaderScanningActions is deterministic
    ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
        objcHeaderThinningInfoByCommandLine = ArrayListMultimap.create();
    for (ObjcHeaderThinningInfo info : headerThinningInfos) {
      ImmutableList.Builder<String> filteredArgumentsBuilder = ImmutableList.builder();
      List<String> arguments = info.arguments;
      for (int i = 0; i < arguments.size(); ++i) {
        String arg = arguments.get(i);
        if (arg.equals("-MF") || arg.equals("-o") || arg.equals("-c")) {
          ++i;
        } else if (!arg.equals("-MD")) {
          filteredArgumentsBuilder.add(arg);
        }
      }
      objcHeaderThinningInfoByCommandLine.put(filteredArgumentsBuilder.build(), info);
    }
    return objcHeaderThinningInfoByCommandLine;
  }

  public static Optional<Artifact> getCustomModuleMap(RuleContext ruleContext) {
    if (ruleContext.attributes().has("module_map", BuildType.LABEL)) {
      return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("module_map", Mode.TARGET));
    }
    return Optional.absent();
  }
}
