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
import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
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
import com.google.common.collect.ImmutableCollection;
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
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.FdoContext;
import com.google.devtools.build.lib.rules.cpp.FeatureConfigurationForStarlark;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ObjcVariablesExtension.VariableCategory;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
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
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

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
public class CompilationSupport implements StarlarkValue {

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

  private static final String DEAD_STRIP_FEATURE_NAME = "dead_strip";

  private static final String GENERATE_LINKMAP_FEATURE_NAME = "generate_linkmap";

  private static final ImmutableList<String> OBJC_ACTIONS =
      ImmutableList.of(
          "objc-compile",
          "objc++-compile",
          "objc-archive",
          "objc-fully-link",
          "objc-executable",
          "objc++-executable");

  /** Returns the location of the xcrunwrapper tool. */
  public static final FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$xcrunwrapper");
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

  private static ImmutableList<String> pathsToIncludeArgs(Iterable<PathFragment> paths) {
    ImmutableList.Builder<String> builder = ImmutableList.<String>builder();
    for (PathFragment path : paths) {
      builder.add("-I" + path);
    }
    return builder.build();
  }

  private String getPurpose() {
    // ProtoSupport creates multiple {@code CcCompilationContext}s for a single rule, potentially
    // multiple archives per build configuration. This covers that worst case.
    return "Objc_build_arch_"
        + buildConfiguration.getMnemonic()
        + "_with_suffix_"
        + intermediateArtifacts.archiveFileNameSuffix();
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
      CppModuleMap moduleMap,
      FeatureConfiguration moduleMapFeatureConfiguration,
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
                cppSemantics,
                moduleMapFeatureConfiguration,
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
            .addCcCompilationContexts(objcCompilationContext.getCcCompilationContexts())
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
            .setCppModuleMap(moduleMap)
            .setPropagateModuleMapToCompileAction(true)
            .addVariableExtension(extension)
            .setPurpose(purpose)
            .setCodeCoverageEnabled(CcCompilationHelper.isCodeCoverageEnabled(ruleContext))
            .setHeadersCheckingMode(cppSemantics.determineHeadersCheckingMode(ruleContext));

    if (pchHdr != null) {
      result.addPublicTextualHeaders(ImmutableList.of(pchHdr));
    }

    if (moduleMap.getArtifact().isSourceArtifact() || !generateModuleMap) {
      result.doNotGenerateModuleMap();
    }

    return result.compile(ruleContext);
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
    FeatureConfiguration featureConfiguration =
        getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, cppSemantics);
    FeatureConfiguration featureConfigurationForSwiftModuleMap =
        getFeatureConfigurationForSwiftModuleMap(
            ruleContext, toolchain, buildConfiguration, cppSemantics);

    // Generate up to two module maps, while minimizing the number of actions created.  If
    // module_map feature is off, generate a swift module map.  If module_map feature is on,
    // generate a layering check and a swift module map.  In the latter case, the layering check
    // module map must be the primary one.
    //
    // TODO(waltl): Delete this logic when swift module map is migrated to swift_library.
    CppModuleMap primaryModuleMap;
    FeatureConfiguration primaryModuleMapFeatureConfiguration;
    Optional<CppModuleMap> extraModuleMap;
    Optional<FeatureConfiguration> extraModuleMapFeatureConfiguration;
    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS)) {
      primaryModuleMap = intermediateArtifacts.internalModuleMap();
      primaryModuleMapFeatureConfiguration = featureConfiguration;
      extraModuleMap = Optional.of(intermediateArtifacts.swiftModuleMap());
      extraModuleMapFeatureConfiguration = Optional.of(featureConfigurationForSwiftModuleMap);
    } else {
      primaryModuleMap = intermediateArtifacts.swiftModuleMap();
      primaryModuleMapFeatureConfiguration = featureConfigurationForSwiftModuleMap;
      extraModuleMap = Optional.absent();
      extraModuleMapFeatureConfiguration = Optional.absent();
    }

    String purpose = String.format("%s_objc_arc", getPurpose());
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
            primaryModuleMap,
            primaryModuleMapFeatureConfiguration,
            purpose,
            /* generateModuleMap= */ true,
            /* shouldProcessHeaders= */ true);

    purpose = String.format("%s_non_objc_arc", getPurpose());
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
            primaryModuleMap,
            primaryModuleMapFeatureConfiguration,
            purpose,
            // Only generate the module map once (see above) and re-use it here.
            /* generateModuleMap= */ false,
            // We only need to validate headers once, in arc compilation above.
            /* shouldProcessHeaders= */ false);

    if (extraModuleMap.isPresent() && !extraModuleMap.get().getArtifact().isSourceArtifact()) {
      generateExtraModuleMap(
          extraModuleMap.get(),
          publicHdrs,
          privateHdrs,
          objcCompilationContext.getPublicTextualHeaders(),
          getPchFile(),
          objcCompilationContext.getCcCompilationContexts(),
          extraModuleMapFeatureConfiguration.get());
    }

    CcLinkingHelper resultLink =
        new CcLinkingHelper(
                ruleContext,
                ruleContext.getLabel(),
                ruleContext,
                ruleContext,
                cppSemantics,
                featureConfiguration,
                ccToolchain,
                fdoContext,
                buildConfiguration,
                buildConfiguration.getFragment(CppConfiguration.class),
                ruleContext.getSymbolGenerator(),
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
            .setTestOrTestOnlyTarget(ruleContext.isTestTarget() || ruleContext.isTestOnlyTarget())
            .addCcLinkingContexts(
                CppHelper.getLinkingContextsFromDeps(
                    ImmutableList.copyOf(ruleContext.getPrerequisites("deps"))))
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
    ccCompilationContextBuilder.addDependentCcCompilationContexts(
        Arrays.asList(
            objcArcCompilationInfo.getCcCompilationContext(),
            nonObjcArcCompilationInfo.getCcCompilationContext()),
        ImmutableList.of());
    ccCompilationContextBuilder.setPurpose(
        String.format("%s_merged_arc_non_arc_objc", getPurpose()));

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

    CppConfiguration cppConfiguration = buildConfiguration.getFragment(CppConfiguration.class);
    Map<String, NestedSet<Artifact>> arcOutputGroups =
        CcCompilationHelper.buildOutputGroupsForEmittingCompileProviders(
            objcArcCompilationInfo.getCcCompilationOutputs(),
            objcArcCompilationInfo.getCcCompilationContext(),
            cppConfiguration,
            ccToolchain,
            featureConfiguration,
            ruleContext,
            /* generateHiddenTopLevelGroup= */ true);

    Map<String, NestedSet<Artifact>> nonArcOutputGroups =
        CcCompilationHelper.buildOutputGroupsForEmittingCompileProviders(
            nonObjcArcCompilationInfo.getCcCompilationOutputs(),
            nonObjcArcCompilationInfo.getCcCompilationContext(),
            cppConfiguration,
            ccToolchain,
            featureConfiguration,
            ruleContext,
            /* generateHiddenTopLevelGroup= */ true);

    Map<String, NestedSet<Artifact>> mergedOutputGroups =
        CcCommon.mergeOutputGroups(ImmutableList.of(arcOutputGroups, nonArcOutputGroups));

    return new CompilationResult(
        ccCompilationContextBuilder.build(),
        compilationOutputs,
        ImmutableMap.copyOf(mergedOutputGroups));
  }

  @StarlarkMethod(name = "feature_configuration", documented = false, structField = true)
  public FeatureConfigurationForStarlark getFeatureConfigurationForStarlark() {
    return FeatureConfigurationForStarlark.from(
        getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, cppSemantics),
        ruleContext.getFragment(CppConfiguration.class),
        buildConfiguration.getOptions());
  }

  @StarlarkMethod(name = "cc_toolchain", documented = false, structField = true)
  public CcToolchainProvider getToolchain() {
    return toolchain;
  }

  @StarlarkMethod(name = "output_group_info", documented = false, structField = true)
  public OutputGroupInfo getOutputGroupInfo() {
    return new OutputGroupInfo(ImmutableMap.copyOf(outputGroupCollector));
  }

  private FeatureConfiguration getFeatureConfiguration(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration configuration,
      CppSemantics cppSemantics,
      boolean forSwiftModuleMap) {
    ImmutableSet.Builder<String> activatedCrosstoolSelectables =
        ImmutableSet.<String>builder()
            .addAll(ruleContext.getFeatures())
            .addAll(OBJC_ACTIONS)
            .add(CppRuleClasses.LANG_OBJC);

    if (configuration.getFragment(ObjcConfiguration.class).shouldStripBinary()) {
      activatedCrosstoolSelectables.add(DEAD_STRIP_FEATURE_NAME);
    }
    if (configuration.getFragment(ObjcConfiguration.class).generateLinkmap()) {
      activatedCrosstoolSelectables.add(GENERATE_LINKMAP_FEATURE_NAME);
    }

    ImmutableSet.Builder<String> disabledFeatures =
        ImmutableSet.<String>builder().addAll(ruleContext.getDisabledFeatures());
    if (disableParseHeaders) {
      disabledFeatures.add(CppRuleClasses.PARSE_HEADERS);
    }
    if (disableLayeringCheck) {
      disabledFeatures.add(CppRuleClasses.LAYERING_CHECK);
    }
    if (forSwiftModuleMap) {
      activatedCrosstoolSelectables
          .add(CppRuleClasses.MODULE_MAPS)
          .add(CppRuleClasses.COMPILE_ALL_MODULES)
          .add(CppRuleClasses.ONLY_DOTH_HEADERS_IN_MODULE_MAPS)
          .add(CppRuleClasses.EXCLUDE_PRIVATE_HEADERS_IN_MODULE_MAPS)
          .add(CppRuleClasses.MODULE_MAP_WITHOUT_EXTERN_MODULE)
          .add(CppRuleClasses.ONLY_DOTH_HEADERS_IN_MODULE_MAPS);
      disabledFeatures.add(CppRuleClasses.GENERATE_SUBMODULES);
    }

    return CcCommon.configureFeaturesOrReportRuleError(
        ruleContext,
        buildConfiguration,
        activatedCrosstoolSelectables.build(),
        disabledFeatures.build(),
        ccToolchain,
        cppSemantics);
  }

  private FeatureConfiguration getFeatureConfiguration(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration configuration,
      CppSemantics cppSemantics) {
    return getFeatureConfiguration(
        ruleContext, ccToolchain, configuration, cppSemantics, /* forSwiftModuleMap= */ false);
  }

  private FeatureConfiguration getFeatureConfigurationForSwiftModuleMap(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration configuration,
      CppSemantics cppSemantics) {
    return getFeatureConfiguration(
        ruleContext, ccToolchain, configuration, cppSemantics, /* forSwiftModuleMap= */ true);
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
        ruleContext.getPrerequisiteArtifacts("srcs").errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(
            ruleContext
                .getPrerequisiteArtifacts("non_arc_srcs")
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
  private final CppSemantics cppSemantics;
  private final CompilationAttributes attributes;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Map<String, NestedSet<Artifact>> outputGroupCollector;
  private final ImmutableList.Builder<Artifact> objectFilesCollector;
  private final CcToolchainProvider toolchain;
  private final boolean usePch;
  private final boolean disableLayeringCheck;
  private final boolean disableParseHeaders;
  private Optional<CcCompilationContext> ccCompilationContext;

  private void setCcCompilationContext(CcCompilationContext ccCompilationContext) {
    checkState(!this.ccCompilationContext.isPresent());
    this.ccCompilationContext = Optional.of(ccCompilationContext);
  }

  @StarlarkMethod(name = "compilation_context", documented = false, structField = true)
  public CcCompilationContext getCcCompilationContext() {
    checkState(ccCompilationContext.isPresent());
    return ccCompilationContext.get();
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
      CppSemantics cppSemantics,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      Map<String, NestedSet<Artifact>> outputGroupCollector,
      ImmutableList.Builder<Artifact> objectFilesCollector,
      CcToolchainProvider toolchain,
      boolean usePch,
      boolean disableLayeringCheck,
      boolean disableParseHeaders)
      throws RuleErrorException {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
    this.cppSemantics = cppSemantics;
    this.attributes = compilationAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
    this.outputGroupCollector = outputGroupCollector;
    this.objectFilesCollector = objectFilesCollector;
    this.ccCompilationContext = Optional.absent();
    this.usePch = usePch;
    this.disableLayeringCheck = disableLayeringCheck;
    this.disableParseHeaders = disableParseHeaders;
    if (toolchain == null
        && (ruleContext
                .attributes()
                .has(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, BuildType.LABEL)
            || ruleContext
                .attributes()
                .has(
                    CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK,
                    BuildType.LABEL))) {
      toolchain = CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    }

    this.toolchain = toolchain;
  }

  /** Builder for {@link CompilationSupport} */
  public static class Builder {
    private final RuleContext ruleContext;
    private final CppSemantics cppSemantics;
    private BuildConfiguration buildConfiguration;
    private IntermediateArtifacts intermediateArtifacts;
    private CompilationAttributes compilationAttributes;
    private Map<String, NestedSet<Artifact>> outputGroupCollector;
    private ImmutableList.Builder<Artifact> objectFilesCollector;
    private CcToolchainProvider toolchain;
    private boolean usePch = true;
    private boolean disableLayeringCheck = false;
    private boolean disableParseHeaders = false;

    public Builder(RuleContext ruleContext, CppSemantics cppSemantics) {
      this.ruleContext = ruleContext;
      this.cppSemantics = cppSemantics;
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

    /** Sets that this {@link CompilationSupport} will disable layering check. */
    public Builder disableLayeringCheck() {
      this.disableLayeringCheck = true;
      return this;
    }

    /** Sets that this {@link CompilationSupport} will disable parse headers. */
    public Builder disableParseHeaders() {
      this.disableParseHeaders = true;
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
    public CompilationSupport build() throws InterruptedException, RuleErrorException {
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
          cppSemantics,
          intermediateArtifacts,
          compilationAttributes,
          outputGroupCollector,
          objectFilesCollector,
          toolchain,
          usePch,
          disableLayeringCheck,
          disableParseHeaders);
    }
  }

  @StarlarkMethod(name = "instrumented_files_info", documented = false, structField = true)
  public InstrumentedFilesInfo getInstrumentedFilesProviderForStarlark() {
    return getInstrumentedFilesProvider(objectFilesCollector.build());
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
        /* withBaselineCoverage= */ true,
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
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("srcs").list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      ImmutableSet<Artifact> nonArcSrcsSet =
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("non_arc_srcs").list());
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

  @StarlarkMethod(name = "validate_attributes", documented = false)
  public void validateAttributesForStarlark() throws EvalException {
    try {
      validateAttributes();
    } catch (RuleErrorException ruleErrorException) {
      throw new EvalException(ruleErrorException);
    }
  }

  @StarlarkMethod(
      name = "register_compile_and_archive_actions",
      documented = false,
      parameters = {@Param(name = "common", positional = false, named = true)})
  public void registerCompileAndArchiveActionsForStarlark(ObjcCommon common)
      throws EvalException, InterruptedException {
    try {
      registerCompileAndArchiveActions(
          common, ExtraCompileArgs.NONE, ImmutableList.<PathFragment>of());
    } catch (RuleErrorException ruleErrorException) {
      throw new EvalException(ruleErrorException);
    }
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
      ExtraCompileArgs extraCompileArgs,
      List<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    checkNotNull(toolchain);
    checkNotNull(toolchain.getFdoContext());
    ObjcVariablesExtension.Builder extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
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

    setCcCompilationContext(compilationResult.getCcCompilationContext());

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
          extraCompileArgs,
          priorityHeaders);
    }
    return this;
  }

  private StrippingType getStrippingType(ExtraLinkArgs extraLinkArgs) {
    if (Iterables.contains(extraLinkArgs, "-dynamiclib")) {
      return StrippingType.DYNAMIC_LIB;
    }
    if (Iterables.contains(extraLinkArgs, "-bundle")) {
      return StrippingType.LOADABLE_BUNDLE;
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
   * -dead_strip}).
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   * @param ccLinkingContexts the linking contexts from this rule's dependencies
   * @param j2ObjcMappingFileProvider contains mapping files for j2objc transpilation
   * @param j2ObjcEntryClassProvider contains j2objc entry class information for dead code removal
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   * @return this compilation support
   */
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      Iterable<CcLinkingContext> ccLinkingContexts,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      boolean isStampingEnabled)
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
    FeatureConfiguration featureConfiguration =
        getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, cppSemantics);

    Label binaryLabel = null;
    try {
      binaryLabel =
          Label.create(ruleContext.getLabel().getPackageIdentifier(), binaryToLink.getFilename());
    } catch (LabelSyntaxException e) {
      // Formed from existing label, just replacing name with artifact name.
    }

    CppConfiguration cppConfiguration = buildConfiguration.getFragment(CppConfiguration.class);
    CcLinkingHelper executableLinkingHelper =
        new CcLinkingHelper(
                ruleContext,
                binaryLabel,
                ruleContext,
                ruleContext,
                cppSemantics,
                featureConfiguration,
                toolchain,
                toolchain.getFdoContext(),
                buildConfiguration,
                cppConfiguration,
                ruleContext.getSymbolGenerator(),
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setIsStampingEnabled(isStampingEnabled)
            .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget())
            .addNonCodeLinkerInputs(bazelBuiltLibraries)
            .addNonCodeLinkerInputs(objcProvider.getCcLibraries())
            .addNonCodeLinkerInputs(ImmutableList.copyOf(prunedJ2ObjcArchives))
            .addNonCodeLinkerInputs(ImmutableList.copyOf(extraLinkInputs))
            .addNonCodeLinkerInputs(ImmutableList.copyOf(attributes.linkInputs()))
            .addNonCodeLinkerInputs(ImmutableList.of(inputFileList))
            .addTransitiveAdditionalLinkerInputs(objcProvider.get(IMPORTED_LIBRARY))
            .addTransitiveAdditionalLinkerInputs(objcProvider.get(STATIC_FRAMEWORK_FILE))
            .addTransitiveAdditionalLinkerInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE))
            .addTransitiveAdditionalLinkerInputs(objcProvider.get(LINK_INPUTS))
            .setShouldCreateStaticLibraries(false)
            .setDynamicLinkType(linkType)
            .setLinkingMode(LinkingMode.STATIC)
            .addLinkopts(ImmutableList.copyOf(extraLinkArgs));

    ImmutableList.Builder<Artifact> linkerOutputs = ImmutableList.builder();

    if (cppConfiguration.appleGenerateDsym()) {
      Artifact dsymSymbol =
          objcConfiguration.shouldStripBinary()
              ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
              : intermediateArtifacts.dsymSymbolForStrippedBinary();
      extensionBuilder
          .setDsymSymbol(dsymSymbol)
          .addVariableCategory(VariableCategory.DSYM_VARIABLES);
      linkerOutputs.add(dsymSymbol);
    }

    if (objcConfiguration.generateLinkmap()) {
      Artifact linkmap = intermediateArtifacts.linkmap();
      extensionBuilder.setLinkmap(linkmap).addVariableCategory(VariableCategory.LINKMAP_VARIABLES);
      linkerOutputs.add(linkmap);
    }

    if (cppConfiguration.getAppleBitcodeMode() == AppleBitcodeMode.EMBEDDED) {
      Artifact bitcodeSymbolMap = intermediateArtifacts.bitcodeSymbolMap();
      extensionBuilder
          .setBitcodeSymbolMap(bitcodeSymbolMap)
          .addVariableCategory(VariableCategory.BITCODE_VARIABLES);
      linkerOutputs.add(bitcodeSymbolMap);
    }

    executableLinkingHelper.addVariableExtension(extensionBuilder.build());

    executableLinkingHelper.addLinkerOutputs(linkerOutputs.build());

    CcLinkingContext.Builder linkstampsBuilder = CcLinkingContext.builder();
    for (CcLinkingContext context : ccLinkingContexts) {
      linkstampsBuilder.addLinkstamps(context.getLinkstamps().toList());
    }
    CcLinkingContext linkstamps = linkstampsBuilder.build();
    executableLinkingHelper.addCcLinkingContexts(ImmutableList.of(linkstamps));

    executableLinkingHelper.link(CcCompilationOutputs.EMPTY);

    ImmutableCollection<Artifact> linkstampValues =
        CppLinkActionBuilder.mapLinkstampsToOutputs(
                linkstamps.getLinkstamps().toSet(),
                ruleContext,
                ruleContext.getRepository(),
                buildConfiguration,
                binaryToLink,
                CppLinkAction.DEFAULT_ARTIFACT_FACTORY)
            .values();

    // Populate the input file list with both the compiled object files and any linkstamp object
    // files.
    registerObjFilelistAction(
        ImmutableSet.<Artifact>builder().addAll(objFiles).addAll(linkstampValues).build(),
        inputFileList);

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
          buildConfiguration.getGenfilesFragment(ruleContext.getRepository())
              + "/"
              + OBJC_MODULE_CACHE_DIR_NAME;
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

    ObjcVariablesExtension extension =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setObjcProvider(objcProvider)
            .setConfiguration(buildConfiguration)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setFullyLinkArchive(outputArchive)
            .addVariableCategory(VariableCategory.FULLY_LINK_VARIABLES)
            .build();

    Label archiveLabel = null;
    try {
      archiveLabel =
          Label.create(
              ruleContext.getLabel().getPackageIdentifier(),
              FileSystemUtils.removeExtension(outputArchive.getFilename()));
    } catch (LabelSyntaxException e) {
      // Formed from existing label, just replacing name with artifact name.
    }

    new CcLinkingHelper(
            ruleContext,
            archiveLabel,
            ruleContext,
            ruleContext,
            cppSemantics,
            getFeatureConfiguration(ruleContext, toolchain, buildConfiguration, cppSemantics),
            toolchain,
            toolchain.getFdoContext(),
            buildConfiguration,
            buildConfiguration.getFragment(CppConfiguration.class),
            ruleContext.getSymbolGenerator(),
            TargetUtils.getExecutionInfo(
                ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
        .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
        .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
        .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget())
        .addNonCodeLinkerInputs(objcProvider.getObjcLibraries())
        .addNonCodeLinkerInputs(objcProvider.getCcLibraries())
        .addTransitiveAdditionalLinkerInputs(objcProvider.get(IMPORTED_LIBRARY))
        .setLinkingMode(LinkingMode.STATIC)
        .setStaticLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE)
        .setShouldCreateDynamicLibrary(false)
        .addVariableExtension(extension)
        .link(CcCompilationOutputs.EMPTY);

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
    Artifact pruner = ruleContext.getPrerequisiteArtifact("$j2objc_dead_code_pruner");
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
              .getPrerequisite("$dummy_lib", ObjcProvider.STARLARK_CONSTRUCTOR)
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
    LOADABLE_BUNDLE,
    KERNEL_EXTENSION
  }

  /**
   * Registers an action that uses the 'strip' tool to perform binary stripping on the given binary
   * subject to the given {@link StrippingType}.
   */
  private void registerBinaryStripAction(Artifact binaryToLink, StrippingType strippingType) {
    final ImmutableList<String> stripArgs;
    switch (strippingType) {
      case DYNAMIC_LIB:
      case LOADABLE_BUNDLE:
      case KERNEL_EXTENSION:
        // For dylibs, loadable bundles, and kexts, must strip only local symbols.
        stripArgs = ImmutableList.of("-x");
        break;
      case DEFAULT:
        stripArgs = ImmutableList.<String>of();
        break;
      default:
        throw new IllegalArgumentException("Unsupported stripping type " + strippingType);
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

  private Optional<Artifact> getPchFile() {
    if (!usePch) {
      return Optional.absent();
    }
    Artifact pchHdr = null;
    if (ruleContext.attributes().has("pch", BuildType.LABEL)) {
      pchHdr = ruleContext.getPrerequisiteArtifact("pch");
    }
    return Optional.fromNullable(pchHdr);
  }

  /**
   * Registers an action that will generate a clang module map for this target, using the hdrs
   * attribute of this rule.
   */
  public CompilationSupport registerGenerateModuleMapAction(
      CompilationArtifacts compilationArtifacts) throws RuleErrorException, InterruptedException {
    // TODO(bazel-team): Include textual headers in the module map when Xcode 6 support is
    // dropped.
    // TODO(b/32225593): Include private headers in the module map.
    CcCompilationHelper ccCompilationHelper =
        new CcCompilationHelper(
            ruleContext,
            ruleContext,
            ruleContext.getLabel(),
            CppHelper.getGrepIncludes(ruleContext),
            cppSemantics,
            getFeatureConfigurationForSwiftModuleMap(
                ruleContext, toolchain, buildConfiguration, cppSemantics),
            CcCompilationHelper.SourceCategory.CC_AND_OBJC,
            toolchain,
            toolchain.getFdoContext(),
            buildConfiguration,
            TargetUtils.getExecutionInfo(
                ruleContext.getRule(), ruleContext.isAllowTagsPropagation()),
            /* shouldProcessHeaders= */ false);

    ImmutableSortedSet<Artifact> publicHeaders =
        Stream.concat(
                attributes.hdrs().toList().stream(),
                compilationArtifacts.getAdditionalHdrs().toList().stream())
            .collect(toImmutableSortedSet(naturalOrder()));

    CppModuleMap moduleMap = intermediateArtifacts.swiftModuleMap();

    ccCompilationHelper.setCppModuleMap(moduleMap).addPublicHeaders(publicHeaders);

    ccCompilationHelper.compile(ruleContext);

    return this;
  }

  /** Registers an action to generate an extra clang module map. */
  private CompilationSupport generateExtraModuleMap(
      CppModuleMap moduleMap,
      Collection<Artifact> publicHeaders,
      Collection<Artifact> privateHeaders,
      List<Artifact> textualHeaders,
      Optional<Artifact> pchHdr,
      Iterable<CcCompilationContext> ccCompilationContexts,
      FeatureConfiguration featureConfiguration)
      throws RuleErrorException, InterruptedException {
    String purpose = String.format("%s_extra_module_map", getPurpose());
    CcCompilationHelper result =
        new CcCompilationHelper(
            ruleContext,
            ruleContext,
            ruleContext.getLabel(),
            CppHelper.getGrepIncludes(ruleContext),
            cppSemantics,
            featureConfiguration,
            CcCompilationHelper.SourceCategory.CC_AND_OBJC,
            toolchain,
            toolchain.getFdoContext(),
            buildConfiguration,
            TargetUtils.getExecutionInfo(
                ruleContext.getRule(), ruleContext.isAllowTagsPropagation()),
            /* shouldProcessHeaders= */ false);

    result
        .addPublicHeaders(publicHeaders)
        .addPublicTextualHeaders(textualHeaders)
        .addPrivateHeaders(privateHeaders)
        .addCcCompilationContexts(ccCompilationContexts)
        .setCppModuleMap(moduleMap)
        .setPurpose(purpose);
    if (pchHdr.isPresent()) {
      result.addPublicTextualHeaders(ImmutableList.of(pchHdr.get()));
    }
    result.compile(ruleContext);

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
      return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("module_map"));
    }
    return Optional.absent();
  }
}
