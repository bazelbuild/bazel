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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.MapBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A ConfiguredTarget for <code>cc_library</code> rules.
 */
public abstract class CcLibrary implements RuleConfiguredTargetFactory {

  /** A string constant for the name of archive library(.a, .lo) output group. */
  public static final String ARCHIVE_LIBRARY_OUTPUT_GROUP_NAME = "archive";

  /** A string constant for the name of dynamic library output group. */
  public static final String DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME = "dynamic_library";

  /** A string constant for the name of Windows def file output group. */
  public static final String DEF_FILE_OUTPUT_GROUP_NAME = "def_file";

  private final CppSemantics semantics;

  protected CcLibrary(CppSemantics semantics) {
    this.semantics = semantics;
  }

  // These file extensions don't generate object files.
  private static final FileTypeSet NO_OBJECT_GENERATING_FILETYPES = FileTypeSet.of(
      CppFileTypes.CPP_HEADER, CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE,
      CppFileTypes.ALWAYS_LINK_LIBRARY, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
      CppFileTypes.SHARED_LIBRARY, CppFileTypes.VERSIONED_SHARED_LIBRARY);

  @Override
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(context);
    boolean linkStatic = context.attributes().get("linkstatic", Type.BOOLEAN);
    init(
        semantics,
        context,
        builder,
        /* additionalCopts= */ ImmutableList.of(),
        /* soFilename= */ null,
        context.attributes().get("alwayslink", Type.BOOLEAN),
        /* neverLink= */ false,
        linkStatic,
        /* addDynamicRuntimeInputArtifactsToRunfiles= */ false);
    return builder.build();
  }

  public static void init(
      CppSemantics semantics,
      RuleContext ruleContext,
      RuleConfiguredTargetBuilder targetBuilder,
      ImmutableList<String> additionalCopts,
      PathFragment soFilename,
      boolean alwaysLink,
      boolean neverLink,
      boolean linkStatic,
      boolean addDynamicRuntimeInputArtifactsToRunfiles)
      throws RuleErrorException, InterruptedException {
    CcCommon.checkRuleLoadedThroughMacro(ruleContext);
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("linked_statically_by")
        && !ruleContext
            .getAnalysisEnvironment()
            .getSkylarkSemantics()
            .experimentalCcSharedLibrary()) {
      ruleContext.ruleError(
          "The attribute 'linked_statically_by' can only be used with the flag"
              + " --experimental_cc_shared_library.");
    }
    semantics.validateDeps(ruleContext);
    if (ruleContext.hasErrors()) {
      return;
    }

    final CcCommon common = new CcCommon(ruleContext);
    common.reportInvalidOptions(ruleContext);

    CcToolchainProvider ccToolchain = common.getToolchain();
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);

    ImmutableMap.Builder<String, String> toolchainMakeVariables = ImmutableMap.builder();
    ccToolchain.addGlobalMakeVariables(toolchainMakeVariables);
    ruleContext.initConfigurationMakeVariableContext(
        new MapBackedMakeVariableSupplier(toolchainMakeVariables.build()),
        new CcFlagsSupplier(ruleContext));

    FdoContext fdoContext = common.getFdoContext();
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, ccToolchain);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    semantics.validateAttributes(ruleContext);
    if (ruleContext.hasErrors()) {
      return;
    }

    ImmutableList<TransitiveInfoCollection> deps =
        ImmutableList.copyOf(ruleContext.getPrerequisites("deps", Mode.TARGET));
    CppHelper.checkProtoLibrariesInDeps(ruleContext, deps);
    if (ruleContext.hasErrors()) {
      return;
    }
    Iterable<CcInfo> ccInfosFromDeps = AnalysisUtils.getProviders(deps, CcInfo.PROVIDER);
    CcCompilationHelper compilationHelper =
        new CcCompilationHelper(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                CppHelper.getGrepIncludes(ruleContext),
                semantics,
                featureConfiguration,
                ccToolchain,
                fdoContext,
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .fromCommon(common, additionalCopts)
            .addSources(common.getSources())
            .addPrivateHeaders(common.getPrivateHeaders())
            .addPublicHeaders(common.getHeaders())
            .setCodeCoverageEnabled(CcCompilationHelper.isCodeCoverageEnabled(ruleContext))
            .addCcCompilationContexts(
                Streams.stream(ccInfosFromDeps)
                    .map(CcInfo::getCcCompilationContext)
                    .collect(ImmutableList.toImmutableList()))
            .addCcCompilationContexts(
                ImmutableList.of(CcCompilationHelper.getStlCcCompilationContext(ruleContext)))
            .setHeadersCheckingMode(semantics.determineHeadersCheckingMode(ruleContext));

    CcLinkingHelper linkingHelper =
        new CcLinkingHelper(
                ruleContext,
                ruleContext.getLabel(),
                ruleContext,
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                fdoContext,
                ruleContext.getConfiguration(),
                ruleContext.getFragment(CppConfiguration.class),
                ruleContext.getSymbolGenerator(),
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .fromCommon(ruleContext, common)
            .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
            .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget())
            .addLinkopts(common.getLinkopts())
            .emitInterfaceSharedLibraries(true)
            .setAlwayslink(alwaysLink)
            .setNeverLink(neverLink)
            .addLinkstamps(ruleContext.getPrerequisites("linkstamp", Mode.TARGET));

    Artifact soImplArtifact = null;
    boolean supportsDynamicLinker = ccToolchain.supportsDynamicLinker(featureConfiguration);
    // TODO(djasper): This is hacky. We should actually try to figure out whether we generate
    // ccOutputs.
    boolean createDynamicLibrary =
        !linkStatic
            && supportsDynamicLinker
            && (appearsToHaveObjectFiles(ruleContext.attributes())
                || featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN));
    if (soFilename != null) {
      if (!soFilename.getPathString().endsWith(".so")) { // Sanity check.
        ruleContext.attributeError("outs", "file name must end in '.so'");
      }
      if (createDynamicLibrary) {
        soImplArtifact = ruleContext.getBinArtifact(soFilename);
      }
    }

    if (ruleContext.getRule().isAttrDefined("textual_hdrs", BuildType.LABEL_LIST)) {
      compilationHelper.addPublicTextualHeaders(
          ruleContext.getPrerequisiteArtifacts("textual_hdrs", Mode.TARGET).list());
    }
    if (ruleContext.getRule().isAttrDefined("include_prefix", Type.STRING)
        && ruleContext.attributes().isAttributeValueExplicitlySpecified("include_prefix")) {
      compilationHelper.setIncludePrefix(
          ruleContext.attributes().get("include_prefix", Type.STRING));
    }
    if (ruleContext.getRule().isAttrDefined("strip_include_prefix", Type.STRING)
        && ruleContext.attributes().isAttributeValueExplicitlySpecified("strip_include_prefix")) {
      compilationHelper.setStripIncludePrefix(
          ruleContext.attributes().get("strip_include_prefix", Type.STRING));
    }

    if (common.getLinkopts().contains("-static")) {
      ruleContext.attributeWarning("linkopts", "Using '-static' here won't work. "
                                   + "Did you mean to use 'linkstatic=1' instead?");
    }

    linkingHelper.setShouldCreateDynamicLibrary(createDynamicLibrary);
    linkingHelper.setLinkerOutputArtifact(soImplArtifact);

    // If the reason we're not creating a dynamic library is that the toolchain
    // doesn't support it, then register an action which complains when triggered,
    // which only happens when some rule explicitly depends on the dynamic library.
    if (!createDynamicLibrary && !supportsDynamicLinker) {
      ImmutableList.Builder<Artifact> dynamicLibraries = ImmutableList.builder();
      dynamicLibraries.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              ruleContext.getConfiguration(),
              LinkTargetType.NODEPS_DYNAMIC_LIBRARY));
      if (CppHelper.useInterfaceSharedLibraries(
          cppConfiguration, ccToolchain, featureConfiguration)) {
        dynamicLibraries.add(
            CppHelper.getLinkedArtifact(
                ruleContext,
                ccToolchain,
                ruleContext.getConfiguration(),
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY));
      }
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          dynamicLibraries.build(), "Toolchain does not support dynamic linking"));
    } else if (!createDynamicLibrary
        && ruleContext.attributes().isConfigurable("srcs")) {
      // If "srcs" is configurable, the .so output is always declared because the logic that
      // determines implicit outs doesn't know which value of "srcs" will ultimately get chosen.
      // Here, where we *do* have the correct value, it may not contain any source files to
      // generate an .so with. If that's the case, register a fake generating action to prevent
      // a "no generating action for this artifact" error.
      ImmutableList.Builder<Artifact> dynamicLibraries = ImmutableList.builder();
      dynamicLibraries.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              ruleContext.getConfiguration(),
              LinkTargetType.NODEPS_DYNAMIC_LIBRARY));
      if (CppHelper.useInterfaceSharedLibraries(
          cppConfiguration, ccToolchain, featureConfiguration)) {
        dynamicLibraries.add(
            CppHelper.getLinkedArtifact(
                ruleContext,
                ccToolchain,
                ruleContext.getConfiguration(),
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY));
      }
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          dynamicLibraries.build(), "configurable \"srcs\" triggers an implicit .so output "
          + "even though there are no sources to compile in this configuration"));
    }

    CompilationInfo compilationInfo = compilationHelper.compile();
    CcCompilationOutputs precompiledFilesObjects =
        CcCompilationOutputs.builder()
            .addObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ true))
            .addPicObjectFiles(precompiledFiles.getObjectFiles(/* usePic= */ true))
            .build();
    CcCompilationOutputs ccCompilationOutputs =
        CcCompilationOutputs.builder()
            .merge(precompiledFilesObjects)
            .merge(compilationInfo.getCcCompilationOutputs())
            .build();

    // Generate .a and .so outputs even without object files to fulfill the rule class
    // contract wrt. implicit output files, if the contract says so. Behavior here differs
    // between Bazel and Blaze.
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (ruleContext.getRule().getImplicitOutputsFunction() != ImplicitOutputsFunction.NONE
        || !ccCompilationOutputs.isEmpty()) {
      if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        Artifact generatedDefFile = null;

        Artifact defParser = common.getDefParser();
        if (defParser != null) {
          try {
            generatedDefFile =
                CppHelper.createDefFileActions(
                    ruleContext,
                    defParser,
                    ccCompilationOutputs.getObjectFiles(false),
                    ccToolchain
                        .getFeatures()
                        .getArtifactNameForCategory(
                            ArtifactCategory.DYNAMIC_LIBRARY, ruleContext.getLabel().getName()));
            targetBuilder.addOutputGroup(DEF_FILE_OUTPUT_GROUP_NAME, generatedDefFile);
          } catch (EvalException e) {
            throw ruleContext.throwWithRuleError(e.getMessage());
          }
        }
        linkingHelper.setDefFile(
            CppHelper.getWindowsDefFileForLinking(
                ruleContext, common.getWinDefFile(), generatedDefFile, featureConfiguration));
      }
      ccLinkingOutputs = linkingHelper.link(ccCompilationOutputs);
    }

    ImmutableSortedMap.Builder<String, NestedSet<Artifact>> outputGroups =
        ImmutableSortedMap.naturalOrder();
    if (!ccLinkingOutputs.isEmpty()) {
      outputGroups.putAll(
          addLinkerOutputArtifacts(
              ruleContext,
              ccToolchain,
              cppConfiguration,
              ruleContext.getConfiguration(),
              ccCompilationOutputs,
              featureConfiguration));
    }
    List<LibraryToLink> precompiledLibraries =
        convertPrecompiledLibrariesToLibraryToLink(
            common, ruleContext.getFragment(CppConfiguration.class).forcePic(), precompiledFiles);

    if (!ccCompilationOutputs.isEmpty()) {
      checkIfLinkOutputsCollidingWithPrecompiledFiles(
          ruleContext, ccLinkingOutputs, precompiledLibraries);
    }

    ImmutableList<LibraryToLink> libraryToLinks =
        createLibrariesToLinkList(
            ccLinkingOutputs.getLibraryToLink(),
            precompiledLibraries,
            ccCompilationOutputs.isEmpty());

    CcLinkingContext ccLinkingContext =
        linkingHelper.buildCcLinkingContextFromLibrariesToLink(
            neverLink ? ImmutableList.of() : libraryToLinks,
            compilationInfo.getCcCompilationContext());
    CcNativeLibraryProvider ccNativeLibraryProvider =
        CppHelper.collectNativeCcLibraries(
            ruleContext.getPrerequisites("deps", Mode.TARGET), libraryToLinks);

    /*
     * We always generate a static library, even if there aren't any source files.
     * This keeps things simpler by avoiding special cases when making use of the library.
     * For example, this is needed to ensure that building a library with "bazel build"
     * will also build all of the library's "deps".
     * However, we only generate a dynamic library if there are source files.
     */
    // For now, we don't add the precompiled libraries to the files to build.

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    if (!ccLinkingOutputs.isEmpty()) {
      LibraryToLink artifactsToBuild = ccLinkingOutputs.getLibraryToLink();
      if (artifactsToBuild.getStaticLibrary() != null) {
        filesBuilder.add(artifactsToBuild.getStaticLibrary());
      }
      if (artifactsToBuild.getPicStaticLibrary() != null) {
        filesBuilder.add(artifactsToBuild.getPicStaticLibrary());
      }
      if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        if (artifactsToBuild.getResolvedSymlinkDynamicLibrary() != null) {
          filesBuilder.add(artifactsToBuild.getResolvedSymlinkDynamicLibrary());
        } else if (artifactsToBuild.getDynamicLibrary() != null) {
          filesBuilder.add(artifactsToBuild.getDynamicLibrary());
        }
        if (artifactsToBuild.getResolvedSymlinkInterfaceLibrary() != null) {
          filesBuilder.add(artifactsToBuild.getResolvedSymlinkInterfaceLibrary());
        } else if (artifactsToBuild.getInterfaceLibrary() != null) {
          filesBuilder.add(artifactsToBuild.getInterfaceLibrary());
        }
      }
    }

    if (!featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
      warnAboutEmptyLibraries(ruleContext, ccCompilationOutputs, linkStatic);
    }
    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    List<Artifact> instrumentedObjectFiles = new ArrayList<>();
    instrumentedObjectFiles.addAll(compilationInfo.getCcCompilationOutputs().getObjectFiles(false));
    instrumentedObjectFiles.addAll(compilationInfo.getCcCompilationOutputs().getObjectFiles(true));
    InstrumentedFilesInfo instrumentedFilesProvider =
        common.getInstrumentedFilesProvider(
            instrumentedObjectFiles,
            /* withBaselineCoverage= */ true,
            /* virtualToOriginalHeaders= */ NestedSetBuilder.create(Order.STABLE_ORDER));
    CppHelper.maybeAddStaticLinkMarkerProvider(targetBuilder, ruleContext);

    Runfiles.Builder builder = new Runfiles.Builder(ruleContext.getWorkspaceName());
    builder.addDataDeps(ruleContext);
    builder.add(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    if (addDynamicRuntimeInputArtifactsToRunfiles) {
      try {
        builder.addTransitiveArtifacts(
            ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration));
      } catch (EvalException e) {
        throw ruleContext.throwWithRuleError(e.getMessage());
      }
    }
    Runfiles runfiles = builder.build();
    Runfiles.Builder defaultRunfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .merge(runfiles)
            .addArtifacts(LibraryToLink.getDynamicLibrariesForRuntime(!neverLink, libraryToLinks));

    Runfiles.Builder dataRunfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .merge(runfiles)
            .addArtifacts(
                LibraryToLink.getDynamicLibrariesForRuntime(
                    /* linkingStatically= */ false, libraryToLinks));

    Map<String, NestedSet<Artifact>> currentOutputGroups =
        CcCompilationHelper.buildOutputGroupsForEmittingCompileProviders(
            compilationInfo.getCcCompilationOutputs(),
            compilationInfo.getCcCompilationContext(),
            ruleContext.getFragment(CppConfiguration.class),
            ccToolchain,
            featureConfiguration,
            ruleContext);
    CcSkylarkApiProvider.maybeAdd(ruleContext, targetBuilder);
    targetBuilder
        .setFilesToBuild(filesToBuild)
        .addProvider(ccNativeLibraryProvider)
        .addNativeDeclaredProvider(
            CcInfo.builder()
                .setCcCompilationContext(compilationInfo.getCcCompilationContext())
                .setCcLinkingContext(ccLinkingContext)
                .setCcDebugInfoContext(
                    CppHelper.mergeCcDebugInfoContexts(
                        compilationInfo.getCcCompilationOutputs(), ccInfosFromDeps))
                .build())
        .addOutputGroups(
            CcCommon.mergeOutputGroups(ImmutableList.of(currentOutputGroups, outputGroups.build())))
        .addNativeDeclaredProvider(instrumentedFilesProvider)
        .addProvider(RunfilesProvider.withData(defaultRunfiles.build(), dataRunfiles.build()))
        .addOutputGroup(
            OutputGroupInfo.HIDDEN_TOP_LEVEL,
            collectHiddenTopLevelArtifacts(
                ruleContext, ccToolchain, ccCompilationOutputs, featureConfiguration))
        .addOutputGroup(
            CcCompilationHelper.HIDDEN_HEADER_TOKENS,
            CcCompilationHelper.collectHeaderTokens(
                ruleContext,
                ruleContext.getFragment(CppConfiguration.class),
                ccCompilationOutputs));
  }

  private static NestedSet<Artifact> collectHiddenTopLevelArtifacts(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      CcCompilationOutputs ccCompilationOutputs,
      FeatureConfiguration featureConfiguration) {
    // Ensure that we build all the dependencies, otherwise users may get confused.
    NestedSetBuilder<Artifact> artifactsToForceBuilder = NestedSetBuilder.stableOrder();
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
    boolean usePic = toolchain.usePicForDynamicLibraries(cppConfiguration, featureConfiguration);
    artifactsToForceBuilder.addTransitive(
        ccCompilationOutputs.getFilesToCompile(processHeadersInDependencies, usePic));
    for (OutputGroupInfo dep :
        ruleContext.getPrerequisites(
            "deps", Mode.TARGET, OutputGroupInfo.SKYLARK_CONSTRUCTOR)) {
      artifactsToForceBuilder.addTransitive(
          dep.getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL));
    }
    return artifactsToForceBuilder.build();
  }

  private static void warnAboutEmptyLibraries(RuleContext ruleContext,
      CcCompilationOutputs ccCompilationOutputs,
      boolean linkstaticAttribute) {
    if (ccCompilationOutputs.getObjectFiles(false).isEmpty()
        && ccCompilationOutputs.getObjectFiles(true).isEmpty()) {
      if (!linkstaticAttribute && appearsToHaveObjectFiles(ruleContext.attributes())) {
        ruleContext.attributeWarning("linkstatic",
            "setting 'linkstatic=1' is recommended if there are no object files");
      }
    } else {
      if (!linkstaticAttribute && !appearsToHaveObjectFiles(ruleContext.attributes())) {
        Artifact element = Iterables.getFirst(
            ccCompilationOutputs.getObjectFiles(false),
            ccCompilationOutputs.getObjectFiles(true).get(0));
        ruleContext.attributeWarning("srcs",
             "this library appears at first glance to have no object files, "
             + "but on closer inspection it does have something to link, e.g. "
             + element.prettyPrint() + ". "
             + "(You may have used some very confusing rule names in srcs? "
             + "Or the library consists entirely of a linker script?) "
             + "Bazel assumed linkstatic=1, but this may be inappropriate. "
             + "You may need to add an explicit '.cc' file to 'srcs'. "
             + "Alternatively, add 'linkstatic=1' to suppress this warning");
      }
    }
  }

  /**
   * Returns true if the rule (which must be a cc_library rule) appears to have object files.
   * This only looks at the rule itself, not at any other rules (from this package or other
   * packages) that it might reference.
   *
   * <p>In some cases, this may return "true" even though the rule actually has no object files.
   * For example, it will return true for a rule such as
   * <code>cc_library(name = 'foo', srcs = [':bar'])</code> because we can't tell what ':bar' is;
   * it might be a genrule that generates a source file, or it might be a genrule that generates a
   * header file. Likewise,
   * <code>cc_library(name = 'foo', srcs = select({':a': ['foo.cc'], ':b': []}))</code> returns
   * "true" even though the sources *may* be empty. This reflects the fact that there's no way
   * to tell which value "srcs" will take without knowing the rule's configuration.
   *
   * <p>In other cases, this may return "false" even though the rule actually does have object
   * files. For example, it will return false for a rule such as
   * <code>cc_library(name = 'foo', srcs = ['bar.h'])</code> but as in the other example above,
   * we can't tell whether 'bar.h' is a file name or a rule name, and 'bar.h' could in fact be the
   * name of a genrule that generates a source file.
   */
  public static boolean appearsToHaveObjectFiles(AttributeMap rule) {
    if ((rule instanceof RawAttributeMapper) && rule.isConfigurable("srcs")) {
      // Since this method gets called by loading phase logic (e.g. the cc_library implicit outputs
      // function), the attribute mapper may not be able to resolve configurable attributes. When
      // that's the case, there's no way to know which value a configurable "srcs" will take, so
      // we conservatively assume object files are possible.
      return true;
    }

    List<Label> srcs = rule.get("srcs", BuildType.LABEL_LIST);
    if (srcs != null) {
      for (Label srcfile : srcs) {
        /*
         * We cheat a little bit here by looking at the file extension
         * of the Label treated as file name.  In general that might
         * not necessarily work, because of the possibility that the
         * user might give a rule a funky name ending in one of these
         * extensions, e.g.
         *    genrule(name = 'foo.h', outs = ['foo.cc'], ...) // Funky rule name!
         *    cc_library(name = 'bar', srcs = ['foo.h']) // This DOES have object files.
         */
        if (!NO_OBJECT_GENERATING_FILETYPES.matches(srcfile.getName())) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Adds linker output artifacts to the given map, to be registered on the configured target as
   * output groups.
   */
  private static Map<String, NestedSet<Artifact>> addLinkerOutputArtifacts(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      CppConfiguration cppConfiguration,
      BuildConfiguration configuration,
      CcCompilationOutputs ccCompilationOutputs,
      FeatureConfiguration featureConfiguration)
      throws RuleErrorException {

    NestedSetBuilder<Artifact> archiveFile = new NestedSetBuilder<>(Order.STABLE_ORDER);
    NestedSetBuilder<Artifact> dynamicLibrary = new NestedSetBuilder<>(Order.STABLE_ORDER);

    ImmutableSortedMap.Builder<String, NestedSet<Artifact>> outputGroups =
        ImmutableSortedMap.naturalOrder();
    if (!ruleContext.attributes().has("alwayslink", Type.BOOLEAN)
        || !ruleContext.attributes().has("linkstatic", Type.BOOLEAN)) {
      return outputGroups.build();
    }

    if (ruleContext.attributes().get("alwayslink", Type.BOOLEAN)) {
      archiveFile.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              configuration,
              Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY,
              /* linkedArtifactNameSuffix= */ ""));
    } else {
      archiveFile.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              configuration,
              Link.LinkTargetType.STATIC_LIBRARY,
              /* linkedArtifactNameSuffix= */ ""));
    }

    if (!ruleContext.attributes().get("linkstatic", Type.BOOLEAN)
        && !ccCompilationOutputs.isEmpty()) {
      dynamicLibrary.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              configuration,
              Link.LinkTargetType.NODEPS_DYNAMIC_LIBRARY,
              /* linkedArtifactNameSuffix= */ ""));

      if (CppHelper.useInterfaceSharedLibraries(
          cppConfiguration, ccToolchain, featureConfiguration)) {
        dynamicLibrary.add(
            CppHelper.getLinkedArtifact(
                ruleContext,
                ccToolchain,
                configuration,
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
                /* linkedArtifactNameSuffix= */ ""));
      }
    }

    outputGroups.put(ARCHIVE_LIBRARY_OUTPUT_GROUP_NAME, archiveFile.build());
    outputGroups.put(DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME, dynamicLibrary.build());
    return outputGroups.build();
  }

  private static ImmutableList<LibraryToLink> createLibrariesToLinkList(
      @Nullable LibraryToLink outputLibrary,
      List<LibraryToLink> precompiledLibraries,
      boolean ccCompilationOutputsIsEmpty) {
    ImmutableList.Builder<LibraryToLink> librariesToLink = ImmutableList.builder();
    librariesToLink.addAll(precompiledLibraries);

    // For cc_library if it contains precompiled libraries we link them. If it contains normal
    // sources we link them as well, if it doesn't contain normal sources, then we don't do
    // anything else if there were  precompiled libraries. However, if there are no precompiled
    // libraries and there are no normal sources, then we use the implicitly created link output
    // files if they exist.
    if (!ccCompilationOutputsIsEmpty
        || (precompiledLibraries.isEmpty()
            && isContentsOfCcLinkingOutputsImplicitlyCreated(
                ccCompilationOutputsIsEmpty, outputLibrary == null))) {
      if (outputLibrary != null) {
        librariesToLink.add(outputLibrary);
      }
    }

    return librariesToLink.build();
  }

  private static boolean isContentsOfCcLinkingOutputsImplicitlyCreated(
      boolean ccCompilationOutputsIsEmpty, boolean ccLinkingOutputsIsEmpty) {
    return ccCompilationOutputsIsEmpty && !ccLinkingOutputsIsEmpty;
  }

  private static Map<String, Artifact> buildMapIdentifierToArtifact(Iterable<Artifact> artifacts) {
    ImmutableMap.Builder<String, Artifact> libraries = ImmutableMap.builder();
    for (Artifact artifact : artifacts) {
      libraries.put(CcLinkingOutputs.libraryIdentifierOf(artifact), artifact);
    }
    return libraries.build();
  }

  /*
   * Add the libraries from srcs, if any. For static/mostly static
   * linking we setup the dynamic libraries if there are no static libraries
   * to choose from. Path to the libraries will be mangled to avoid using
   * absolute path names on the -rpath, but library filenames will be
   * preserved (since some libraries might have SONAME tag) - symlink will
   * be created to the parent directory instead.
   *
   * For compatibility with existing BUILD files, any ".a" or ".lo" files listed in
   * srcs are assumed to be position-independent code, or at least suitable for
   * inclusion in shared libraries, unless they end with ".nopic.a" or ".nopic.lo".
   *
   * Note that some target platforms do not require shared library code to be PIC.
   */
  private static List<LibraryToLink> convertPrecompiledLibrariesToLibraryToLink(
      CcCommon common, boolean forcePic, PrecompiledFiles precompiledFiles) {
    ImmutableList.Builder<LibraryToLink> librariesToLink = ImmutableList.builder();

    Map<String, Artifact> staticLibraries =
        buildMapIdentifierToArtifact(precompiledFiles.getStaticLibraries());
    Map<String, Artifact> picStaticLibraries =
        buildMapIdentifierToArtifact(precompiledFiles.getPicStaticLibraries());
    Map<String, Artifact> alwayslinkStaticLibraries =
        buildMapIdentifierToArtifact(precompiledFiles.getAlwayslinkStaticLibraries());
    Map<String, Artifact> alwayslinkPicStaticLibraries =
        buildMapIdentifierToArtifact(precompiledFiles.getPicAlwayslinkLibraries());
    Map<String, Artifact> dynamicLibraries =
        buildMapIdentifierToArtifact(precompiledFiles.getSharedLibraries());

    Set<String> identifiersUsed = new HashSet<>();
    for (Map.Entry<String, Artifact> staticLibraryEntry :
        Iterables.concat(staticLibraries.entrySet(), alwayslinkStaticLibraries.entrySet())) {
      LibraryToLink.Builder libraryToLinkBuilder = LibraryToLink.builder();
      String identifier = staticLibraryEntry.getKey();
      libraryToLinkBuilder.setLibraryIdentifier(identifier);
      boolean hasPic = picStaticLibraries.containsKey(identifier);
      boolean hasAlwaysPic = alwayslinkPicStaticLibraries.containsKey(identifier);
      if (hasPic || hasAlwaysPic) {
        Artifact picStaticLibrary = null;
        if (hasPic) {
          picStaticLibrary = picStaticLibraries.get(identifier);
        } else {
          picStaticLibrary = alwayslinkPicStaticLibraries.get(identifier);
        }
        libraryToLinkBuilder.setPicStaticLibrary(picStaticLibrary);
      }
      if (!forcePic || !(hasPic || hasAlwaysPic)) {
        libraryToLinkBuilder.setStaticLibrary(staticLibraryEntry.getValue());
      }
      if (dynamicLibraries.containsKey(identifier)) {
        Artifact library = dynamicLibraries.get(identifier);
        Artifact symlink = common.getDynamicLibrarySymlink(library, true);
        libraryToLinkBuilder.setDynamicLibrary(symlink);
        libraryToLinkBuilder.setResolvedSymlinkDynamicLibrary(library);
      }
      libraryToLinkBuilder.setAlwayslink(alwayslinkStaticLibraries.containsKey(identifier));
      identifiersUsed.add(identifier);
      librariesToLink.add(libraryToLinkBuilder.build());
    }

    for (Map.Entry<String, Artifact> picStaticLibraryEntry :
        Iterables.concat(picStaticLibraries.entrySet(), alwayslinkPicStaticLibraries.entrySet())) {
      String identifier = picStaticLibraryEntry.getKey();
      if (identifiersUsed.contains(identifier)) {
        continue;
      }
      LibraryToLink.Builder libraryToLinkBuilder = LibraryToLink.builder();
      libraryToLinkBuilder.setLibraryIdentifier(identifier);
      libraryToLinkBuilder.setPicStaticLibrary(picStaticLibraryEntry.getValue());
      if (dynamicLibraries.containsKey(identifier)) {
        Artifact library = dynamicLibraries.get(identifier);
        Artifact symlink = common.getDynamicLibrarySymlink(library, true);
        libraryToLinkBuilder.setDynamicLibrary(symlink);
        libraryToLinkBuilder.setResolvedSymlinkDynamicLibrary(library);
      }
      libraryToLinkBuilder.setAlwayslink(alwayslinkPicStaticLibraries.containsKey(identifier));
      identifiersUsed.add(identifier);
      librariesToLink.add(libraryToLinkBuilder.build());
    }

    for (Map.Entry<String, Artifact> dynamicLibraryEntry : dynamicLibraries.entrySet()) {
      String identifier = dynamicLibraryEntry.getKey();
      if (identifiersUsed.contains(identifier)) {
        continue;
      }
      LibraryToLink.Builder libraryToLinkBuilder = LibraryToLink.builder();
      libraryToLinkBuilder.setLibraryIdentifier(identifier);
      Artifact library = dynamicLibraryEntry.getValue();
      Artifact symlink = common.getDynamicLibrarySymlink(library, true);
      libraryToLinkBuilder.setDynamicLibrary(symlink);
      libraryToLinkBuilder.setResolvedSymlinkDynamicLibrary(library);
      librariesToLink.add(libraryToLinkBuilder.build());
    }
    return librariesToLink.build();
  }



  private static void checkIfLinkOutputsCollidingWithPrecompiledFiles(
      RuleContext ruleContext,
      CcLinkingOutputs ccLinkingOutputs,
      List<LibraryToLink> precompiledLibraries) {
    String identifier = ccLinkingOutputs.getLibraryToLink().getLibraryIdentifier();
    for (LibraryToLink precompiledLibrary : precompiledLibraries) {
      if (identifier.equals(precompiledLibrary.getLibraryIdentifier())) {
        ruleContext.ruleError(
            "Can't put library with identifier '"
                + precompiledLibrary.getLibraryIdentifier()
                + "' into the srcs of a "
                + ruleContext.getRuleClassNameForLogging()
                + " with the same name ("
                + ruleContext.getRule().getName()
                + ") which also contains other code or objects to link");
      }
    }
  }
}
