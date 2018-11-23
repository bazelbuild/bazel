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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.MapBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
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
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.SolibLibraryToLink;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A ConfiguredTarget for <code>cc_library</code> rules.
 */
public abstract class CcLibrary implements RuleConfiguredTargetFactory {

  /** A string constant for the name of archive library(.a, .lo) output group. */
  public static final String ARCHIVE_LIBRARY_OUTPUT_GROUP_NAME = "archive";

  /** A string constant for the name of dynamic library output group. */
  public static final String DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME = "dynamic_library";

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
    CppHelper.checkAllowedDeps(ruleContext);

    final CcCommon common = new CcCommon(ruleContext);

    CcToolchainProvider ccToolchain = common.getToolchain();

      ImmutableMap.Builder<String, String> toolchainMakeVariables = ImmutableMap.builder();
      ccToolchain.addGlobalMakeVariables(toolchainMakeVariables);
      ruleContext.initConfigurationMakeVariableContext(
          new MapBackedMakeVariableSupplier(toolchainMakeVariables.build()),
          new CcFlagsSupplier(ruleContext));

    FdoProvider fdoProvider = common.getFdoProvider();
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, ccToolchain);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    semantics.validateAttributes(ruleContext);
    if (ruleContext.hasErrors()) {
      return;
    }

    CcCompilationHelper compilationHelper =
        new CcCompilationHelper(
                ruleContext, semantics, featureConfiguration, ccToolchain, fdoProvider)
            .fromCommon(common, additionalCopts)
            .addSources(common.getSources())
            .addPrivateHeaders(common.getPrivateHeaders())
            .addPublicHeaders(common.getHeaders())
            .enableCompileProviders()
            .addPrecompiledFiles(precompiledFiles);

    CcLinkingHelper linkingHelper =
        new CcLinkingHelper(
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                fdoProvider,
                ruleContext.getConfiguration())
            .fromCommon(common)
            .addLinkopts(common.getLinkopts())
            .emitInterfaceSharedObjects(true)
            .setAlwayslink(alwaysLink)
            .setNeverLink(neverLink)
            .addLinkstamps(ruleContext.getPrerequisites("linkstamp", Mode.TARGET));

    Artifact soImplArtifact = null;
    boolean supportsDynamicLinker = ccToolchain.supportsDynamicLinker();
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

    if (ruleContext.getRule().isAttrDefined("srcs", BuildType.LABEL_LIST)) {
      ruleContext.checkSrcsSamePackage(true);
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
      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)) {
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
      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)) {
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
    CcCompilationOutputs ccCompilationOutputs = compilationInfo.getCcCompilationOutputs();
    // Generate .a and .so outputs even without object files to fulfill the rule class
    // contract wrt. implicit output files, if the contract says so. Behavior here differs
    // between Bazel and Blaze.
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (ruleContext.getRule().getImplicitOutputsFunction() != ImplicitOutputsFunction.NONE
        || !ccCompilationOutputs.isEmpty()) {
      if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        // If windows_export_all_symbols feature is enabled, bazel parses object files to generate
        // DEF file and use it to export symbols. The generated DEF file won't be used if a custom
        // DEF file is specified by win_def_file attribute.
        if (CppHelper.shouldUseGeneratedDefFile(ruleContext, featureConfiguration)) {
          try {
            Artifact generatedDefFile =
                CppHelper.createDefFileActions(
                    ruleContext,
                    ruleContext.getPrerequisiteArtifact("$def_parser", Mode.HOST),
                    ccCompilationOutputs.getObjectFiles(false),
                    ccToolchain
                        .getFeatures()
                        .getArtifactNameForCategory(
                            ArtifactCategory.DYNAMIC_LIBRARY, ruleContext.getLabel().getName()));
            linkingHelper.setDefFile(generatedDefFile);
          } catch (EvalException e) {
            ruleContext.throwWithRuleError(e.getMessage());
            throw new IllegalStateException("Should not be reached");
          }
        }

        // If user specifies a custom DEF file, then we use this one instead of the generated one.
        Artifact customDefFile = null;
        if (ruleContext.isAttrDefined("win_def_file", LABEL)) {
          customDefFile = ruleContext.getPrerequisiteArtifact("win_def_file", Mode.TARGET);
          if (customDefFile != null) {
            linkingHelper.setDefFile(customDefFile);
          }
        }
      }
      ccLinkingOutputs = linkingHelper.link(ccCompilationOutputs);
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
    ImmutableList<LibraryToLink> precompiledStaticLibraries =
        ImmutableList.<LibraryToLink>builder()
            .addAll(
                LinkerInputs.opaqueLibrariesToLink(
                    ArtifactCategory.STATIC_LIBRARY, precompiledFiles.getStaticLibraries()))
            .addAll(
                LinkerInputs.opaqueLibrariesToLink(
                    ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
                    precompiledFiles.getAlwayslinkStaticLibraries()))
            .build();

    ImmutableList<LibraryToLink> precompiledPicStaticLibraries =
        ImmutableList.<LibraryToLink>builder()
            .addAll(
                LinkerInputs.opaqueLibrariesToLink(
                    ArtifactCategory.STATIC_LIBRARY, precompiledFiles.getPicStaticLibraries()))
            .addAll(
                LinkerInputs.opaqueLibrariesToLink(
                    ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
                    precompiledFiles.getPicAlwayslinkLibraries()))
            .build();

    List<LibraryToLink> dynamicLibraries =
        ImmutableList.copyOf(
            Iterables.transform(
                precompiledFiles.getSharedLibraries(),
                library ->
                    LinkerInputs.solibLibraryToLink(
                        common.getDynamicLibrarySymlink(library, true),
                        library,
                        CcLinkingOutputs.libraryIdentifierOf(library))));

    ImmutableSortedMap.Builder<String, NestedSet<Artifact>> outputGroups =
        ImmutableSortedMap.naturalOrder();
    if (!ccLinkingOutputs.isEmpty()) {
      outputGroups.putAll(
          addLinkerOutputArtifacts(
              ruleContext, ccToolchain, ruleContext.getConfiguration(), ccCompilationOutputs));
    }
    CcLinkingOutputs ccLinkingOutputsWithPrecompiledLibraries =
        addPrecompiledLibrariesToLinkingOutputs(
            ruleContext,
            ccLinkingOutputs,
            precompiledStaticLibraries,
            precompiledPicStaticLibraries,
            dynamicLibraries,
            ccCompilationOutputs);

    ImmutableList<LibraryToLinkWrapper> libraryToLinkWrappers = ImmutableList.of();
    if (!neverLink) {
      libraryToLinkWrappers =
          createLibraryToLinkWrappersList(
              ruleContext,
              ccLinkingOutputs,
              precompiledStaticLibraries,
              precompiledPicStaticLibraries,
              dynamicLibraries,
              ccCompilationOutputs);
    }

    CcLinkingInfo ccLinkingInfo =
        linkingHelper.buildCcLinkingInfoFromLibraryToLinkWrappers(
            libraryToLinkWrappers, compilationInfo.getCcCompilationContext());
    CcNativeLibraryProvider ccNativeLibraryProvider =
        CppHelper.collectNativeCcLibraries(
            ruleContext.getPrerequisites("deps", Mode.TARGET),
            ccLinkingOutputsWithPrecompiledLibraries);

    /*
     * We always generate a static library, even if there aren't any source files.
     * This keeps things simpler by avoiding special cases when making use of the library.
     * For example, this is needed to ensure that building a library with "bazel build"
     * will also build all of the library's "deps".
     * However, we only generate a dynamic library if there are source files.
     */
    // For now, we don't add the precompiled libraries to the files to build.

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(ccLinkingOutputs.getStaticLibraries()));
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(ccLinkingOutputs.getPicStaticLibraries()));

    if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
      filesBuilder.addAll(
          LinkerInputs.toNonSolibArtifacts(ccLinkingOutputs.getDynamicLibrariesForLinking()));
      filesBuilder.addAll(
          LinkerInputs.toNonSolibArtifacts(ccLinkingOutputs.getDynamicLibrariesForRuntime()));
    }

    if (!featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
      warnAboutEmptyLibraries(ruleContext, compilationInfo.getCcCompilationOutputs(), linkStatic);
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
      builder.addTransitiveArtifacts(ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration));
    }
    Runfiles runfiles = builder.build();
    Runfiles.Builder defaultRunfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .merge(runfiles)
            .addArtifacts(
                ccLinkingOutputsWithPrecompiledLibraries.getLibrariesForRunfiles(!neverLink));

    Runfiles.Builder dataRunfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .merge(runfiles)
            .addArtifacts(ccLinkingOutputsWithPrecompiledLibraries.getLibrariesForRunfiles(false));

    targetBuilder
        .setFilesToBuild(filesToBuild)
        .addProvider(compilationInfo.getCppDebugFileProvider())
        .addProvider(ccNativeLibraryProvider)
        .addNativeDeclaredProvider(
            CcInfo.builder()
                .setCcCompilationContext(compilationInfo.getCcCompilationContext())
                .setCcLinkingInfo(ccLinkingInfo)
                .build())
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(
            CcCommon.mergeOutputGroups(
                ImmutableList.of(compilationInfo.getOutputGroups(), outputGroups.build())))
        .addNativeDeclaredProvider(instrumentedFilesProvider)
        .addProvider(RunfilesProvider.withData(defaultRunfiles.build(), dataRunfiles.build()))
        .addOutputGroup(
            OutputGroupInfo.HIDDEN_TOP_LEVEL,
            collectHiddenTopLevelArtifacts(ruleContext, ccToolchain, ccCompilationOutputs))
        .addOutputGroup(
            CcCompilationHelper.HIDDEN_HEADER_TOKENS,
            CcCompilationHelper.collectHeaderTokens(ruleContext, ccCompilationOutputs));
  }

  private static NestedSet<Artifact> collectHiddenTopLevelArtifacts(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      CcCompilationOutputs ccCompilationOutputs) {
    // Ensure that we build all the dependencies, otherwise users may get confused.
    NestedSetBuilder<Artifact> artifactsToForceBuilder = NestedSetBuilder.stableOrder();
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
    boolean usePic = toolchain.usePicForDynamicLibraries();
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
      BuildConfiguration configuration,
      CcCompilationOutputs ccCompilationOutputs)
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

      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)) {
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

  private static ImmutableList<LibraryToLinkWrapper> createLibraryToLinkWrappersList(
      RuleContext ruleContext,
      CcLinkingOutputs ccLinkingOutputs,
      List<LibraryToLink> staticLibraries,
      List<LibraryToLink> picStaticLibraries,
      List<LibraryToLink> dynamicLibrariesForRuntime,
      CcCompilationOutputs ccCompilationOutputs) {
    Preconditions.checkState(ccLinkingOutputs.getStaticLibraries().size() <= 1);
    Preconditions.checkState(ccLinkingOutputs.getPicStaticLibraries().size() <= 1);
    Preconditions.checkState(ccLinkingOutputs.getDynamicLibrariesForLinking().size() <= 1);
    Preconditions.checkState(ccLinkingOutputs.getDynamicLibrariesForRuntime().size() <= 1);

    ImmutableList.Builder<LibraryToLinkWrapper> libraryToLinkWrappers = ImmutableList.builder();

    checkIfLinkOutputsCollidingWithPrecompiledFiles(
        ruleContext,
        ccLinkingOutputs,
        staticLibraries,
        picStaticLibraries,
        dynamicLibrariesForRuntime,
        ccCompilationOutputs);

    if (ruleContext.hasErrors()) {
      return libraryToLinkWrappers.build();
    }

    // For cc_library if it contains precompiled libraries we link them. If it contains normal
    // sources we link them as well, if it doesn't contain normal sources, then we don't do
    // anything else if there were  precompiled libraries. However, if there are no precompiled
    // libraries and there are no normal sources, then we use the implicitly created link output
    // files if they exist.
    libraryToLinkWrappers.addAll(
        convertPrecompiledLibrariesToLibraryToLinkWrapper(
            staticLibraries, picStaticLibraries, dynamicLibrariesForRuntime));
    if (!ccCompilationOutputs.isEmpty()
        || (staticLibraries.isEmpty()
            && picStaticLibraries.isEmpty()
            && dynamicLibrariesForRuntime.isEmpty()
            && isContentsOfCcLinkingOutputsImplicitlyCreated(
                ccCompilationOutputs, ccLinkingOutputs))) {
      LibraryToLinkWrapper linkOutputsLibraryToLinkWrapper =
          convertLinkOutputsToLibraryToLinkWrapper(ccLinkingOutputs);
      if (linkOutputsLibraryToLinkWrapper != null) {
        libraryToLinkWrappers.add(linkOutputsLibraryToLinkWrapper);
      }
    }

    return libraryToLinkWrappers.build();
  }

  private static boolean isContentsOfCcLinkingOutputsImplicitlyCreated(
      CcCompilationOutputs ccCompilationOutputs, CcLinkingOutputs ccLinkingOutputs) {
    return ccCompilationOutputs.isEmpty() && !ccLinkingOutputs.isEmpty();
  }

  private static List<LibraryToLinkWrapper> convertPrecompiledLibrariesToLibraryToLinkWrapper(
      List<LibraryToLink> staticLibraries,
      List<LibraryToLink> picStaticLibraries,
      List<LibraryToLink> dynamicLibrariesForRuntime) {
    ImmutableList.Builder<LibraryToLinkWrapper> libraryToLinkWrappers = ImmutableList.builder();

    Set<String> identifiersUsed = new HashSet<>();
    // Here we hae an O(n^2) algorithm, the size of the inputs is never big though, we only work
    // here with the local libraries, none of the libraries of the transitive closure.
    for (LibraryToLink staticLibrary : staticLibraries) {
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder = LibraryToLinkWrapper.builder();
      libraryToLinkWrapperBuilder.setStaticLibrary(staticLibrary.getArtifact());
      String identifier = staticLibrary.getLibraryIdentifier();
      libraryToLinkWrapperBuilder.setLibraryIdentifier(identifier);
      List<LibraryToLink> sameIdentifierPicStaticLibraries =
          picStaticLibraries.stream()
              .filter(x -> x.getLibraryIdentifier().equals(identifier))
              .collect(ImmutableList.toImmutableList());
      if (!sameIdentifierPicStaticLibraries.isEmpty()) {
        libraryToLinkWrapperBuilder.setPicStaticLibrary(
            sameIdentifierPicStaticLibraries.get(0).getArtifact());
      }
      List<LibraryToLink> sameIdentifierDynamicLibraries =
          dynamicLibrariesForRuntime.stream()
              .filter(x -> x.getLibraryIdentifier().equals(identifier))
              .collect(ImmutableList.toImmutableList());
      if (!sameIdentifierDynamicLibraries.isEmpty()) {
        LibraryToLink dynamicLibrary = sameIdentifierDynamicLibraries.get(0);
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibrary.getArtifact());
        if (dynamicLibrary instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibrary.getOriginalLibraryArtifact());
        }
      }
      libraryToLinkWrapperBuilder.setAlwayslink(
          staticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      identifiersUsed.add(identifier);
      libraryToLinkWrappers.add(libraryToLinkWrapperBuilder.build());
    }

    for (LibraryToLink picStaticLibrary : picStaticLibraries) {
      String identifier = picStaticLibrary.getLibraryIdentifier();
      if (identifiersUsed.contains(identifier)) {
        continue;
      }
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder = LibraryToLinkWrapper.builder();
      libraryToLinkWrapperBuilder.setPicStaticLibrary(picStaticLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setLibraryIdentifier(identifier);
      List<LibraryToLink> sameIdentifierDynamicLibraries =
          dynamicLibrariesForRuntime.stream()
              .filter(x -> x.getLibraryIdentifier().equals(identifier))
              .collect(ImmutableList.toImmutableList());
      if (!sameIdentifierDynamicLibraries.isEmpty()) {
        LibraryToLink dynamicLibrary = sameIdentifierDynamicLibraries.get(0);
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibrary.getArtifact());
        if (dynamicLibrary instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibrary.getOriginalLibraryArtifact());
        }
      }
      libraryToLinkWrapperBuilder.setAlwayslink(
          picStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      identifiersUsed.add(identifier);
      libraryToLinkWrappers.add(libraryToLinkWrapperBuilder.build());
    }

    for (LibraryToLink dynamicLibrary : dynamicLibrariesForRuntime) {
      String identifier = dynamicLibrary.getLibraryIdentifier();
      if (identifiersUsed.contains(identifier)) {
        continue;
      }
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder = LibraryToLinkWrapper.builder();
      libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setLibraryIdentifier(identifier);
      if (dynamicLibrary instanceof SolibLibraryToLink) {
        libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
            dynamicLibrary.getOriginalLibraryArtifact());
      }
      libraryToLinkWrappers.add(libraryToLinkWrapperBuilder.build());
    }
    return libraryToLinkWrappers.build();
  }

  private static LibraryToLinkWrapper convertLinkOutputsToLibraryToLinkWrapper(
      CcLinkingOutputs ccLinkingOutputs) {
    Preconditions.checkState(!ccLinkingOutputs.isEmpty());

    LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder = LibraryToLinkWrapper.builder();
    if (!ccLinkingOutputs.getStaticLibraries().isEmpty()) {
      LibraryToLink staticLibrary = ccLinkingOutputs.getStaticLibraries().get(0);
      libraryToLinkWrapperBuilder.setStaticLibrary(staticLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setObjectFiles(
          ImmutableList.copyOf(staticLibrary.getObjectFiles()));
      libraryToLinkWrapperBuilder.setLtoBitcodeFiles(
          ImmutableMap.copyOf(staticLibrary.getLtoBitcodeFiles()));
      libraryToLinkWrapperBuilder.setSharedNonLtoBackends(
          ImmutableMap.copyOf(staticLibrary.getSharedNonLtoBackends()));
      libraryToLinkWrapperBuilder.setAlwayslink(
          staticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      libraryToLinkWrapperBuilder.setLibraryIdentifier(staticLibrary.getLibraryIdentifier());
    }

    if (!ccLinkingOutputs.getPicStaticLibraries().isEmpty()) {
      LibraryToLink picStaticLibrary = ccLinkingOutputs.getPicStaticLibraries().get(0);
      libraryToLinkWrapperBuilder.setPicStaticLibrary(picStaticLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setPicObjectFiles(
          ImmutableList.copyOf(picStaticLibrary.getObjectFiles()));
      libraryToLinkWrapperBuilder.setPicLtoBitcodeFiles(
          ImmutableMap.copyOf(picStaticLibrary.getLtoBitcodeFiles()));
      libraryToLinkWrapperBuilder.setPicSharedNonLtoBackends(
          ImmutableMap.copyOf(picStaticLibrary.getSharedNonLtoBackends()));
      libraryToLinkWrapperBuilder.setAlwayslink(
          picStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      libraryToLinkWrapperBuilder.setLibraryIdentifier(picStaticLibrary.getLibraryIdentifier());
    }

    if (!ccLinkingOutputs.getDynamicLibrariesForLinking().isEmpty()) {
      LibraryToLink dynamicLibraryForLinking =
          ccLinkingOutputs.getDynamicLibrariesForLinking().get(0);
      Preconditions.checkState(!ccLinkingOutputs.getDynamicLibrariesForRuntime().isEmpty());
      LibraryToLink dynamicLibraryForRuntime =
          ccLinkingOutputs.getDynamicLibrariesForRuntime().get(0);
      if (dynamicLibraryForLinking != dynamicLibraryForRuntime) {
        libraryToLinkWrapperBuilder.setInterfaceLibrary(dynamicLibraryForLinking.getArtifact());
        if (dynamicLibraryForLinking instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkInterfaceLibrary(
              dynamicLibraryForLinking.getOriginalLibraryArtifact());
        }
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibraryForRuntime.getArtifact());
        if (dynamicLibraryForRuntime instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibraryForRuntime.getOriginalLibraryArtifact());
        }
      } else {
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibraryForRuntime.getArtifact());
        if (dynamicLibraryForRuntime instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibraryForRuntime.getOriginalLibraryArtifact());
        }
      }
      libraryToLinkWrapperBuilder.setLibraryIdentifier(
          dynamicLibraryForLinking.getLibraryIdentifier());
    }
    return libraryToLinkWrapperBuilder.build();
  }

  private static CcLinkingOutputs addPrecompiledLibrariesToLinkingOutputs(
      RuleContext ruleContext,
      CcLinkingOutputs ccLinkingOutputs,
      List<LibraryToLink> staticLibraries,
      List<LibraryToLink> picStaticLibraries,
      List<LibraryToLink> dynamicLibrariesForRuntime,
      CcCompilationOutputs ccCompilationOutputs) {
    if (staticLibraries.isEmpty()
        && picStaticLibraries.isEmpty()
        && dynamicLibrariesForRuntime.isEmpty()) {
      return ccLinkingOutputs;
    }

    CcLinkingOutputs.Builder newOutputsBuilder = new CcLinkingOutputs.Builder();
    if (!ccCompilationOutputs.isEmpty()) {
      newOutputsBuilder.merge(ccLinkingOutputs);
    }

    checkIfLinkOutputsCollidingWithPrecompiledFiles(
        ruleContext,
        ccLinkingOutputs,
        staticLibraries,
        picStaticLibraries,
        dynamicLibrariesForRuntime,
        ccCompilationOutputs);

    // Merge the pre-compiled libraries (static & dynamic) into the linker outputs.
    return newOutputsBuilder
        .addStaticLibraries(staticLibraries)
        .addPicStaticLibraries(picStaticLibraries)
        .addDynamicLibraries(dynamicLibrariesForRuntime)
        .addDynamicLibrariesForRuntime(dynamicLibrariesForRuntime)
        .build();
  }

  private static void checkIfLinkOutputsCollidingWithPrecompiledFiles(
      RuleContext ruleContext,
      CcLinkingOutputs ccLinkingOutputs,
      List<LibraryToLink> staticLibraries,
      List<LibraryToLink> picStaticLibraries,
      List<LibraryToLink> dynamicLibrariesForRuntime,
      CcCompilationOutputs ccCompilationOutputs) {
    if (!ccCompilationOutputs.isEmpty()) {
      ImmutableSetMultimap<String, LibraryToLink> precompiledLibraryMap =
          CcLinkingOutputs.getLibrariesByIdentifier(
              Iterables.concat(
                  staticLibraries,
                  picStaticLibraries,
                  dynamicLibrariesForRuntime));
      ImmutableSetMultimap<String, LibraryToLink> linkedLibraryMap =
          ccLinkingOutputs.getLibrariesByIdentifier();
      for (String matchingIdentifier :
          Sets.intersection(precompiledLibraryMap.keySet(), linkedLibraryMap.keySet())) {
        Iterable<Artifact> matchingInputLibs =
            LinkerInputs.toNonSolibArtifacts(precompiledLibraryMap.get(matchingIdentifier));
        Iterable<Artifact> matchingOutputLibs =
            LinkerInputs.toNonSolibArtifacts(linkedLibraryMap.get(matchingIdentifier));
        ruleContext.ruleError(
            "Can't put "
                + Streams.stream(matchingInputLibs)
                    .map(Artifact::getFilename)
                    .collect(joining(", "))
                + " into the srcs of a "
                + ruleContext.getRuleClassNameForLogging()
                + " with the same name ("
                + ruleContext.getRule().getName()
                + ") which also contains other code or objects to link; it shares a name with "
                + Streams.stream(matchingOutputLibs)
                    .map(Artifact::getFilename)
                    .collect(joining(", "))
                + " (output compiled and linked from the non-library sources of this rule), "
                + "which could cause confusion");
      }
    }
  }
}
