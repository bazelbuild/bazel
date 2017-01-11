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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.test.ExecutionInfoProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
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

  private static Runfiles collectRunfiles(
      RuleContext context,
      CcLinkingOutputs linkingOutputs,
      CcLibraryHelper.Info info,
      LinkStaticness linkStaticness,
      NestedSet<Artifact> filesToBuild,
      Iterable<Artifact> fakeLinkerInputs,
      boolean fake,
      ImmutableSet<CppSource> cAndCppSources,
      boolean linkCompileOutputSeparately) {
    Runfiles.Builder builder = new Runfiles.Builder(
        context.getWorkspaceName(), context.getConfiguration().legacyExternalRunfiles());
    Function<TransitiveInfoCollection, Runfiles> runfilesMapping =
        CppRunfilesProvider.runfilesFunction(linkStaticness != LinkStaticness.DYNAMIC);
    builder.addTransitiveArtifacts(filesToBuild);
    // Add the shared libraries to the runfiles. This adds any shared libraries that are in the
    // srcs of this target.
    builder.addArtifacts(linkingOutputs.getLibrariesForRunfiles(true));
    builder.addRunfiles(context, RunfilesProvider.DEFAULT_RUNFILES);
    builder.add(context, runfilesMapping);
    CcToolchainProvider toolchain = CppHelper.getToolchain(context);
    // Add the C++ runtime libraries if linking them dynamically.
    if (linkStaticness == LinkStaticness.DYNAMIC) {
      builder.addTransitiveArtifacts(toolchain.getDynamicRuntimeLinkInputs());
    }
    if (linkCompileOutputSeparately) {
      builder.addArtifacts(
          LinkerInputs.toLibraryArtifacts(
              info.getCcLinkingOutputs().getExecutionDynamicLibraries()));
    }
    // For cc_binary and cc_test rules, there is an implicit dependency on
    // the malloc library package, which is specified by the "malloc" attribute.
    // As the BUILD encyclopedia says, the "malloc" attribute should be ignored
    // if linkshared=1.
    boolean linkshared = isLinkShared(context);
    if (!linkshared) {
      TransitiveInfoCollection malloc = CppHelper.mallocForTarget(context);
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
      CppCompilationContext cppCompilationContext = info.getCppCompilationContext();
      builder.addSymlinksToArtifacts(cppCompilationContext.getDeclaredIncludeSrcs());
      // Add additional files that are referenced from the compile command, like module maps
      // or header modules.
      builder.addSymlinksToArtifacts(cppCompilationContext.getAdditionalInputs());
      builder.addSymlinksToArtifacts(
          cppCompilationContext.getTransitiveModules(
              CppHelper.usePic(context, !isLinkShared(context))));
    }
    return builder.build();
  }

  @Override
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException {
    return CcBinary.init(semantics, context, /*fake =*/ false);
  }

  public static ConfiguredTarget init(CppSemantics semantics, RuleContext ruleContext, boolean fake)
      throws InterruptedException, RuleErrorException {
    ruleContext.checkSrcsSamePackage(true);
    FeatureConfiguration featureConfiguration = CcCommon.configureFeatures(ruleContext);
    CcCommon common = new CcCommon(ruleContext);
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);
    LinkTargetType linkType =
        isLinkShared(ruleContext) ? LinkTargetType.DYNAMIC_LIBRARY : LinkTargetType.EXECUTABLE;

    semantics.validateAttributes(ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }
    
    List<String> linkopts = common.getLinkopts();
    LinkStaticness linkStaticness = getLinkStaticness(ruleContext, linkopts, cppConfiguration);

    // We currently only want link the dynamic library generated for test code separately.
    boolean linkCompileOutputSeparately =
        ruleContext.isTestTarget()
            && cppConfiguration.getLinkCompileOutputSeparately()
            && linkStaticness == LinkStaticness.DYNAMIC;

    CcLibraryHelper helper =
        new CcLibraryHelper(ruleContext, semantics, featureConfiguration)
            .fromCommon(common)
            .addSources(common.getSources())
            .addDeps(ImmutableList.of(CppHelper.mallocForTarget(ruleContext)))
            .setFake(fake)
            .addPrecompiledFiles(precompiledFiles)
            .enableInterfaceSharedObjects();
    // When linking the object files directly into the resulting binary, we do not need
    // library-level link outputs; thus, we do not let CcLibraryHelper produce link outputs
    // (either shared object files or archives) for a non-library link type [*], and add
    // the object files explicitly in determineLinkerArguments.
    //
    // When linking the object files into their own library, we want CcLibraryHelper to
    // take care of creating the library link outputs for us, so we need to set the link
    // type to STATIC_LIBRARY.
    //
    // [*] The only library link type is STATIC_LIBRARY. EXECUTABLE specifies a normal
    // cc_binary output, while DYNAMIC_LIBRARY is a cc_binary rules that produces an
    // output matching a shared object, for example cc_binary(name="foo.so", ...) on linux.
    helper.setLinkType(linkCompileOutputSeparately ? LinkTargetType.STATIC_LIBRARY : linkType);

    CcLibraryHelper.Info info = helper.build();
    CppCompilationContext cppCompilationContext = info.getCppCompilationContext();
    CcCompilationOutputs ccCompilationOutputs = info.getCcCompilationOutputs();

    // if cc_binary includes "linkshared=1", then gcc will be invoked with
    // linkopt "-shared", which causes the result of linking to be a shared
    // library. In this case, the name of the executable target should end
    // in ".so" or "dylib" or ".dll".
    PathFragment binaryPath = new PathFragment(ruleContext.getTarget().getName());
    if (!isLinkShared(ruleContext)) {
      binaryPath = new PathFragment(binaryPath.getPathString() + OsUtils.executableExtension());
    }

    Artifact binary = ruleContext.getBinArtifact(binaryPath);
    if (isLinkShared(ruleContext)
        && !CppFileTypes.SHARED_LIBRARY.matches(binary.getFilename())
        && !CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(binary.getFilename())) {
      ruleContext.attributeError("linkshared", "'linkshared' used in non-shared library");
      return null;
    }

    CppLinkActionBuilder linkActionBuilder =
        determineLinkerArguments(
            ruleContext,
            common,
            precompiledFiles,
            info,
            cppCompilationContext.getTransitiveCompilationPrerequisites(),
            fake,
            binary,
            linkStaticness,
            linkopts,
            linkCompileOutputSeparately);
    linkActionBuilder.setUseTestOnlyFlags(ruleContext.isTestTarget());

    CcToolchainProvider ccToolchain = CppHelper.getToolchain(ruleContext);
    if (linkStaticness == LinkStaticness.DYNAMIC) {
      linkActionBuilder.setRuntimeInputs(
          ArtifactCategory.DYNAMIC_LIBRARY,
          ccToolchain.getDynamicRuntimeLinkMiddleman(),
          ccToolchain.getDynamicRuntimeLinkInputs());
    } else {
      linkActionBuilder.setRuntimeInputs(
          ArtifactCategory.STATIC_LIBRARY,
          ccToolchain.getStaticRuntimeLinkMiddleman(),
          ccToolchain.getStaticRuntimeLinkInputs());
      // Only force a static link of libgcc if static runtime linking is enabled (which
      // can't be true if runtimeInputs is empty).
      // TODO(bazel-team): Move this to CcToolchain.
      if (!ccToolchain.getStaticRuntimeLinkInputs().isEmpty()) {
        linkActionBuilder.addLinkopt("-static-libgcc");
      }
    }

    linkActionBuilder.setLinkType(linkType);
    linkActionBuilder.setLinkStaticness(linkStaticness);
    linkActionBuilder.setFake(fake);
    linkActionBuilder.setFeatureConfiguration(featureConfiguration);

    if (CppLinkAction.enableSymbolsCounts(cppConfiguration, fake, linkType)) {
      linkActionBuilder.setSymbolCountsOutput(ruleContext.getBinArtifact(
          CppLinkAction.symbolCountsFileName(binaryPath)));
    }

    if (isLinkShared(ruleContext)) {
      linkActionBuilder.setLibraryIdentifier(CcLinkingOutputs.libraryIdentifierOf(binary));
    }

    // Store immutable context for use in other *_binary rules that are implemented by
    // linking the interpreter (Java, Python, etc.) together with native deps.
    CppLinkAction.Context linkContext = new CppLinkAction.Context(linkActionBuilder);

    if (featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)) {
      linkActionBuilder.setLTOIndexing(true);
      CppLinkAction indexAction = linkActionBuilder.build();
      ruleContext.registerAction(indexAction);

      for (LTOBackendArtifacts ltoArtifacts : indexAction.getAllLTOBackendArtifacts()) {
        boolean usePic = CppHelper.usePic(ruleContext, !isLinkShared(ruleContext));
        ltoArtifacts.scheduleLTOBackendAction(ruleContext, featureConfiguration, usePic);
      }

      linkActionBuilder.setLTOIndexing(false);
    }

    CppLinkAction linkAction = linkActionBuilder.build();
    ruleContext.registerAction(linkAction);
    LibraryToLink outputLibrary = linkAction.getOutputLibrary();
    Iterable<Artifact> fakeLinkerInputs =
        fake ? linkAction.getInputs() : ImmutableList.<Artifact>of();
    Artifact executable = linkAction.getLinkOutput();
    CcLinkingOutputs.Builder linkingOutputsBuilder = new CcLinkingOutputs.Builder();
    if (isLinkShared(ruleContext)) {
      linkingOutputsBuilder.addDynamicLibrary(outputLibrary);
      linkingOutputsBuilder.addExecutionDynamicLibrary(outputLibrary);
    }
    // Also add all shared libraries from srcs.
    for (Artifact library : precompiledFiles.getSharedLibraries()) {
      Artifact symlink = common.getDynamicLibrarySymlink(library, true);
      LibraryToLink symlinkLibrary = LinkerInputs.solibLibraryToLink(
          symlink, library, CcLinkingOutputs.libraryIdentifierOf(library));
      linkingOutputsBuilder.addDynamicLibrary(symlinkLibrary);
      linkingOutputsBuilder.addExecutionDynamicLibrary(symlinkLibrary);
    }
    CcLinkingOutputs linkingOutputs = linkingOutputsBuilder.build();
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER, executable);

    // Create the stripped binary, but don't add it to filesToBuild; it's only built when requested.
    Artifact strippedFile = ruleContext.getImplicitOutputArtifact(
        CppRuleClasses.CC_BINARY_STRIPPED);
    CppHelper.createStripAction(ruleContext, cppConfiguration, executable, strippedFile);

    DwoArtifactsCollector dwoArtifacts =
        collectTransitiveDwoArtifacts(ruleContext, ccCompilationOutputs, linkStaticness);
    Artifact dwpFile =
        ruleContext.getImplicitOutputArtifact(CppRuleClasses.CC_BINARY_DEBUG_PACKAGE);
    createDebugPackagerActions(ruleContext, cppConfiguration, dwpFile, dwoArtifacts);

    // The debug package should include the dwp file only if it was explicitly requested.
    Artifact explicitDwpFile = dwpFile;
    if (!cppConfiguration.useFission()) {
      explicitDwpFile = null;
    } else {
      // For cc_test rules, include the dwp in the runfiles if Fission is enabled and the test was
      // built statically.
      if (TargetUtils.isTestRule(ruleContext.getRule())
          && linkStaticness != LinkStaticness.DYNAMIC
          && cppConfiguration.shouldBuildTestDwp()) {
        filesToBuild = NestedSetBuilder.fromNestedSet(filesToBuild).add(dwpFile).build();
      }
    }

    // TODO(bazel-team): Do we need to put original shared libraries (along with
    // mangled symlinks) into the RunfilesSupport object? It does not seem
    // logical since all symlinked libraries will be linked anyway and would
    // not require manual loading but if we do, then we would need to collect
    // their names and use a different constructor below.
    Runfiles runfiles =
        collectRunfiles(
            ruleContext,
            linkingOutputs,
            info,
            linkStaticness,
            filesToBuild,
            fakeLinkerInputs,
            fake,
            helper.getCompilationUnitSources(),
            linkCompileOutputSeparately);
    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(
        ruleContext, runfiles, executable, ruleContext.getConfiguration().buildRunfiles());

    TransitiveLipoInfoProvider transitiveLipoInfo;
    if (cppConfiguration.isLipoContextCollector()) {
      transitiveLipoInfo = common.collectTransitiveLipoLabels(ccCompilationOutputs);
    } else {
      transitiveLipoInfo = TransitiveLipoInfoProvider.EMPTY;
    }

    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    addTransitiveInfoProviders(
        ruleContext,
        cppConfiguration,
        common,
        ruleBuilder,
        filesToBuild,
        ccCompilationOutputs,
        cppCompilationContext,
        linkingOutputs,
        dwoArtifacts,
        transitiveLipoInfo,
        fake);

    Map<Artifact, IncludeScannable> scannableMap = new LinkedHashMap<>();
    Map<PathFragment, Artifact> sourceFileMap = new LinkedHashMap<>();
    if (cppConfiguration.isLipoContextCollector()) {
      for (IncludeScannable scannable : transitiveLipoInfo.getTransitiveIncludeScannables()) {
        // These should all be CppCompileActions, which should have only one source file.
        // This is also checked when they are put into the nested set.
        Artifact source =
            Iterables.getOnlyElement(scannable.getIncludeScannerSources());
        scannableMap.put(source, scannable);
        sourceFileMap.put(source.getExecPath(), source);
      }
    }
   
    // Support test execution on darwin.
    if (Platform.isApplePlatform(cppConfiguration.getTargetCpu())
        && TargetUtils.isTestRule(ruleContext.getRule())) {
      ruleBuilder.addNativeDeclaredProvider(
          new ExecutionInfoProvider(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, "")));
    }

    return ruleBuilder
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .addProvider(
            CppDebugPackageProvider.class,
            new CppDebugPackageProvider(
                ruleContext.getLabel(), strippedFile, executable, explicitDwpFile))
        .setRunfilesSupport(runfilesSupport, executable)
        .addProvider(
            LipoContextProvider.class,
            new LipoContextProvider(
                cppCompilationContext,
                ImmutableMap.copyOf(scannableMap),
                ImmutableMap.copyOf(sourceFileMap)))
        .addProvider(CppLinkAction.Context.class, linkContext)
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .build();
  }

  /**
   * Given 'temps', traverse this target and its dependencies and collect up all the object files,
   * libraries, linker options, linkstamps attributes and linker scripts.
   */
  private static CppLinkActionBuilder determineLinkerArguments(
      RuleContext context,
      CcCommon common,
      PrecompiledFiles precompiledFiles,
      CcLibraryHelper.Info info,
      ImmutableSet<Artifact> compilationPrerequisites,
      boolean fake,
      Artifact binary,
      LinkStaticness linkStaticness,
      List<String> linkopts,
      boolean linkCompileOutputSeparately)
      throws InterruptedException {
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(context, binary)
            .setCrosstoolInputs(CppHelper.getToolchain(context).getLink())
            .addNonCodeInputs(compilationPrerequisites);

    // Either link in the .o files generated for the sources of this target or link in the
    // generated dynamic library they are compiled into.
    if (linkCompileOutputSeparately) {
      for (LibraryToLink library : info.getCcLinkingOutputs().getDynamicLibraries()) {
        builder.addLibrary(library);
      }
    } else {
      boolean usePic = CppHelper.usePic(context, !isLinkShared(context));
      Iterable<Artifact> objectFiles = info.getCcCompilationOutputs().getObjectFiles(usePic);

      if (fake) {
        builder.addFakeObjectFiles(objectFiles);
      } else {
        builder.addObjectFiles(objectFiles);
      }
    }

    builder.addLTOBitcodeFiles(info.getCcCompilationOutputs().getLtoBitcodeFiles());
    builder.addNonCodeInputs(common.getLinkerScripts());

    // Determine the libraries to link in.
    // First libraries from srcs. Shared library artifacts here are substituted with mangled symlink
    // artifacts generated by getDynamicLibraryLink(). This is done to minimize number of -rpath
    // entries during linking process.
    for (Artifact library : precompiledFiles.getLibraries()) {
      if (Link.SHARED_LIBRARY_FILETYPES.matches(library.getFilename())) {
        builder.addLibrary(LinkerInputs.solibLibraryToLink(
            common.getDynamicLibrarySymlink(library, true), library,
            CcLinkingOutputs.libraryIdentifierOf(library)));
      } else if (Link.LINK_LIBRARY_FILETYPES.matches(library.getFilename())) {
        builder.addLibrary(LinkerInputs.precompiledLibraryToLink(
            library, ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY));
      } else if (Link.ARCHIVE_FILETYPES.matches(library.getFilename())) {
        builder.addLibrary(LinkerInputs.precompiledLibraryToLink(
            library, ArtifactCategory.STATIC_LIBRARY));
      } else {
        throw new IllegalStateException();
      }
    }

    // Then the link params from the closure of deps.
    CcLinkParams linkParams = collectCcLinkParams(
        context, linkStaticness != LinkStaticness.DYNAMIC, isLinkShared(context), linkopts);
    builder.addLinkParams(linkParams, context);
    return builder;
  }

  /**
   * Returns "true" if the {@code linkshared} attribute exists and is set.
   */
  private static final boolean isLinkShared(RuleContext context) {
    return context.attributes().has("linkshared", Type.BOOLEAN)
        && context.attributes().get("linkshared", Type.BOOLEAN);
  }

  private static final boolean dashStaticInLinkopts(List<String> linkopts,
      CppConfiguration cppConfiguration) {
    return linkopts.contains("-static")
        || cppConfiguration.getLinkOptions().contains("-static");
  }

  private static final LinkStaticness getLinkStaticness(RuleContext context,
      List<String> linkopts, CppConfiguration cppConfiguration) {
    if (cppConfiguration.getDynamicMode() == DynamicMode.FULLY) {
      return LinkStaticness.DYNAMIC;
    } else if (dashStaticInLinkopts(linkopts, cppConfiguration)) {
      return LinkStaticness.FULLY_STATIC;
    } else if (cppConfiguration.getDynamicMode() == DynamicMode.OFF
        || context.attributes().get("linkstatic", Type.BOOLEAN)) {
      return LinkStaticness.MOSTLY_STATIC;
    } else {
      return LinkStaticness.DYNAMIC;
    }
  }

  /**
   * Collects .dwo artifacts either transitively or directly, depending on the link type.
   *
   * <p>For a cc_binary, we only include the .dwo files corresponding to the .o files that are
   * passed into the link. For static linking, this includes all transitive dependencies. But
   * for dynamic linking, dependencies are separately linked into their own shared libraries,
   * so we don't need them here.
   */
  private static DwoArtifactsCollector collectTransitiveDwoArtifacts(RuleContext context,
      CcCompilationOutputs compilationOutputs, LinkStaticness linkStaticness) {
    if (linkStaticness == LinkStaticness.DYNAMIC) {
      return DwoArtifactsCollector.directCollector(compilationOutputs);
    } else {
      return CcCommon.collectTransitiveDwoArtifacts(context, compilationOutputs);
    }
  }

  @VisibleForTesting
  public static Iterable<Artifact> getDwpInputs(
      RuleContext context, NestedSet<Artifact> picDwoArtifacts, NestedSet<Artifact> dwoArtifacts) {
    return CppHelper.usePic(context, !isLinkShared(context)) ? picDwoArtifacts : dwoArtifacts;
  }

  /**
   * Creates the actions needed to generate this target's "debug info package"
   * (i.e. its .dwp file).
   */
  private static void createDebugPackagerActions(RuleContext context,
      CppConfiguration cppConfiguration, Artifact dwpOutput,
      DwoArtifactsCollector dwoArtifactsCollector) {
    Iterable<Artifact> allInputs = getDwpInputs(context,
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
    NestedSet<Artifact> dwpTools = CppHelper.getToolchain(context).getDwp();
    Preconditions.checkState(!dwpTools.isEmpty());

    List<SpawnAction.Builder> packagers = createIntermediateDwpPackagers(
        context, dwpOutput, cppConfiguration, dwpTools, allInputs, 1);

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
    context.registerAction(Iterables.getOnlyElement(packagers)
        .addArgument("-o")
        .addOutputArgument(dwpOutput)
        .setMnemonic("CcGenerateDwp")
        .build(context));
  }

  /**
   * Creates the intermediate actions needed to generate this target's
   * "debug info package" (i.e. its .dwp file).
   */
  private static List<SpawnAction.Builder> createIntermediateDwpPackagers(RuleContext context,
      Artifact dwpOutput, CppConfiguration cppConfiguration, NestedSet<Artifact> dwpTools,
      Iterable<Artifact> inputs, int intermediateDwpCount) {
    List<SpawnAction.Builder> packagers = new ArrayList<>();

    // Step 1: generate our batches. We currently break into arbitrary batches of fixed maximum
    // input counts, but we can always apply more intelligent heuristics if the need arises.
    SpawnAction.Builder currentPackager = newDwpAction(cppConfiguration, dwpTools);
    int inputsForCurrentPackager = 0;

    for (Artifact dwoInput : inputs) {
      if (inputsForCurrentPackager == MAX_INPUTS_PER_DWP_ACTION) {
        packagers.add(currentPackager);
        currentPackager = newDwpAction(cppConfiguration, dwpTools);
        inputsForCurrentPackager = 0;
      }
      currentPackager.addInputArgument(dwoInput);
      inputsForCurrentPackager++;
    }
    packagers.add(currentPackager);
    // Step 2: given the batches, create the actions.
    if (packagers.size() > 1) {
      // If we have multiple batches, make them all intermediate actions, then pipe their outputs
      // into an additional level.
      List<Artifact> intermediateOutputs = new ArrayList<>();

      for (SpawnAction.Builder packager : packagers) {
        Artifact intermediateOutput =
            getIntermediateDwpFile(context, dwpOutput, intermediateDwpCount++);
        context.registerAction(packager
            .addArgument("-o")
            .addOutputArgument(intermediateOutput)
            .setMnemonic("CcGenerateIntermediateDwp")
            .build(context));
        intermediateOutputs.add(intermediateOutput);
      }
      return createIntermediateDwpPackagers(
          context, dwpOutput, cppConfiguration, dwpTools, intermediateOutputs,
          intermediateDwpCount);
    }
    return packagers;
  }

  /**
   * Returns a new SpawnAction builder for generating dwp files, pre-initialized with
   * standard settings.
   */
  private static SpawnAction.Builder newDwpAction(CppConfiguration cppConfiguration,
      NestedSet<Artifact> dwpTools) {
    return new SpawnAction.Builder()
        .addTransitiveInputs(dwpTools)
        .setExecutable(cppConfiguration.getDwpExecutable())
        .useParameterFile(ParameterFile.ParameterFileType.UNQUOTED);
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
        new PathFragment(INTERMEDIATE_DWP_DIR + "/" + intermediatePath.getPathString()),
        dwpOutput.getRoot());
  }

  /**
   * Collect link parameters from the transitive closure.
   */
  private static CcLinkParams collectCcLinkParams(RuleContext context,
      boolean linkingStatically, boolean linkShared, List<String> linkopts) {
    CcLinkParams.Builder builder = CcLinkParams.builder(linkingStatically, linkShared);

    if (isLinkShared(context)) {
      // CcLinkingOutputs is empty because this target is not configured yet
      builder.addCcLibrary(context, false, linkopts, CcLinkingOutputs.EMPTY);
    } else {
      builder.addTransitiveTargets(
          context.getPrerequisites("deps", Mode.TARGET),
          CcLinkParamsProvider.TO_LINK_PARAMS, CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
      builder.addTransitiveTarget(CppHelper.mallocForTarget(context));
      builder.addLinkOpts(linkopts);
    }
    return builder.build();
  }

  private static void addTransitiveInfoProviders(
      RuleContext ruleContext,
      CppConfiguration cppConfiguration,
      CcCommon common,
      RuleConfiguredTargetBuilder builder,
      NestedSet<Artifact> filesToBuild,
      CcCompilationOutputs ccCompilationOutputs,
      CppCompilationContext cppCompilationContext,
      CcLinkingOutputs linkingOutputs,
      DwoArtifactsCollector dwoArtifacts,
      TransitiveLipoInfoProvider transitiveLipoInfo,
      boolean fake) {
    List<Artifact> instrumentedObjectFiles = new ArrayList<>();
    instrumentedObjectFiles.addAll(ccCompilationOutputs.getObjectFiles(false));
    instrumentedObjectFiles.addAll(ccCompilationOutputs.getObjectFiles(true));
    InstrumentedFilesProvider instrumentedFilesProvider = common.getInstrumentedFilesProvider(
        instrumentedObjectFiles, !TargetUtils.isTestRule(ruleContext.getRule()) && !fake);

    NestedSet<Artifact> headerTokens =
        CcLibraryHelper.collectHeaderTokens(ruleContext, ccCompilationOutputs);
    NestedSet<Artifact> filesToCompile =
        ccCompilationOutputs.getFilesToCompile(
            cppConfiguration.isLipoContextCollector(),
            cppConfiguration.processHeadersInDependencies(),
            CppHelper.usePic(ruleContext, false));
    builder
        .setFilesToBuild(filesToBuild)
        .addProvider(CppCompilationContext.class, cppCompilationContext)
        .addProvider(TransitiveLipoInfoProvider.class, transitiveLipoInfo)
        .addProvider(
            CcExecutionDynamicLibrariesProvider.class,
            new CcExecutionDynamicLibrariesProvider(
                collectExecutionDynamicLibraryArtifacts(
                    ruleContext, linkingOutputs.getExecutionDynamicLibraries())))
        .addProvider(
            CcNativeLibraryProvider.class,
            new CcNativeLibraryProvider(
                collectTransitiveCcNativeLibraries(
                    ruleContext, linkingOutputs.getDynamicLibraries())))
        .addProvider(InstrumentedFilesProvider.class, instrumentedFilesProvider)
        .addProvider(
            CppDebugFileProvider.class,
            new CppDebugFileProvider(
                dwoArtifacts.getDwoArtifacts(), dwoArtifacts.getPicDwoArtifacts()))
        .addOutputGroup(
            OutputGroupProvider.TEMP_FILES, getTemps(cppConfiguration, ccCompilationOutputs))
        .addOutputGroup(OutputGroupProvider.FILES_TO_COMPILE, filesToCompile)
        // For CcBinary targets, we only want to ensure that we process headers in dependencies and
        // thus only add header tokens to HIDDEN_TOP_LEVEL. If we add all HIDDEN_TOP_LEVEL artifacts
        // from dependent CcLibrary targets, we'd be building .pic.o files in nopic builds.
        .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, headerTokens)
        .addOutputGroup(
            OutputGroupProvider.COMPILATION_PREREQUISITES,
            CcCommon.collectCompilationPrerequisites(ruleContext, cppCompilationContext));

    CppHelper.maybeAddStaticLinkMarkerProvider(builder, ruleContext);
  }

  private static NestedSet<Artifact> collectExecutionDynamicLibraryArtifacts(
      RuleContext ruleContext,
      List<LibraryToLink> executionDynamicLibraries) {
    Iterable<Artifact> artifacts = LinkerInputs.toLibraryArtifacts(executionDynamicLibraries);
    if (!Iterables.isEmpty(artifacts)) {
      return NestedSetBuilder.wrap(Order.STABLE_ORDER, artifacts);
    }

    Iterable<CcExecutionDynamicLibrariesProvider> deps = ruleContext
        .getPrerequisites("deps", Mode.TARGET, CcExecutionDynamicLibrariesProvider.class);

    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (CcExecutionDynamicLibrariesProvider dep : deps) {
      builder.addTransitive(dep.getExecutionDynamicLibraryArtifacts());
    }
    return builder.build();
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

  private static NestedSet<Artifact> getTemps(CppConfiguration cppConfiguration,
      CcCompilationOutputs compilationOutputs) {
    return cppConfiguration.isLipoContextCollector()
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : compilationOutputs.getTemps();
  }
}
