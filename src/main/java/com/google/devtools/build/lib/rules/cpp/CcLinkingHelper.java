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
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.Link.Picness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A class to create C/C++ link actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppLinkActionBuilder.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the toolchain,
 * the STL, the lipo context, and so on. Optionally, they can also have copts, and malloc
 * attributes, but note that these require explicit calls to the corresponding setter methods.
 */
public final class CcLinkingHelper {

  /** A string constant for the name of archive library(.a, .lo) output group. */
  public static final String ARCHIVE_LIBRARY_OUTPUT_GROUP_NAME = "archive";

  /** A string constant for the name of dynamic library output group. */
  public static final String DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME = "dynamic_library";

  /** Contains the providers as well as the linking outputs. */
  @SkylarkModule(
    name = "linking_info",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Helper class containing CC linking providers."
  )
  public static final class LinkingInfo {
    private final TransitiveInfoProviderMap providers;
    private final Map<String, NestedSet<Artifact>> outputGroups;
    private final CcLinkingOutputs linkingOutputs;
    private final CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries;

    private LinkingInfo(
        TransitiveInfoProviderMap providers,
        Map<String, NestedSet<Artifact>> outputGroups,
        CcLinkingOutputs linkingOutputs,
        CcLinkingOutputs linkingOutputsExcludingPrecompiledLibraries) {
      this.providers = providers;
      this.outputGroups = outputGroups;
      this.linkingOutputs = linkingOutputs;
      this.linkingOutputsExcludingPrecompiledLibraries =
          linkingOutputsExcludingPrecompiledLibraries;
    }

    public TransitiveInfoProviderMap getProviders() {
      return providers;
    }

    @SkylarkCallable(name = "cc_linking_info", documented = false)
    public CcLinkingInfo getCcLinkParamsInfo() {
      return (CcLinkingInfo) providers.getProvider(CcLinkingInfo.PROVIDER.getKey());
    }

    public Map<String, NestedSet<Artifact>> getOutputGroups() {
      return outputGroups;
    }

    public CcLinkingOutputs getCcLinkingOutputs() {
      return linkingOutputs;
    }

    /**
     * Returns the linking outputs before adding the pre-compiled libraries. Avoid using this -
     * pre-compiled and locally compiled libraries should be treated identically. This method only
     * exists for backwards compatibility.
     */
    public CcLinkingOutputs getCcLinkingOutputsExcludingPrecompiledLibraries() {
      return linkingOutputsExcludingPrecompiledLibraries;
    }

    /**
     * Adds the static, pic-static libraries to the given builder. If addDynamicLibraries parameter
     * is true, it also adds dynamic(both compile-time and execution-time) libraries.
     */
    public void addLinkingOutputsTo(
        NestedSetBuilder<Artifact> filesBuilder, boolean addDynamicLibraries) {
      filesBuilder
          .addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getStaticLibraries()))
          .addAll(LinkerInputs.toLibraryArtifacts(linkingOutputs.getPicStaticLibraries()));
      if (addDynamicLibraries) {
        filesBuilder
            .addAll(LinkerInputs.toNonSolibArtifacts(linkingOutputs.getDynamicLibraries()))
            .addAll(
                LinkerInputs.toNonSolibArtifacts(linkingOutputs.getExecutionDynamicLibraries()));
      }
    }

    public void addLinkingOutputsTo(NestedSetBuilder<Artifact> filesBuilder) {
      addLinkingOutputsTo(filesBuilder, true);
    }
  }

  private final RuleContext ruleContext;
  private final CppSemantics semantics;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  private final List<Artifact> nonCodeLinkerInputs = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkstamps = NestedSetBuilder.stableOrder();
  private final List<Artifact> linkActionInputs = new ArrayList<>();

  @Nullable private Artifact dynamicLibrary;
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverlink;

  private final List<LibraryToLink> staticLibraries = new ArrayList<>();
  private final List<LibraryToLink> picStaticLibraries = new ArrayList<>();
  private final List<LibraryToLink> dynamicLibraries = new ArrayList<>();
  private final List<LibraryToLink> executionDynamicLibraries = new ArrayList<>();

  private boolean checkDepsGenerateCpp = true;
  private boolean emitLinkActionsIfEmpty;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitCcSpecificLinkParamsProvider;
  private boolean emitInterfaceSharedObjects;
  private boolean shouldCreateDynamicLibrary = true;
  private boolean shouldCreateStaticLibraries = true;
  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainProvider ccToolchain;
  private final FdoSupportProvider fdoSupport;
  private String linkedArtifactNameSuffix = "";

  /**
   * Creates a CcLinkingHelper that outputs artifacts in a given configuration.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoSupport the C++ FDO optimization support provider for the build
   * @param configuration the configuration that gives the directory of output artifacts
   */
  public CcLinkingHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoSupportProvider fdoSupport,
      BuildConfiguration configuration) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = Preconditions.checkNotNull(semantics);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoSupport = Preconditions.checkNotNull(fdoSupport);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration =
        Preconditions.checkNotNull(ruleContext.getFragment(CppConfiguration.class));
  }

  /** Sets fields that overlap for cc_library and cc_binary rules. */
  public CcLinkingHelper fromCommon(CcCommon common) {
    addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET));
    addNonCodeLinkerInputs(common.getLinkerScripts());
    return this;
  }

  /** Adds the corresponding non-code files as linker inputs. */
  public void addNonCodeLinkerInputs(Iterable<Artifact> nonCodeLinkerInputs) {
    for (Artifact nonCodeLinkerInput : nonCodeLinkerInputs) {
      String basename = nonCodeLinkerInput.getFilename();
      Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.SHARED_LIBRARY_FILETYPES.matches(basename));
      this.nonCodeLinkerInputs.add(nonCodeLinkerInput);
    }
  }

  /**
   * Add the corresponding files as static libraries into the linker outputs (i.e., after the linker
   * action) - this makes them available for linking to binary rules that depend on this rule.
   */
  public CcLinkingHelper addStaticLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(staticLibraries, libraries);
    return this;
  }

  /**
   * Add the corresponding files as static libraries into the linker outputs (i.e., after the linker
   * action) - this makes them available for linking to binary rules that depend on this rule.
   */
  public CcLinkingHelper addPicStaticLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(picStaticLibraries, libraries);
    return this;
  }

  /**
   * Add the corresponding files as dynamic libraries into the linker outputs (i.e., after the
   * linker action) - this makes them available for linking to binary rules that depend on this
   * rule.
   */
  public CcLinkingHelper addDynamicLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(dynamicLibraries, libraries);
    return this;
  }

  /** Add the corresponding files as dynamic libraries required at runtime */
  public CcLinkingHelper addExecutionDynamicLibraries(Iterable<LibraryToLink> libraries) {
    Iterables.addAll(executionDynamicLibraries, libraries);
    return this;
  }

  /** Adds the given options as linker options to the link command. */
  public CcLinkingHelper addLinkopts(Iterable<String> linkopts) {
    Iterables.addAll(this.linkopts, linkopts);
    return this;
  }

  /**
   * Adds the given targets as dependencies - this can include explicit dependencies on other rules
   * (like from a "deps" attribute) and also implicit dependencies on runtime libraries.
   */
  public CcLinkingHelper addDeps(Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      this.deps.add(dep);
    }
    return this;
  }

  /**
   * Adds the given linkstamps. Note that linkstamps are usually not compiled at the library level,
   * but only in the dependent binary rules.
   */
  public CcLinkingHelper addLinkstamps(Iterable<? extends TransitiveInfoCollection> linkstamps) {
    for (TransitiveInfoCollection linkstamp : linkstamps) {
      this.linkstamps.addTransitive(linkstamp.getProvider(FileProvider.class).getFilesToBuild());
    }
    return this;
  }

  /** Adds the given artifact to the input of any generated link actions. */
  public CcLinkingHelper addLinkActionInput(Artifact input) {
    Preconditions.checkNotNull(input);
    this.linkActionInputs.add(input);
    return this;
  }

  /** Adds a variableExtension to template the crosstool. */
  public CcLinkingHelper addVariableExtension(VariablesExtension variableExtension) {
    Preconditions.checkNotNull(variableExtension);
    this.variablesExtensions.add(variableExtension);
    return this;
  }

  /**
   * Overrides the path for the generated dynamic library - this should only be called if the
   * dynamic library is an implicit or explicit output of the rule, i.e., if it is accessible by
   * name from other rules in the same package. Set to {@code null} to use the default computation.
   */
  public CcLinkingHelper setDynamicLibrary(@Nullable Artifact dynamicLibrary) {
    this.dynamicLibrary = dynamicLibrary;
    return this;
  }

  /**
   * Marks the output of this rule as alwayslink, i.e., the corresponding symbols will be retained
   * by the linker even if they are not otherwise used. This is useful for libraries that register
   * themselves somewhere during initialization.
   *
   * <p>This only sets the link type (see {@link #setStaticLinkType}), either to a static library or
   * to an alwayslink static library (blaze uses a different file extension to signal alwayslink to
   * downstream code).
   */
  public CcLinkingHelper setAlwayslink(boolean alwayslink) {
    linkType =
        alwayslink ? LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY : LinkTargetType.STATIC_LIBRARY;
    return this;
  }

  /**
   * Directly set the link type. This can be used instead of {@link #setAlwayslink}. Setting
   * anything other than a static link causes this class to skip the link action creation.
   */
  public CcLinkingHelper setStaticLinkType(LinkTargetType linkType) {
    Preconditions.checkNotNull(linkType);
    Preconditions.checkState(linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER);
    this.linkType = linkType;
    return this;
  }

  /**
   * Marks the resulting code as neverlink, i.e., the code will not be linked into dependent
   * libraries or binaries - the header files are still available.
   */
  public CcLinkingHelper setNeverLink(boolean neverlink) {
    this.neverlink = neverlink;
    return this;
  }

  /**
   * Disables checking that the deps actually are C++ rules. By default, the {@link #link} method
   * uses {@link LanguageDependentFragment.Checker#depSupportsLanguage} to check that all deps
   * provide C++ providers.
   */
  public CcLinkingHelper setCheckDepsGenerateCpp(boolean checkDepsGenerateCpp) {
    this.checkDepsGenerateCpp = checkDepsGenerateCpp;
    return this;
  }

  /*
   * Adds a suffix for paths of linked artifacts. Normally their paths are derived solely from rule
   * labels. In the case of multiple callers (e.g., aspects) acting on a single rule, they may
   * generate the same linked artifact and therefore lead to artifact conflicts. This method
   * provides a way to avoid this artifact conflict by allowing different callers acting on the same
   * rule to provide a suffix that will be used to scope their own linked artifacts.
   */
  public CcLinkingHelper setLinkedArtifactNameSuffix(String suffix) {
    this.linkedArtifactNameSuffix = Preconditions.checkNotNull(suffix);
    return this;
  }

  /** This adds the {@link CcNativeLibraryProvider} to the providers created by this class. */
  public CcLinkingHelper enableCcNativeLibrariesProvider() {
    this.emitCcNativeLibrariesProvider = true;
    return this;
  }

  /**
   * This adds the {@link CcSpecificLinkParamsProvider} to the providers created by this class.
   * Otherwise the result will contain an instance of {@link CcLinkParamsInfo}.
   */
  public CcLinkingHelper enableCcSpecificLinkParamsProvider() {
    this.emitCcSpecificLinkParamsProvider = true;
    return this;
  }

  /**
   * Enables or disables generation of link actions if there are no object files. Some rules declare
   * a <code>.a</code> or <code>.so</code> implicit output, which requires that these files are
   * created even if there are no object files, so be careful when calling this.
   *
   * <p>This is disabled by default.
   */
  public CcLinkingHelper setGenerateLinkActionsIfEmpty(boolean emitLinkActionsIfEmpty) {
    this.emitLinkActionsIfEmpty = emitLinkActionsIfEmpty;
    return this;
  }

  /**
   * Enables the optional generation of interface dynamic libraries - this is only used when the
   * linker generates a dynamic library, and only if the crosstool supports it. The default is not
   * to generate interface dynamic libraries.
   */
  public CcLinkingHelper enableInterfaceSharedObjects() {
    this.emitInterfaceSharedObjects = true;
    return this;
  }

  /**
   * This enables or disables the generation of a dynamic library link action. The default is to
   * generate a dynamic library. Note that the selection between dynamic or static linking is
   * performed at the binary rule level.
   */
  public CcLinkingHelper setShouldCreateDynamicLibrary(boolean emitDynamicLibrary) {
    this.shouldCreateDynamicLibrary = emitDynamicLibrary;
    return this;
  }

  /**
   * When shouldCreateStaticLibraries is true, there are no actions created for static libraries.
   */
  public CcLinkingHelper setShouldCreateStaticLibraries(boolean emitStaticLibraries) {
    this.shouldCreateStaticLibraries = emitStaticLibraries;
    return this;
  }

  public CcLinkingHelper setNeverlink(boolean neverlink) {
    this.neverlink = neverlink;
    return this;
  }

  /**
   * Create the C++ link actions, and the corresponding linking related providers.
   *
   * @throws RuleErrorException
   */
  // TODO(b/73997894): Try to remove CcCompilationContextInfo. Right now headers are passed as non
  // code
  // inputs to the linker.
  public LinkingInfo link(
      CcCompilationOutputs ccOutputs, CcCompilationContextInfo ccCompilationContextInfo)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(ccOutputs);
    Preconditions.checkNotNull(ccCompilationContextInfo);

    if (checkDepsGenerateCpp) {
      for (LanguageDependentFragment dep :
          AnalysisUtils.getProviders(deps, LanguageDependentFragment.class)) {
        LanguageDependentFragment.Checker.depSupportsLanguage(
            ruleContext, dep, CppRuleClasses.LANGUAGE, "deps");
      }
    }

    // Create link actions (only if there are object files or if explicitly requested).
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (emitLinkActionsIfEmpty || !ccOutputs.isEmpty()) {
      // On some systems, the linker gives an error message if there are no input files. Even with
      // the check above, this can still happen if there is a .nopic.o or .o files in srcs, but no
      // other files. To fix that, we'd have to check for each link action individually.
      //
      // An additional pre-existing issue is that the header check tokens are dropped if we don't
      // generate any link actions, effectively disabling header checking in some cases.
      if (linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER) {
        // TODO(bazel-team): This can't create the link action for a cc_binary yet.
        ccLinkingOutputs = createCcLinkActions(ccOutputs, nonCodeLinkerInputs);
      }
    }
    CcLinkingOutputs originalLinkingOutputs = ccLinkingOutputs;
    if (!(staticLibraries.isEmpty()
        && picStaticLibraries.isEmpty()
        && dynamicLibraries.isEmpty()
        && executionDynamicLibraries.isEmpty())) {

      CcLinkingOutputs.Builder newOutputsBuilder = new CcLinkingOutputs.Builder();
      if (!ccOutputs.isEmpty()) {
        // Add the linked outputs of this rule iff we had anything to put in them, but then
        // make sure we're not colliding with some library added from the inputs.
        newOutputsBuilder.merge(originalLinkingOutputs);
        ImmutableSetMultimap<String, LibraryToLink> precompiledLibraryMap =
            CcLinkingOutputs.getLibrariesByIdentifier(
                Iterables.concat(
                    staticLibraries, picStaticLibraries,
                    dynamicLibraries, executionDynamicLibraries));
        ImmutableSetMultimap<String, LibraryToLink> linkedLibraryMap =
            originalLinkingOutputs.getLibrariesByIdentifier();
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

      // Merge the pre-compiled libraries (static & dynamic) into the linker outputs.
      ccLinkingOutputs =
          newOutputsBuilder
              .addStaticLibraries(staticLibraries)
              .addPicStaticLibraries(picStaticLibraries)
              .addDynamicLibraries(dynamicLibraries)
              .addExecutionDynamicLibraries(executionDynamicLibraries)
              .build();
    }

    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();

    if (shouldAddLinkerOutputArtifacts(ruleContext, ccOutputs)) {
      addLinkerOutputArtifacts(outputGroups, ccOutputs);
    }

    // Be very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    TransitiveInfoProviderMapBuilder providers = new TransitiveInfoProviderMapBuilder();

    // TODO(bazel-team): Maybe we can infer these from other data at the places where they are
    // used.
    if (emitCcNativeLibrariesProvider) {
      providers.add(new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }

    Runfiles cppStaticRunfiles = collectCppRunfiles(ccLinkingOutputs, true);
    Runfiles cppSharedRunfiles = collectCppRunfiles(ccLinkingOutputs, false);

    CcLinkingInfo.Builder ccLinkingInfoBuilder = CcLinkingInfo.Builder.create();
    ccLinkingInfoBuilder.setCcRunfilesInfo(
        new CcRunfilesInfo(cppStaticRunfiles, cppSharedRunfiles));
    ccLinkingInfoBuilder.setCcExecutionDynamicLibrariesInfo(
        collectExecutionDynamicLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));

    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean forcePic = cppConfiguration.forcePic();
    if (emitCcSpecificLinkParamsProvider) {
      providers.add(
          new CcSpecificLinkParamsProvider(
              createCcLinkParamsStore(ccLinkingOutputs, ccCompilationContextInfo, forcePic)));
    } else {
      ccLinkingInfoBuilder.setCcLinkParamsInfo(
          new CcLinkParamsInfo(
              createCcLinkParamsStore(ccLinkingOutputs, ccCompilationContextInfo, forcePic)));
    }
    providers.put(ccLinkingInfoBuilder.build());
    return new LinkingInfo(
        providers.build(), outputGroups, ccLinkingOutputs, originalLinkingOutputs);
  }

  /**
   * Returns true if the appropriate attributes for linker output artifacts are defined, and either
   * the compile action produces object files or the build is configured to produce an archive or
   * dynamic library even in the absence of object files.
   */
  private boolean shouldAddLinkerOutputArtifacts(
      RuleContext ruleContext, CcCompilationOutputs ccOutputs) {
    return (ruleContext.attributes().has("alwayslink", Type.BOOLEAN)
        && ruleContext.attributes().has("linkstatic", Type.BOOLEAN)
        && (emitLinkActionsIfEmpty || !ccOutputs.isEmpty()));
  }

  /**
   * Adds linker output artifacts to the given map, to be registered on the configured target as
   * output groups.
   */
  private void addLinkerOutputArtifacts(
      Map<String, NestedSet<Artifact>> outputGroups, CcCompilationOutputs ccOutputs) {
    NestedSetBuilder<Artifact> archiveFile = new NestedSetBuilder<>(Order.STABLE_ORDER);
    NestedSetBuilder<Artifact> dynamicLibrary = new NestedSetBuilder<>(Order.STABLE_ORDER);

    if (ruleContext.attributes().get("alwayslink", Type.BOOLEAN)) {
      archiveFile.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext,
              configuration,
              Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY,
              linkedArtifactNameSuffix));
    } else {
      archiveFile.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext,
              configuration,
              Link.LinkTargetType.STATIC_LIBRARY,
              linkedArtifactNameSuffix));
    }

    if (!ruleContext.attributes().get("linkstatic", Type.BOOLEAN) && !ccOutputs.isEmpty()) {
      dynamicLibrary.add(
          CppHelper.getLinuxLinkedArtifact(
              ruleContext,
              configuration,
              Link.LinkTargetType.NODEPS_DYNAMIC_LIBRARY,
              linkedArtifactNameSuffix));

      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)
          && emitInterfaceSharedObjects) {
        dynamicLibrary.add(
            CppHelper.getLinuxLinkedArtifact(
                ruleContext,
                configuration,
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
                linkedArtifactNameSuffix));
      }
    }

    outputGroups.put(ARCHIVE_LIBRARY_OUTPUT_GROUP_NAME, archiveFile.build());
    outputGroups.put(DYNAMIC_LIBRARY_OUTPUT_GROUP_NAME, dynamicLibrary.build());
  }

  private Runfiles collectCppRunfiles(
      CcLinkingOutputs ccLinkingOutputs, boolean linkingStatically) {
    Runfiles.Builder builder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());
    builder.addTargets(deps, RunfilesProvider.DEFAULT_RUNFILES);
    builder.addTargets(deps, CcRunfilesInfo.runfilesFunction(linkingStatically));
    // Add the shared libraries to the runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically));
    return builder.build();
  }

  private CcLinkParamsStore createCcLinkParamsStore(
      final CcLinkingOutputs ccLinkingOutputs,
      final CcCompilationContextInfo ccCompilationContextInfo,
      final boolean forcePic) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(
          CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
        builder.addLinkstamps(linkstamps.build(), ccCompilationContextInfo);
        builder.addTransitiveTargets(
            deps, CcLinkParamsInfo.TO_LINK_PARAMS, CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        if (!neverlink) {
          builder.addLibraries(
              ccLinkingOutputs.getPreferredLibraries(
                  linkingStatically, /*preferPic=*/ linkShared || forcePic));
          if (!linkingStatically
              || (ccLinkingOutputs.getStaticLibraries().isEmpty()
                  && ccLinkingOutputs.getPicStaticLibraries().isEmpty())) {
            builder.addExecutionDynamicLibraries(
                LinkerInputs.toLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));
          }
          builder.addLinkOpts(linkopts);
          builder.addNonCodeInputs(nonCodeLinkerInputs);
        }
      }
    };
  }

  private NestedSet<LinkerInput> collectNativeCcLibraries(CcLinkingOutputs ccLinkingOutputs) {
    NestedSetBuilder<LinkerInput> result = NestedSetBuilder.linkOrder();
    result.addAll(ccLinkingOutputs.getDynamicLibraries());
    for (CcNativeLibraryProvider dep :
        AnalysisUtils.getProviders(deps, CcNativeLibraryProvider.class)) {
      result.addTransitive(dep.getTransitiveCcNativeLibraries());
    }

    return result.build();
  }

  private CcExecutionDynamicLibrariesInfo collectExecutionDynamicLibraryArtifacts(
      List<LibraryToLink> executionDynamicLibraries) {
    Iterable<Artifact> artifacts = LinkerInputs.toLibraryArtifacts(executionDynamicLibraries);
    if (!Iterables.isEmpty(artifacts)) {
      return new CcExecutionDynamicLibrariesInfo(
          NestedSetBuilder.wrap(Order.STABLE_ORDER, artifacts));
    }

    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (CcLinkingInfo dep : AnalysisUtils.getProviders(deps, CcLinkingInfo.PROVIDER)) {
      CcExecutionDynamicLibrariesInfo ccExecutionDynamicLibrariesInfo =
          dep.getCcExecutionDynamicLibrariesInfo();
      if (ccExecutionDynamicLibrariesInfo != null) {
        builder.addTransitive(
            ccExecutionDynamicLibrariesInfo.getExecutionDynamicLibraryArtifacts());
      }
    }
    return builder.isEmpty()
        ? CcExecutionDynamicLibrariesInfo.EMPTY
        : new CcExecutionDynamicLibrariesInfo(builder.build());
  }

  /**
   * Constructs the C++ linker actions. It generally generates two actions, one for a static library
   * and one for a dynamic library. If PIC is required for shared libraries, but not for binaries,
   * it additionally creates a third action to generate a PIC static library. If PIC is required for
   * shared libraries and binaries, then only PIC actions are registered.
   *
   * <p>For dynamic libraries, this method can additionally create an interface shared library that
   * can be used for linking, but doesn't contain any executable code. This increases the number of
   * cache hits for link actions. Call {@link #enableInterfaceSharedObjects()} to enable this
   * behavior.
   *
   * @throws RuleErrorException
   */
  private CcLinkingOutputs createCcLinkActions(
      CcCompilationOutputs ccOutputs, Iterable<Artifact> nonCodeLinkerInputs)
      throws RuleErrorException, InterruptedException {
    // For now only handle static links. Note that the dynamic library link below ignores linkType.
    // TODO(bazel-team): Either support non-static links or move this check to setStaticLinkType().
    Preconditions.checkState(
        linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER, "can only handle static links");

    CcLinkingOutputs.Builder result = new CcLinkingOutputs.Builder();
    if (cppConfiguration.isLipoContextCollector()) {
      // Don't try to create LIPO link actions in collector mode,
      // because it needs some data that's not available at this point.
      return result.build();
    }
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    boolean usePicForBinaries = CppHelper.usePicForBinaries(ruleContext, ccToolchain);
    boolean usePicForDynamicLibs = CppHelper.usePicForDynamicLibraries(ruleContext, ccToolchain);

    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();

    if (shouldCreateStaticLibraries) {
      createStaticLibraries(
          result,
          env,
          usePicForBinaries,
          usePicForDynamicLibs,
          libraryIdentifier,
          ccOutputs,
          nonCodeLinkerInputs);
    }

    if (shouldCreateDynamicLibrary) {
      createDynamicLibrary(result, env, usePicForDynamicLibs, libraryIdentifier, ccOutputs);
    }

    return result.build();
  }

  private void createStaticLibraries(
      CcLinkingOutputs.Builder result,
      AnalysisEnvironment env,
      boolean usePicForBinaries,
      boolean usePicForDynamicLibs,
      String libraryIdentifier,
      CcCompilationOutputs ccOutputs,
      Iterable<Artifact> nonCodeLinkerInputs)
      throws RuleErrorException, InterruptedException {
    // Create static library (.a). The linkType only reflects whether the library is alwayslink or
    // not. The PIC-ness is determined by whether we need to use PIC or not. There are three cases
    // for (usePicForDynamicLibs usePicForBinaries):
    //
    // (1) (false false) -> no pic code
    // (2) (true false)  -> shared libraries as pic, but not binaries
    // (3) (true true)   -> both shared libraries and binaries as pic
    //
    // In case (3), we always need PIC, so only create one static library containing the PIC
    // object
    // files. The name therefore does not match the content.
    //
    // Presumably, it is done this way because the .a file is an implicit output of every
    // cc_library
    // rule, so we can't use ".pic.a" that in the always-PIC case.

    // If the crosstool is configured to select an output artifact, we use that selection.
    // Otherwise, we use linux defaults.
    Artifact linkedArtifact = getLinkedArtifact(linkType);

    CppLinkAction maybePicAction =
        newLinkActionBuilder(linkedArtifact)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForBinaries))
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setUsePicForLtoBackendActions(usePicForBinaries)
            .setLinkType(linkType)
            .setLinkingMode(LinkingMode.LEGACY_FULLY_STATIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtensions(variablesExtensions)
            .build();
    env.registerAction(maybePicAction);
    if (usePicForBinaries) {
      result.addPicStaticLibrary(maybePicAction.getOutputLibrary());
    } else {
      result.addStaticLibrary(maybePicAction.getOutputLibrary());
      // Create a second static library (.pic.a). Only in case (2) do we need both PIC and non-PIC
      // static libraries. In that case, the first static library contains the non-PIC code, and
      // this
      // one contains the PIC code, so the names match the content.
      if (usePicForDynamicLibs) {
        LinkTargetType picLinkType =
            (linkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
                ? LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY
                : LinkTargetType.PIC_STATIC_LIBRARY;

        // If the crosstool is configured to select an output artifact, we use that selection.
        // Otherwise, we use linux defaults.
        Artifact picArtifact = getLinkedArtifact(picLinkType);
        CppLinkAction picAction =
            newLinkActionBuilder(picArtifact)
                .addObjectFiles(ccOutputs.getObjectFiles(/* usePic= */ true))
                .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
                .setUsePicForLtoBackendActions(true)
                .setLinkType(picLinkType)
                .setLinkingMode(LinkingMode.LEGACY_FULLY_STATIC)
                .addActionInputs(linkActionInputs)
                .setLibraryIdentifier(libraryIdentifier)
                .addVariablesExtensions(variablesExtensions)
                .build();
        env.registerAction(picAction);
        result.addPicStaticLibrary(picAction.getOutputLibrary());
      }
    }
  }

  private void createDynamicLibrary(
      CcLinkingOutputs.Builder result,
      AnalysisEnvironment env,
      boolean usePicForDynamicLibs,
      String libraryIdentifier,
      CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    // Create dynamic library.
    Artifact soImpl;
    String mainLibraryIdentifier;
    if (dynamicLibrary == null) {
      // If the crosstool is configured to select an output artifact, we use that selection.
      // Otherwise, we use linux defaults.
      soImpl = getLinkedArtifact(LinkTargetType.NODEPS_DYNAMIC_LIBRARY);
      mainLibraryIdentifier = libraryIdentifier;
    } else {
      // This branch is only used for vestigial Google-internal rules where the name of the output
      // file is explicitly specified in the BUILD file and as such, is platform-dependent. Thus,
      // we just hardcode some reasonable logic to compute the library identifier and hope that this
      // will eventually go away.
      soImpl = dynamicLibrary;
      mainLibraryIdentifier =
          FileSystemUtils.removeExtension(soImpl.getRootRelativePath().getPathString());
    }

    List<String> sonameLinkopts = ImmutableList.of();
    Artifact soInterface = null;
    if (CppHelper.useInterfaceSharedObjects(cppConfiguration, ccToolchain)
        && emitInterfaceSharedObjects) {
      soInterface =
          CppHelper.getLinuxLinkedArtifact(
              ruleContext,
              configuration,
              LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
              linkedArtifactNameSuffix);
      // TODO(b/28946988): Remove this hard-coded flag.
      if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        sonameLinkopts =
            ImmutableList.of(
                "-Wl,-soname="
                    + SolibSymlinkAction.getDynamicLibrarySoname(
                        soImpl.getRootRelativePath(), /* preserveName= */ false));
      }
    }

    CppLinkActionBuilder dynamicLinkActionBuilder =
        newLinkActionBuilder(soImpl)
            .setInterfaceOutput(soInterface)
            .addObjectFiles(ccOutputs.getObjectFiles(usePicForDynamicLibs))
            .addNonCodeInputs(ccOutputs.getHeaderTokenFiles())
            .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setLinkType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .setLinkingMode(LinkingMode.DYNAMIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(mainLibraryIdentifier)
            .addLinkopts(linkopts)
            .addLinkopts(sonameLinkopts)
            .setRuntimeInputs(
                ArtifactCategory.DYNAMIC_LIBRARY,
                ccToolchain.getDynamicRuntimeLinkMiddleman(featureConfiguration),
                ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration))
            .addVariablesExtensions(variablesExtensions);

    if (featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
      // On Windows, we cannot build a shared library with symbols unresolved, so here we
      // dynamically
      // link to all it's dependencies.
      CcLinkParams.Builder ccLinkParamsBuilder =
          CcLinkParams.builder(/* linkingStatically= */ false, /* linkShared= */ true);
      ccLinkParamsBuilder.addCcLibrary(ruleContext);
      dynamicLinkActionBuilder.addLinkParams(ccLinkParamsBuilder.build(), ruleContext);

      // If windows_export_all_symbols feature is enabled, bazel parses object files to generate
      // DEF file and use it to export symbols. The generated DEF file won't be used if a custom
      // DEF file is specified by win_def_file attribute.
      if (CppHelper.shouldUseGeneratedDefFile(ruleContext, featureConfiguration)) {
        Artifact generatedDefFile =
            CppHelper.createDefFileActions(
                ruleContext,
                ruleContext.getPrerequisiteArtifact("$def_parser", Mode.HOST),
                ccOutputs.getObjectFiles(false),
                SolibSymlinkAction.getDynamicLibrarySoname(soImpl.getRootRelativePath(), true));
        dynamicLinkActionBuilder.setDefFile(generatedDefFile);
      }

      // If user specifies a custom DEF file, then we use this one instead of the generated one.
      Artifact customDefFile = null;
      if (ruleContext.isAttrDefined("win_def_file", LABEL)) {
        customDefFile = ruleContext.getPrerequisiteArtifact("win_def_file", Mode.TARGET);
      }
      if (customDefFile != null) {
        dynamicLinkActionBuilder.setDefFile(customDefFile);
      }
    }

    if (!ccOutputs.getLtoBitcodeFiles().isEmpty()
        && featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)) {
      dynamicLinkActionBuilder.setLtoIndexing(true);
      dynamicLinkActionBuilder.setUsePicForLtoBackendActions(usePicForDynamicLibs);
      CppLinkAction indexAction = dynamicLinkActionBuilder.build();
      if (indexAction != null) {
        env.registerAction(indexAction);
      }

      dynamicLinkActionBuilder.setLtoIndexing(false);
    }

    CppLinkAction dynamicLinkAction = dynamicLinkActionBuilder.build();
    env.registerAction(dynamicLinkAction);

    LibraryToLink dynamicLibrary = dynamicLinkAction.getOutputLibrary();
    LibraryToLink interfaceLibrary = dynamicLinkAction.getInterfaceOutputLibrary();

    // If shared library has neverlink=1, then leave it untouched. Otherwise,
    // create a mangled symlink for it and from now on reference it through
    // mangled name only.
    //
    // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, we don't need to create the special
    // solibDir, instead we use the original interface library and dynamic library.
    if (neverlink
        || featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
      result.addDynamicLibrary(interfaceLibrary == null ? dynamicLibrary : interfaceLibrary);
      result.addExecutionDynamicLibrary(dynamicLibrary);
    } else {
      Artifact implLibraryLinkArtifact =
          SolibSymlinkAction.getDynamicLibrarySymlink(
              ruleContext,
              ccToolchain.getSolibDirectory(),
              dynamicLibrary.getArtifact(),
              /* preserveName= */ false,
              /* prefixConsumer= */ false,
              ruleContext.getConfiguration());
      LibraryToLink implLibraryLink =
          LinkerInputs.solibLibraryToLink(
              implLibraryLinkArtifact, dynamicLibrary.getArtifact(), libraryIdentifier);
      result.addExecutionDynamicLibrary(implLibraryLink);

      LibraryToLink libraryLink;
      if (interfaceLibrary == null) {
        libraryLink = implLibraryLink;
      } else {
        Artifact libraryLinkArtifact =
            SolibSymlinkAction.getDynamicLibrarySymlink(
                ruleContext,
                ccToolchain.getSolibDirectory(),
                interfaceLibrary.getArtifact(),
                /* preserveName= */ false,
                /* prefixConsumer= */ false,
                ruleContext.getConfiguration());
        libraryLink =
            LinkerInputs.solibLibraryToLink(
                libraryLinkArtifact, interfaceLibrary.getArtifact(), libraryIdentifier);
      }
      result.addDynamicLibrary(libraryLink);
    }
  }

  private CppLinkActionBuilder newLinkActionBuilder(Artifact outputArtifact) {
    return new CppLinkActionBuilder(
            ruleContext, outputArtifact, ccToolchain, fdoSupport, featureConfiguration, semantics)
        .setCrosstoolInputs(ccToolchain.getLink());
  }

  /**
   * Returns the linked artifact resulting from a linking of the given type. Consults the feature
   * configuration to obtain an action_config that provides the artifact. If the feature
   * configuration provides no artifact, uses a default.
   *
   * <p>We cannot assume that the feature configuration contains an action_config for the link
   * action, because the linux link action depends on hardcoded values in
   * LinkCommandLine.getRawLinkArgv(), which are applied on the condition that an action_config is
   * not present. TODO(b/30393154): Assert that the given link action has an action_config.
   *
   * @throws RuleErrorException
   */
  private Artifact getLinkedArtifact(LinkTargetType linkTargetType) throws RuleErrorException {
    Artifact result = null;
    Artifact linuxDefault =
        CppHelper.getLinuxLinkedArtifact(
            ruleContext, configuration, linkTargetType, linkedArtifactNameSuffix);

    try {
      String maybePicName = ruleContext.getLabel().getName() + linkedArtifactNameSuffix;
      if (linkTargetType.picness() == Picness.PIC) {
        maybePicName =
            CppHelper.getArtifactNameForCategory(
                ruleContext, ccToolchain, ArtifactCategory.PIC_FILE, maybePicName);
      }
      String linkedName =
          CppHelper.getArtifactNameForCategory(
              ruleContext, ccToolchain, linkTargetType.getLinkerOutput(), maybePicName);
      PathFragment artifactFragment =
          PathFragment.create(ruleContext.getLabel().getName())
              .getParentDirectory()
              .getRelative(linkedName);

      result =
          ruleContext.getPackageRelativeArtifact(
              artifactFragment,
              configuration.getBinDirectory(ruleContext.getRule().getRepository()));
    } catch (ExpansionException e) {
      ruleContext.throwWithRuleError(e.getMessage());
    }

    // If the linked artifact is not the linux default, then a FailAction is generated for the
    // linux default to satisfy the requirement of the implicit output.
    // TODO(b/30132703): Remove the implicit outputs of cc_library.
    if (!result.equals(linuxDefault)) {
      ruleContext.registerAction(
          new FailAction(
              ruleContext.getActionOwner(),
              ImmutableList.of(linuxDefault),
              String.format(
                  "the given toolchain supports creation of %s instead of %s",
                  linuxDefault.getExecPathString(), result.getExecPathString())));
    }

    return result;
  }
}
