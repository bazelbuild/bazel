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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
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
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A class to create C/C++ link actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppModel.
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

  private final List<Artifact> nonCodeLinkerInputs = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkstamps = NestedSetBuilder.stableOrder();
  private final List<Artifact> linkActionInputs = new ArrayList<>();

  @Nullable private Artifact dynamicLibrary;
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverlink;
  private boolean fake;

  private final List<LibraryToLink> staticLibraries = new ArrayList<>();
  private final List<LibraryToLink> picStaticLibraries = new ArrayList<>();
  private final List<LibraryToLink> dynamicLibraries = new ArrayList<>();
  private final List<LibraryToLink> executionDynamicLibraries = new ArrayList<>();

  private boolean checkDepsGenerateCpp = true;
  private boolean emitLinkActionsIfEmpty;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitCcSpecificLinkParamsProvider;
  private boolean emitInterfaceSharedObjects;
  private boolean createDynamicLibrary = true;
  private boolean createStaticLibraries = true;
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
    Preconditions.checkState(linkType.staticness() == Staticness.STATIC);
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

  /**
   * Marks the resulting code as fake, i.e., the code will not actually be compiled or linked, but
   * instead, the compile command is written to a file and added to the runfiles. This is currently
   * used for non-compilation tests. Unfortunately, the design is problematic, so please don't add
   * any further uses.
   */
  public CcLinkingHelper setFake(boolean fake) {
    this.fake = fake;
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
  public CcLinkingHelper setCreateDynamicLibrary(boolean emitDynamicLibrary) {
    this.createDynamicLibrary = emitDynamicLibrary;
    return this;
  }

  /** When createStaticLibraries is true, there are no actions created for static libraries. */
  public CcLinkingHelper setCreateStaticLibraries(boolean emitStaticLibraries) {
    this.createStaticLibraries = emitStaticLibraries;
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
  public LinkingInfo link(
      CcCompilationOutputs ccOutputs, CppCompilationContext cppCompilationContext)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(ccOutputs);
    Preconditions.checkNotNull(cppCompilationContext);

    if (checkDepsGenerateCpp) {
      for (LanguageDependentFragment dep :
          AnalysisUtils.getProviders(deps, LanguageDependentFragment.class)) {
        LanguageDependentFragment.Checker.depSupportsLanguage(
            ruleContext, dep, CppRuleClasses.LANGUAGE, "deps");
      }
    }

    CppModel model = initializeCppModel();
    model.setContext(cppCompilationContext);

    // Create link actions (only if there are object files or if explicitly requested).
    CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
    if (emitLinkActionsIfEmpty || !ccOutputs.isEmpty()) {
      // On some systems, the linker gives an error message if there are no input files. Even with
      // the check above, this can still happen if there is a .nopic.o or .o files in srcs, but no
      // other files. To fix that, we'd have to check for each link action individually.
      //
      // An additional pre-existing issue is that the header check tokens are dropped if we don't
      // generate any link actions, effectively disabling header checking in some cases.
      if (linkType.staticness() == Staticness.STATIC) {
        // TODO(bazel-team): This can't create the link action for a cc_binary yet.
        ccLinkingOutputs = model.createCcLinkActions(ccOutputs, nonCodeLinkerInputs);
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

    Runfiles cppStaticRunfiles = collectCppRunfiles(ccLinkingOutputs, true);
    Runfiles cppSharedRunfiles = collectCppRunfiles(ccLinkingOutputs, false);

    // By very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    TransitiveInfoProviderMapBuilder providers =
        new TransitiveInfoProviderMapBuilder()
            .add(new CppRunfilesProvider(cppStaticRunfiles, cppSharedRunfiles));

    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();

    if (shouldAddLinkerOutputArtifacts(ruleContext, ccOutputs)) {
      addLinkerOutputArtifacts(outputGroups, ccOutputs);
    }

    // TODO(bazel-team): Maybe we can infer these from other data at the places where they are
    // used.
    if (emitCcNativeLibrariesProvider) {
      providers.add(new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }
    providers.put(
        CcExecutionDynamicLibrariesProvider.class,
        collectExecutionDynamicLibraryArtifacts(ccLinkingOutputs.getExecutionDynamicLibraries()));

    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean forcePic = cppConfiguration.forcePic();
    if (emitCcSpecificLinkParamsProvider) {
      providers.add(
          new CcSpecificLinkParamsProvider(
              createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    } else {
      providers.put(
          new CcLinkParamsInfo(
              createCcLinkParamsStore(ccLinkingOutputs, cppCompilationContext, forcePic)));
    }
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
              Link.LinkTargetType.DYNAMIC_LIBRARY,
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

  /** Creates the C/C++ compilation action creator. */
  private CppModel initializeCppModel() {
    // TODO(plf): Split CppModel into compilation and linking and stop passing last two arguments.
    return new CppModel(
            ruleContext,
            semantics,
            ccToolchain,
            fdoSupport,
            configuration,
            ImmutableList.of(),
            CoptsFilter.alwaysPasses())
        .setLinkTargetType(linkType)
        .setNeverLink(neverlink)
        .addLinkActionInputs(linkActionInputs)
        .setFake(fake)
        .setAllowInterfaceSharedObjects(emitInterfaceSharedObjects)
        .setCreateDynamicLibrary(createDynamicLibrary)
        .setCreateStaticLibraries(createStaticLibraries)
        // Note: this doesn't actually save the temps, it just makes the CppModel use the
        // configurations --save_temps setting to decide whether to actually save the temps.
        .setSaveTemps(true)
        .setDynamicLibrary(dynamicLibrary)
        .addLinkopts(linkopts)
        .setFeatureConfiguration(featureConfiguration)
        .addVariablesExtension(variablesExtensions)
        .setLinkedArtifactNameSuffix(linkedArtifactNameSuffix);
  }

  private Runfiles collectCppRunfiles(
      CcLinkingOutputs ccLinkingOutputs, boolean linkingStatically) {
    Runfiles.Builder builder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());
    builder.addTargets(deps, RunfilesProvider.DEFAULT_RUNFILES);
    builder.addTargets(deps, CppRunfilesProvider.runfilesFunction(linkingStatically));
    // Add the shared libraries to the runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically));
    return builder.build();
  }

  private CcLinkParamsStore createCcLinkParamsStore(
      final CcLinkingOutputs ccLinkingOutputs,
      final CppCompilationContext cppCompilationContext,
      final boolean forcePic) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(
          CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
        builder.addLinkstamps(linkstamps.build(), cppCompilationContext);
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

  private CcExecutionDynamicLibrariesProvider collectExecutionDynamicLibraryArtifacts(
      List<LibraryToLink> executionDynamicLibraries) {
    Iterable<Artifact> artifacts = LinkerInputs.toLibraryArtifacts(executionDynamicLibraries);
    if (!Iterables.isEmpty(artifacts)) {
      return new CcExecutionDynamicLibrariesProvider(
          NestedSetBuilder.wrap(Order.STABLE_ORDER, artifacts));
    }

    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (CcExecutionDynamicLibrariesProvider dep :
        AnalysisUtils.getProviders(deps, CcExecutionDynamicLibrariesProvider.class)) {
      builder.addTransitive(dep.getExecutionDynamicLibraryArtifacts());
    }
    return builder.isEmpty()
        ? CcExecutionDynamicLibrariesProvider.EMPTY
        : new CcExecutionDynamicLibrariesProvider(builder.build());
  }
}
