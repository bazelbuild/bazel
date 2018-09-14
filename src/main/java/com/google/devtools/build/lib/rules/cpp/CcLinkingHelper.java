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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.Link.Picness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LinkingInfoApi;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/**
 * A class to create C/C++ link actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppLinkActionBuilder.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the toolchain,
 * the STL, and so on. Optionally, they can also have copts, and malloc attributes, but note that
 * these require explicit calls to the corresponding setter methods.
 */
public final class CcLinkingHelper {

  /** Contains the providers as well as the linking outputs. */
  public static final class LinkingInfo implements LinkingInfoApi {
    private final TransitiveInfoProviderMap providers;
    private final CcLinkingOutputs linkingOutputs;

    private LinkingInfo(TransitiveInfoProviderMap providers, CcLinkingOutputs linkingOutputs) {
      this.providers = providers;
      this.linkingOutputs = linkingOutputs;
    }

    public TransitiveInfoProviderMap getProviders() {
      return providers;
    }

    @Override
    public CcLinkingInfo getCcLinkingInfo() {
      return (CcLinkingInfo) providers.getProvider(CcLinkingInfo.PROVIDER.getKey());
    }

    @Override
    public CcLinkingOutputs getCcLinkingOutputs() {
      return linkingOutputs;
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
            .addAll(
                LinkerInputs.toNonSolibArtifacts(linkingOutputs.getDynamicLibrariesForLinking()))
            .addAll(
                LinkerInputs.toNonSolibArtifacts(linkingOutputs.getDynamicLibrariesForRuntime()));
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
  private final List<CcLinkingInfo> ccLinkingInfos = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkstamps = NestedSetBuilder.stableOrder();
  private final List<Artifact> linkActionInputs = new ArrayList<>();

  @Nullable private Artifact dynamicLibrary;
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverlink;

  private boolean checkDepsGenerateCpp = true;
  private boolean emitLinkActionsIfEmpty;
  private boolean emitCcNativeLibrariesProvider;
  private boolean emitInterfaceSharedObjects;
  private boolean shouldCreateDynamicLibrary = true;
  private boolean shouldCreateStaticLibraries = true;
  private boolean willOnlyBeLinkedIntoDynamicLibraries;
  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainProvider ccToolchain;
  private final FdoProvider fdoProvider;
  private String linkedArtifactNameSuffix = "";

  /**
   * Creates a CcLinkingHelper that outputs artifacts in a given configuration.
   *
   * @param ruleContext the RuleContext for the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoProvider the C++ FDO optimization support provider for the build
   * @param configuration the configuration that gives the directory of output artifacts
   */
  public CcLinkingHelper(
      RuleContext ruleContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoProvider fdoProvider,
      BuildConfiguration configuration) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = Preconditions.checkNotNull(semantics);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoProvider = Preconditions.checkNotNull(fdoProvider);
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
    Iterables.addAll(this.ccLinkingInfos, AnalysisUtils.getProviders(deps, CcLinkingInfo.PROVIDER));
    Iterables.addAll(this.deps, deps);
    return this;
  }

  /**
   * Adds additional {@link CcLinkingInfo} that will be used everywhere where CcLinkingInfos were
   * obtained from deps.
   */
  public CcLinkingHelper addCcLinkingInfos(Iterable<CcLinkingInfo> ccLinkingInfos) {
    Iterables.addAll(this.ccLinkingInfos, ccLinkingInfos);
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
   * anything other than a static link causes this class to skip the link action creation. This
   * exists only for Objective-C.
   */
  @Deprecated
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
  public CcLinkingOutputs link(CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(ccOutputs);

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
        ccLinkingOutputs = createCcLinkActions(ccOutputs);
      }
    }
    return ccLinkingOutputs;
  }

  public LinkingInfo buildLinkingProviders(
      CcLinkingOutputs ccLinkingOutputs, CcCompilationContext ccCompilationContext) {
    Preconditions.checkNotNull(ccCompilationContext);

    // Be very careful when adding new providers here - it can potentially affect a lot of rules.
    // We should consider merging most of these providers into a single provider.
    TransitiveInfoProviderMapBuilder providers = new TransitiveInfoProviderMapBuilder();

    // TODO(bazel-team): Maybe we can infer these from other data at the places where they are
    // used.
    if (emitCcNativeLibrariesProvider) {
      providers.add(new CcNativeLibraryProvider(collectNativeCcLibraries(ccLinkingOutputs)));
    }


    final CcLinkingOutputs ccLinkingOutputsFinalized = ccLinkingOutputs;
    BiFunction<Boolean, Boolean, CcLinkParams> createParams =
        (staticMode, forDynamicLibrary) -> {
          CcLinkParams.Builder builder = CcLinkParams.builder();
          builder.addLinkstamps(linkstamps.build(), ccCompilationContext);
          for (CcLinkingInfo ccLinkingInfo : ccLinkingInfos) {
            builder.addTransitiveArgs(
                ccLinkingInfo.getCcLinkParams(
                    /* staticMode= */ staticMode, /* forDynamicLibrary */ forDynamicLibrary));
          }
          if (!neverlink) {
            builder.addLibraries(
                ccLinkingOutputsFinalized.getPreferredLibraries(
                    staticMode,
                    /*preferPic=*/ forDynamicLibrary
                        || ruleContext.getFragment(CppConfiguration.class).forcePic()));
            if (!staticMode
                || (ccLinkingOutputsFinalized.getStaticLibraries().isEmpty()
                    && ccLinkingOutputsFinalized.getPicStaticLibraries().isEmpty())) {
              builder.addDynamicLibrariesForRuntime(
                  LinkerInputs.toLibraryArtifacts(
                      ccLinkingOutputsFinalized.getDynamicLibrariesForRuntime()));
            }
            builder.addLinkOpts(linkopts);
            builder.addNonCodeInputs(nonCodeLinkerInputs);
          }
          return builder.build();
        };

    CcLinkingInfo.Builder ccLinkingInfoBuilder =
        CcLinkingInfo.Builder.create()
            .setStaticModeParamsForDynamicLibrary(
                createParams.apply(/* staticMode= */ true, /* forDynamicLibrary= */ true))
            .setStaticModeParamsForExecutable(
                createParams.apply(/* staticMode= */ true, /* forDynamicLibrary= */ false))
            .setDynamicModeParamsForDynamicLibrary(
                createParams.apply(/* staticMode= */ false, /* forDynamicLibrary= */ true))
            .setDynamicModeParamsForExecutable(
                createParams.apply(/* staticMode= */ false, /* forDynamicLibrary= */ false));
    providers.put(ccLinkingInfoBuilder.build());
    return new LinkingInfo(providers.build(), ccLinkingOutputs);
  }

  private NestedSet<LinkerInput> collectNativeCcLibraries(CcLinkingOutputs ccLinkingOutputs) {
    NestedSetBuilder<LinkerInput> result = NestedSetBuilder.linkOrder();
    result.addAll(ccLinkingOutputs.getDynamicLibrariesForLinking());
    for (CcNativeLibraryProvider dep :
        AnalysisUtils.getProviders(deps, CcNativeLibraryProvider.class)) {
      result.addTransitive(dep.getTransitiveCcNativeLibraries());
    }

    return result.build();
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
  private CcLinkingOutputs createCcLinkActions(CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    // For now only handle static links. Note that the dynamic library link below ignores linkType.
    // TODO(bazel-team): Either support non-static links or move this check to setStaticLinkType().
    Preconditions.checkState(
        linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER, "can only handle static links");

    CcLinkingOutputs.Builder result = new CcLinkingOutputs.Builder();
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    boolean usePicForBinaries = CppHelper.usePicForBinaries(ruleContext, ccToolchain);
    boolean usePicForDynamicLibs = ccToolchain.usePicForDynamicLibraries();

    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    String libraryIdentifier =
        ruleContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();

    if (shouldCreateStaticLibraries) {
      Pair<LibraryToLink, LibraryToLink> staticLibrariesToLink =
          createNoPicAndPicStaticLibraries(
              env, usePicForBinaries, usePicForDynamicLibs, libraryIdentifier, ccOutputs);
      if (staticLibrariesToLink.first != null) {
        result.addStaticLibrary(staticLibrariesToLink.first);
      }
      if (staticLibrariesToLink.second != null) {
        result.addPicStaticLibrary(staticLibrariesToLink.second);
      }
    }

    if (shouldCreateDynamicLibrary) {
      createDynamicLibrary(result, env, usePicForDynamicLibs, libraryIdentifier, ccOutputs);
    }

    return result.build();
  }

  public CcLinkingHelper setWillOnlyBeLinkedIntoDynamicLibraries(
      boolean willOnlyBeLinkedIntoDynamicLibraries) {
    this.willOnlyBeLinkedIntoDynamicLibraries = willOnlyBeLinkedIntoDynamicLibraries;
    return this;
  }

  private Pair<LibraryToLink, LibraryToLink> createNoPicAndPicStaticLibraries(
      AnalysisEnvironment env,
      boolean usePicForBinaries,
      boolean usePicForDynamicLibs,
      String libraryIdentifier,
      CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    LibraryToLink staticLibrary = null;
    LibraryToLink picStaticLibrary = null;
    // Create static library (.a). The linkType only reflects whether the library is alwayslink or
    // not. The PIC-ness is determined by whether we need to use PIC or not. There are four cases
    // for (usePicForDynamicLibs usePicForBinaries):
    //
    // (1) (false false) -> no pic code is when toolchain and cppOptions don't need pic code for
    //  dynamic libraries or binaries
    // (2) (true false)  -> shared libraries as pic, but not binaries
    // (3) (true true)   -> both shared libraries and binaries as pic
    // (4) (false true) -> only pic files generated when toolchain needs pic for shared libraries
    //  and {@link #willOnlyBeLinkedIntoDynamicLibraries} is set to true.

    // In case (3), we always need PIC, so only create one static library containing the PIC
    // object files. The name therefore does not match the content.
    //
    // Presumably, it is done this way because the .a file is an implicit output of every
    // cc_library rule, so we can't use ".pic.a" that in the always-PIC case.

    // If the crosstool is configured to select an output artifact, we use that selection.
    // Otherwise, we use linux defaults.
    boolean createNoPicAction;
    boolean createPicAction;
    if (willOnlyBeLinkedIntoDynamicLibraries) {
      createNoPicAction = !usePicForDynamicLibs;
      createPicAction = usePicForDynamicLibs;
    } else {
      createNoPicAction = !usePicForBinaries;
      createPicAction = usePicForBinaries || usePicForDynamicLibs;
    }

    if (createNoPicAction) {
      staticLibrary =
          registerActionForStaticLibrary(
                  linkType, ccOutputs, /* usePic */ false, libraryIdentifier, env)
              .getOutputLibrary();
    }

    if (createPicAction) {
      LinkTargetType linkTargetTypeUsedForNaming;
      if (!createNoPicAction) {
        // Only PIC library created, name does not match content.
        linkTargetTypeUsedForNaming = linkType;
      } else {
        linkTargetTypeUsedForNaming =
            (linkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
                ? LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY
                : LinkTargetType.PIC_STATIC_LIBRARY;
      }
      picStaticLibrary =
          registerActionForStaticLibrary(
                  linkTargetTypeUsedForNaming,
                  ccOutputs,
                  /* usePic= */ true,
                  libraryIdentifier,
                  env)
              .getOutputLibrary();
    }
    return new Pair<>(staticLibrary, picStaticLibrary);
  }

  private CppLinkAction registerActionForStaticLibrary(
      LinkTargetType linkTargetTypeUsedForNaming,
      CcCompilationOutputs ccOutputs,
      boolean usePic,
      String libraryIdentifier,
      AnalysisEnvironment env)
      throws RuleErrorException, InterruptedException {
    Artifact linkedArtifact = getLinkedArtifact(linkTargetTypeUsedForNaming);
    CppLinkAction action =
        newLinkActionBuilder(linkedArtifact)
            .addObjectFiles(ccOutputs.getObjectFiles(usePic))
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addLtoBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
            .setUsePicForLtoBackendActions(usePic)
            .setLinkType(linkTargetTypeUsedForNaming)
            .setLinkingMode(LinkingMode.STATIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtensions(variablesExtensions)
            .build();
    env.registerAction(action);
    return action;
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
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
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
      result.addDynamicLibraryForLinking(
          interfaceLibrary == null ? dynamicLibrary : interfaceLibrary);
      result.addDynamicLibraryForRuntime(dynamicLibrary);
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
      result.addDynamicLibraryForRuntime(implLibraryLink);

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
      result.addDynamicLibraryForLinking(libraryLink);
    }
  }

  private CppLinkActionBuilder newLinkActionBuilder(Artifact outputArtifact) {
    return new CppLinkActionBuilder(
            ruleContext, outputArtifact, ccToolchain, fdoProvider, featureConfiguration, semantics)
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
    Artifact linuxDefault =
        CppHelper.getLinuxLinkedArtifact(
            ruleContext, configuration, linkTargetType, linkedArtifactNameSuffix);
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
