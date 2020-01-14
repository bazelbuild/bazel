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


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.Link.Picness;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LinkingInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
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
  // TODO(plf): Only used by Skylark API. Remove after migrating.
  @Deprecated
  public static final class LinkingInfo implements LinkingInfoApi {
    private final CcLinkingContext ccLinkingContext;
    private final CcLinkingOutputs linkingOutputs;

    public LinkingInfo(CcLinkingContext ccLinkingContext, CcLinkingOutputs linkingOutputs) {
      this.ccLinkingContext = ccLinkingContext;
      this.linkingOutputs = linkingOutputs;
    }

    @Override
    public CcLinkingContext getCcLinkingContext() {
      return ccLinkingContext;
    }

    @Override
    public CcLinkingOutputs getCcLinkingOutputs() {
      return linkingOutputs;
    }
  }

  private final CppSemantics semantics;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  private final List<Artifact> nonCodeLinkerInputs = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  private final List<CcLinkingContext> ccLinkingContexts = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkstamps = NestedSetBuilder.stableOrder();
  private final List<Artifact> linkActionInputs = new ArrayList<>();

  @Nullable private Artifact linkerOutputArtifact;
  private LinkTargetType staticLinkType = LinkTargetType.STATIC_LIBRARY;
  private LinkTargetType dynamicLinkType = LinkTargetType.NODEPS_DYNAMIC_LIBRARY;
  private boolean neverlink;

  private boolean emitInterfaceSharedLibraries;
  private boolean shouldCreateDynamicLibrary = true;
  private boolean shouldCreateStaticLibraries = true;
  private boolean willOnlyBeLinkedIntoDynamicLibraries;
  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();
  private boolean useTestOnlyFlags;
  private Artifact pdbFile;
  private Artifact defFile;
  private LinkingMode linkingMode = LinkingMode.DYNAMIC;
  private boolean fake;
  private boolean nativeDeps;
  private boolean wholeArchive;
  private LinkArtifactFactory linkArtifactFactory = CppLinkAction.DEFAULT_ARTIFACT_FACTORY;
  private final ImmutableList.Builder<String> additionalLinkstampDefines = ImmutableList.builder();

  private final FeatureConfiguration featureConfiguration;
  private final CcToolchainProvider ccToolchain;
  private final FdoContext fdoContext;
  private String linkedArtifactNameSuffix = "";

  private final ActionConstructionContext actionConstructionContext;
  private final Label label;
  private final ActionRegistry actionRegistry;
  private final RuleErrorConsumer ruleErrorConsumer;
  private final SymbolGenerator<?> symbolGenerator;
  private final ImmutableMap<String, String> executionInfo;

  private Artifact grepIncludes;
  private boolean isStampingEnabled;
  private boolean isTestOrTestOnlyTarget;

  /**
   * Creates a CcLinkingHelper that outputs artifacts in a given configuration.
   *
   * @param ruleErrorConsumer the RuleErrorConsumer
   * @param label the Label of the rule being built
   * @param actionRegistry the ActionRegistry of the rule being built
   * @param actionConstructionContext the ActionConstructionContext of the rule being built
   * @param semantics CppSemantics for the build
   * @param featureConfiguration activated features and action configs for the build
   * @param ccToolchain the C++ toolchain provider for the build
   * @param fdoContext the C++ FDO optimization support provider for the build
   * @param configuration the configuration that gives the directory of output artifacts
   * @param executionInfo the execution info data associated with a rule
   */
  public CcLinkingHelper(
      RuleErrorConsumer ruleErrorConsumer,
      Label label,
      ActionRegistry actionRegistry,
      ActionConstructionContext actionConstructionContext,
      CppSemantics semantics,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      BuildConfiguration configuration,
      CppConfiguration cppConfiguration,
      SymbolGenerator<?> symbolGenerator,
      ImmutableMap<String, String> executionInfo) {
    this.semantics = Preconditions.checkNotNull(semantics);
    this.featureConfiguration = Preconditions.checkNotNull(featureConfiguration);
    this.ccToolchain = Preconditions.checkNotNull(ccToolchain);
    this.fdoContext = Preconditions.checkNotNull(fdoContext);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration = cppConfiguration;
    this.ruleErrorConsumer = ruleErrorConsumer;
    this.label = label;
    this.actionRegistry = actionRegistry;
    this.actionConstructionContext = actionConstructionContext;
    this.symbolGenerator = symbolGenerator;
    this.executionInfo = executionInfo;
  }

  /** Sets fields that overlap for cc_library and cc_binary rules. */
  public CcLinkingHelper fromCommon(RuleContext ruleContext, CcCommon common) {
    addCcLinkingContexts(
        CppHelper.getLinkingContextsFromDeps(
            ImmutableList.copyOf(ruleContext.getPrerequisites("deps", Mode.TARGET))));
    addNonCodeLinkerInputs(common.getLinkerScripts());
    return this;
  }

  public CcLinkingHelper setNativeDeps(boolean nativeDeps) {
    this.nativeDeps = nativeDeps;
    return this;
  }

  public CcLinkingHelper setWholeArchive(boolean wholeArchive) {
    this.wholeArchive = wholeArchive;
    return this;
  }

  public CcLinkingHelper addAdditionalLinkstampDefines(List<String> additionalLinkstampDefines) {
    this.additionalLinkstampDefines.addAll(additionalLinkstampDefines);
    return this;
  }

  /** Adds the corresponding non-code files as linker inputs. */
  public CcLinkingHelper addNonCodeLinkerInputs(List<Artifact> nonCodeLinkerInputs) {
    for (Artifact nonCodeLinkerInput : nonCodeLinkerInputs) {
      String basename = nonCodeLinkerInput.getFilename();
      Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename));
      Preconditions.checkArgument(!Link.SHARED_LIBRARY_FILETYPES.matches(basename));
      this.nonCodeLinkerInputs.add(nonCodeLinkerInput);
    }
    return this;
  }

  /** Adds the given options as linker options to the link command. */
  public CcLinkingHelper addLinkopts(Iterable<String> linkopts) {
    Iterables.addAll(this.linkopts, linkopts);
    return this;
  }

  /**
   * Adds additional {@link CcLinkingContext} that will be used everywhere where CcLinkingInfos were
   * obtained from deps.
   */
  public CcLinkingHelper addCcLinkingContexts(Iterable<CcLinkingContext> ccLinkingContexts) {
    Iterables.addAll(this.ccLinkingContexts, ccLinkingContexts);
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
  public CcLinkingHelper setLinkerOutputArtifact(@Nullable Artifact linkerOutputArtifact) {
    this.linkerOutputArtifact = linkerOutputArtifact;
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
   *
   * @deprecated This is only set here for naming always to link static libraries with the *.lo
   *     extension. This is no longer needed because {@link
   *     com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink}s already carry
   *     information about whether a library should always be linked or not. This method will be
   *     removed when we no longer use *.lo for always to link static libraries in native
   *     cc_library.
   */
  @Deprecated
  public CcLinkingHelper setAlwayslink(boolean alwayslink) {
    staticLinkType =
        alwayslink ? LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY : LinkTargetType.STATIC_LIBRARY;
    return this;
  }

  /**
   * Directly set the link type. This can be used instead of {@link #setAlwayslink}. Setting
   * anything other than a static link causes this class to skip the link action creation. This
   * exists only for Objective-C.
   */
  public CcLinkingHelper setStaticLinkType(LinkTargetType linkType) {
    Preconditions.checkNotNull(linkType);
    Preconditions.checkState(linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER);
    this.staticLinkType = linkType;
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

  /**
   * Enables the optional generation of interface dynamic libraries - this is only used when the
   * linker generates a dynamic library, and only if the crosstool supports it. The default is not
   * to generate interface dynamic libraries.
   */
  public CcLinkingHelper emitInterfaceSharedLibraries(boolean emitInterfaceSharedLibraries) {
    this.emitInterfaceSharedLibraries = emitInterfaceSharedLibraries;
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

  /** Create the C++ link actions, and the corresponding linking related providers. */
  public CcLinkingOutputs link(CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkNotNull(ccOutputs);

    // Create link actions (only if there are object files or if explicitly requested).
    //
    // On some systems, the linker gives an error message if there are no input files. Even with
    // the check above, this can still happen if there is a .nopic.o or .o files in srcs, but no
    // other files. To fix that, we'd have to check for each link action individually.
    //
    // An additional pre-existing issue is that the header check tokens are dropped if we don't
    // generate any link actions, effectively disabling header checking in some cases.
    return createCcLinkActions(ccOutputs);
  }

  public CcLinkingContext buildCcLinkingContextFromLibrariesToLink(
      List<LibraryToLink> librariesToLink, CcCompilationContext ccCompilationContext) {
    ImmutableList.Builder<Linkstamp> linkstampBuilder = ImmutableList.builder();
    for (Artifact linkstamp : linkstamps.build().toList()) {
      linkstampBuilder.add(
          new Linkstamp(
              linkstamp,
              ccCompilationContext.getDeclaredIncludeSrcs(),
              actionConstructionContext.getActionKeyContext()));
    }
    CcLinkingContext ccLinkingContext = CcLinkingContext.EMPTY;
    if (!neverlink) {
      // We want an empty set if there are no link options. We have to make sure we don't
      // create a LinkOptions instance that contains an empty list.
      ccLinkingContext =
          CcLinkingContext.builder()
              .setOwner(label)
              .addUserLinkFlags(
                  linkopts.isEmpty()
                      ? ImmutableList.of()
                      : ImmutableList.of(
                          CcLinkingContext.LinkOptions.of(
                              ImmutableList.copyOf(linkopts), symbolGenerator)))
              .addLibraries(librariesToLink)
              .addNonCodeInputs(nonCodeLinkerInputs)
              .addLinkstamps(linkstampBuilder.build())
              .build();
    }
    ImmutableList.Builder<CcLinkingContext> mergedCcLinkingContexts = ImmutableList.builder();
    mergedCcLinkingContexts.add(ccLinkingContext);
    mergedCcLinkingContexts.addAll(ccLinkingContexts);
    return CcLinkingContext.merge(mergedCcLinkingContexts.build());
  }

  /**
   * Constructs the C++ linker actions. It generally generates two actions, one for a static library
   * and one for a dynamic library. If PIC is required for shared libraries, but not for binaries,
   * it additionally creates a third action to generate a PIC static library. If PIC is required for
   * shared libraries and binaries, then only PIC actions are registered.
   *
   * <p>For dynamic libraries, this method can additionally create an interface shared library that
   * can be used for linking, but doesn't contain any executable code. This increases the number of
   * cache hits for link actions. Call {@link #emitInterfaceSharedLibraries(boolean)} to enable this
   * behavior.
   */
  private CcLinkingOutputs createCcLinkActions(CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    // For now only handle static links. Note that the dynamic library link below ignores
    // staticLinkType.
    // TODO(bazel-team): Either support non-static links or move this check to setStaticLinkType().
    Preconditions.checkState(
        staticLinkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER,
        "can only handle static links");

    LibraryToLink.Builder libraryToLinkBuilder = LibraryToLink.builder();
    boolean usePicForBinaries =
        CppHelper.usePicForBinaries(ccToolchain, cppConfiguration, featureConfiguration);
    boolean usePicForDynamicLibs =
        ccToolchain.usePicForDynamicLibraries(cppConfiguration, featureConfiguration);

    PathFragment labelName = PathFragment.create(label.getName());
    String libraryIdentifier =
        actionConstructionContext
            .getPackageDirectory()
            .getRelative(labelName.replaceName("lib" + labelName.getBaseName()))
            .getPathString();

    if (shouldCreateStaticLibraries) {
      createNoPicAndPicStaticLibraries(
          libraryToLinkBuilder,
          usePicForBinaries,
          usePicForDynamicLibs,
          libraryIdentifier,
          ccOutputs);
    }

    boolean hasBuiltDynamicLibrary = false;
    CcLinkingOutputs.Builder ccLinkingOutputsBuilder = CcLinkingOutputs.builder();
    if (shouldCreateDynamicLibrary) {
      boolean usePic =
          (!dynamicLinkType.isExecutable() && usePicForDynamicLibs)
              || (dynamicLinkType.isExecutable() && usePicForBinaries);
      hasBuiltDynamicLibrary =
          createDynamicLinkAction(
              ccLinkingOutputsBuilder, libraryToLinkBuilder, usePic, libraryIdentifier, ccOutputs);
    }

    if (shouldCreateStaticLibraries
        && featureConfiguration.isEnabled(CppRuleClasses.DISABLE_WHOLE_ARCHIVE_FOR_STATIC_LIB)
        && (staticLinkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)) {
      ruleErrorConsumer.throwWithAttributeError(
          "alwayslink",
          "alwayslink should not be True for a target with the"
              + " disable_whole_archive_for_static_lib feature enabled.");
    }

    libraryToLinkBuilder.setDisableWholeArchive(
        featureConfiguration.isEnabled(CppRuleClasses.DISABLE_WHOLE_ARCHIVE_FOR_STATIC_LIB));

    if (hasBuiltDynamicLibrary || shouldCreateStaticLibraries) {
      ccLinkingOutputsBuilder.setLibraryToLink(libraryToLinkBuilder.build());
    }

    return ccLinkingOutputsBuilder.build();
  }

  public CcLinkingHelper setWillOnlyBeLinkedIntoDynamicLibraries(
      boolean willOnlyBeLinkedIntoDynamicLibraries) {
    this.willOnlyBeLinkedIntoDynamicLibraries = willOnlyBeLinkedIntoDynamicLibraries;
    return this;
  }

  public CcLinkingHelper setUseTestOnlyFlags(boolean useTestOnlyFlags) {
    this.useTestOnlyFlags = useTestOnlyFlags;
    return this;
  }

  /** Used to set the runfiles path for test rules' dynamic libraries. */
  public CcLinkingHelper setTestOrTestOnlyTarget(boolean isTestOrTestOnlyTarget) {
    this.isTestOrTestOnlyTarget = isTestOrTestOnlyTarget;
    return this;
  }

  /**
   * Used to test the grep-includes tool. This is needing during linking because of linkstamping.
   */
  public CcLinkingHelper setGrepIncludes(Artifact grepIncludes) {
    this.grepIncludes = grepIncludes;
    return this;
  }

  /** Whether linkstamping is enabled. */
  public CcLinkingHelper setIsStampingEnabled(boolean isStampingEnabled) {
    this.isStampingEnabled = isStampingEnabled;
    return this;
  }

  public CcLinkingHelper setLinkingMode(LinkingMode linkingMode) {
    this.linkingMode = linkingMode;
    return this;
  }

  public CcLinkingHelper setDynamicLinkType(LinkTargetType dynamicLinkType) {
    this.dynamicLinkType = dynamicLinkType;
    return this;
  }

  public CcLinkingHelper setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  public CcLinkingHelper setPdbFile(Artifact pdbFile) {
    this.pdbFile = pdbFile;
    return this;
  }

  public CcLinkingHelper setDefFile(Artifact defFile) {
    this.defFile = defFile;
    return this;
  }

  /** Needed for NativeDepsHelper. It is unclear whether native deps will be in the Starlark API. */
  @Deprecated
  public CcLinkingHelper setLinkArtifactFactory(LinkArtifactFactory linkArtifactFactory) {
    this.linkArtifactFactory = linkArtifactFactory;
    return this;
  }

  private void createNoPicAndPicStaticLibraries(
      LibraryToLink.Builder libraryToLinkBuilder,
      boolean usePicForBinaries,
      boolean usePicForDynamicLibs,
      String libraryIdentifier,
      CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    // Create static library (.a). The staticLinkType only reflects whether the library is
    // alwayslink or not. The PIC-ness is determined by whether we need to use PIC or not. There
    // are four cases:
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
      LinkerInputs.LibraryToLink staticLibrary =
          registerActionForStaticLibrary(
                  staticLinkType, ccOutputs, /* usePic= */ false, libraryIdentifier)
              .getOutputLibrary();
      libraryToLinkBuilder
          .setLibraryIdentifier(staticLibrary.getLibraryIdentifier())
          .setStaticLibrary(staticLibrary.getArtifact())
          .setObjectFiles(ImmutableList.copyOf(staticLibrary.getObjectFiles()))
          .setLtoCompilationContext(staticLibrary.getLtoCompilationContext())
          .setSharedNonLtoBackends(staticLibrary.getSharedNonLtoBackends())
          .setAlwayslink(
              staticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
    }

    if (createPicAction) {
      LinkTargetType linkTargetTypeUsedForNaming;
      if (!createNoPicAction) {
        // Only PIC library created, name does not match content.
        linkTargetTypeUsedForNaming = staticLinkType;
      } else {
        linkTargetTypeUsedForNaming =
            (staticLinkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
                ? LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY
                : LinkTargetType.PIC_STATIC_LIBRARY;
      }
      LinkerInputs.LibraryToLink picStaticLibrary =
          registerActionForStaticLibrary(
                  linkTargetTypeUsedForNaming, ccOutputs, /* usePic= */ true, libraryIdentifier)
              .getOutputLibrary();
      libraryToLinkBuilder
          .setLibraryIdentifier(picStaticLibrary.getLibraryIdentifier())
          .setPicStaticLibrary(picStaticLibrary.getArtifact())
          .setPicObjectFiles(ImmutableList.copyOf(picStaticLibrary.getObjectFiles()))
          .setPicLtoCompilationContext(picStaticLibrary.getLtoCompilationContext())
          .setPicSharedNonLtoBackends(picStaticLibrary.getSharedNonLtoBackends())
          .setAlwayslink(
              picStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
    }
  }

  private CppLinkAction registerActionForStaticLibrary(
      LinkTargetType linkTargetTypeUsedForNaming,
      CcCompilationOutputs ccOutputs,
      boolean usePic,
      String libraryIdentifier)
      throws RuleErrorException, InterruptedException {
    Artifact linkedArtifact = getLinkedArtifact(linkTargetTypeUsedForNaming);
    CppLinkAction action =
        newLinkActionBuilder(linkedArtifact, linkTargetTypeUsedForNaming)
            .addObjectFiles(ccOutputs.getObjectFiles(usePic))
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addLtoCompilationContext(ccOutputs.getLtoCompilationContext())
            .setUsePicForLtoBackendActions(usePic)
            .setLinkingMode(LinkingMode.STATIC)
            .addActionInputs(linkActionInputs)
            .setLibraryIdentifier(libraryIdentifier)
            .addVariablesExtensions(variablesExtensions)
            .build();
    actionConstructionContext.registerAction(action);
    return action;
  }

  private boolean createDynamicLinkAction(
      CcLinkingOutputs.Builder ccLinkingOutputs,
      LibraryToLink.Builder libraryToLinkBuilder,
      boolean usePic,
      String libraryIdentifier,
      CcCompilationOutputs ccOutputs)
      throws RuleErrorException, InterruptedException {
    boolean hasBuiltDynamicLibrary = false;
    // Create dynamic library.
    Artifact linkerOutput;
    String mainLibraryIdentifier;
    if (linkerOutputArtifact == null) {
      // If the crosstool is configured to select an output artifact, we use that selection.
      // Otherwise, we use linux defaults.
      linkerOutput = getLinkedArtifact(dynamicLinkType);
      mainLibraryIdentifier = libraryIdentifier;
    } else {
      // This branch is only used for vestigial Google-internal rules where the name of the output
      // file is explicitly specified in the BUILD file and as such, is platform-dependent. Thus,
      // we just hardcode some reasonable logic to compute the library identifier and hope that this
      // will eventually go away.
      linkerOutput = linkerOutputArtifact;
      mainLibraryIdentifier =
          FileSystemUtils.removeExtension(linkerOutput.getRootRelativePath().getPathString());
    }

    List<String> sonameLinkopts = ImmutableList.of();
    Artifact soInterface = null;
    if (CppHelper.useInterfaceSharedLibraries(cppConfiguration, ccToolchain, featureConfiguration)
        && emitInterfaceSharedLibraries) {
      soInterface = getLinkedArtifact(LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
      // TODO(b/28946988): Remove this hard-coded flag.
      if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
        sonameLinkopts =
            ImmutableList.of(
                "-Wl,-soname="
                    + SolibSymlinkAction.getDynamicLibrarySoname(
                        linkerOutput.getRootRelativePath(), /* preserveName= */ false));
      }
    }

    CppLinkActionBuilder dynamicLinkActionBuilder =
        newLinkActionBuilder(linkerOutput, dynamicLinkType)
            .setWholeArchive(wholeArchive)
            .setNativeDeps(nativeDeps)
            .setAdditionalLinkstampDefines(additionalLinkstampDefines.build())
            .setInterfaceOutput(soInterface)
            .addNonCodeInputs(ccOutputs.getHeaderTokenFiles())
            .addLtoCompilationContext(ccOutputs.getLtoCompilationContext())
            .setLinkingMode(linkingMode)
            .setFake(fake)
            .addActionInputs(linkActionInputs)
            .addLinkopts(linkopts)
            .addLinkopts(sonameLinkopts)
            .addNonCodeInputs(nonCodeLinkerInputs)
            .addVariablesExtensions(variablesExtensions);

    if (fake) {
      dynamicLinkActionBuilder.addFakeObjectFiles(ccOutputs.getObjectFiles(usePic));
    } else {
      dynamicLinkActionBuilder.addObjectFiles(ccOutputs.getObjectFiles(usePic));
    }

    if (!dynamicLinkType.isExecutable()) {
      dynamicLinkActionBuilder.setLibraryIdentifier(mainLibraryIdentifier);
    }

    if (linkingMode == LinkingMode.DYNAMIC) {
      try {
        dynamicLinkActionBuilder.setRuntimeInputs(
            ArtifactCategory.DYNAMIC_LIBRARY,
            ccToolchain.getDynamicRuntimeLinkMiddleman(ruleErrorConsumer, featureConfiguration),
            ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration));
      } catch (EvalException e) {
        throw ruleErrorConsumer.throwWithRuleError(e.getMessage());
      }
    } else {
      try {
        dynamicLinkActionBuilder.setRuntimeInputs(
            ArtifactCategory.STATIC_LIBRARY,
            ccToolchain.getStaticRuntimeLinkMiddleman(ruleErrorConsumer, featureConfiguration),
            ccToolchain.getStaticRuntimeLinkInputs(featureConfiguration));
      } catch (EvalException e) {
        throw ruleErrorConsumer.throwWithRuleError(e.getMessage());
      }
    }

    // On Windows, we cannot build a shared library with symbols unresolved, so here we
    // dynamically link to all its dependencies, even for LinkTargetType.NODEPS_DYNAMIC_LIBRARY.
    boolean shouldLinkTransitively =
        featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)
            || dynamicLinkType != LinkTargetType.NODEPS_DYNAMIC_LIBRARY;

    if (shouldLinkTransitively) {
      CcLinkingContext ccLinkingContext = CcLinkingContext.merge(ccLinkingContexts);
      List<LinkerInputs.LibraryToLink> libraries =
          convertLibraryToLinkListToLinkerInputList(
              ccLinkingContext.getLibraries(),
              linkingMode != LinkingMode.DYNAMIC,
              dynamicLinkType.isDynamicLibrary());
      dynamicLinkActionBuilder.addLinkParams(
          libraries,
          ccLinkingContext.getFlattenedUserLinkFlags(),
          ccLinkingContext.getLinkstamps().toList(),
          ccLinkingContext.getNonCodeInputs().toList(),
          ruleErrorConsumer);
    }

    if (pdbFile != null) {
      dynamicLinkActionBuilder.addActionOutput(pdbFile);
    }

    if (defFile != null) {
      dynamicLinkActionBuilder.setDefFile(defFile);
    }

    if (dynamicLinkActionBuilder.hasLtoBitcodeInputs()
        && featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)) {
      dynamicLinkActionBuilder.setLtoIndexing(true);
      dynamicLinkActionBuilder.setUsePicForLtoBackendActions(usePic);
      CppLinkAction indexAction = dynamicLinkActionBuilder.build();
      if (indexAction != null) {
        actionConstructionContext.registerAction(indexAction);
      }

      dynamicLinkActionBuilder.setLtoIndexing(false);
    }

    if (dynamicLinkActionBuilder.getAllLtoBackendArtifacts() != null) {
      ccLinkingOutputs.addAllLtoArtifacts(dynamicLinkActionBuilder.getAllLtoBackendArtifacts());
    }
    CppLinkAction dynamicLinkAction = dynamicLinkActionBuilder.build();
    if (dynamicLinkType.isExecutable()) {
      ccLinkingOutputs.setExecutable(linkerOutput);
    }
    if (fake) {
      ccLinkingOutputs.addLinkActionInputs(dynamicLinkAction.getInputs());
    }
    actionConstructionContext.registerAction(dynamicLinkAction);

    LinkerInputs.LibraryToLink dynamicLibrary = dynamicLinkAction.getOutputLibrary();
    LinkerInputs.LibraryToLink interfaceLibrary = dynamicLinkAction.getInterfaceOutputLibrary();

    // If shared library has neverlink=1, then leave it untouched. Otherwise,
    // create a mangled symlink for it and from now on reference it through
    // mangled name only.
    //
    // When COPY_DYNAMIC_LIBRARIES_TO_BINARY is enabled, we don't need to create the special
    // solibDir, instead we use the original interface library and dynamic library.
    if (dynamicLibrary != null) {
      hasBuiltDynamicLibrary = true;
      libraryToLinkBuilder.setLibraryIdentifier(dynamicLibrary.getLibraryIdentifier());
      if (neverlink
          || featureConfiguration.isEnabled(CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY)) {
        if (interfaceLibrary != null) {
          libraryToLinkBuilder.setInterfaceLibrary(interfaceLibrary.getArtifact());
        }
        libraryToLinkBuilder.setDynamicLibrary(dynamicLibrary.getArtifact());
      } else {
        Artifact implLibraryLinkArtifact =
            SolibSymlinkAction.getDynamicLibrarySymlink(
                /* actionRegistry= */ actionRegistry,
                /* actionConstructionContext= */ actionConstructionContext,
                ccToolchain.getSolibDirectory(),
                dynamicLibrary.getArtifact(),
                /* preserveName= */ false,
                /* prefixConsumer= */ false);
        libraryToLinkBuilder.setDynamicLibrary(implLibraryLinkArtifact);
        libraryToLinkBuilder.setResolvedSymlinkDynamicLibrary(dynamicLibrary.getArtifact());

        if (interfaceLibrary != null) {
          Artifact libraryLinkArtifact =
              SolibSymlinkAction.getDynamicLibrarySymlink(
                  /* actionRegistry= */ actionRegistry,
                  /* actionConstructionContext= */ actionConstructionContext,
                  ccToolchain.getSolibDirectory(),
                  interfaceLibrary.getArtifact(),
                  /* preserveName= */ false,
                  /* prefixConsumer= */ false);
          libraryToLinkBuilder.setInterfaceLibrary(libraryLinkArtifact);
          libraryToLinkBuilder.setResolvedSymlinkInterfaceLibrary(interfaceLibrary.getArtifact());
        }
      }
    }
    return hasBuiltDynamicLibrary;
  }

  private CppLinkActionBuilder newLinkActionBuilder(
      Artifact outputArtifact, LinkTargetType linkType) {
    return new CppLinkActionBuilder(
            ruleErrorConsumer,
            actionConstructionContext,
            label,
            outputArtifact,
            configuration,
            ccToolchain,
            fdoContext,
            featureConfiguration,
            semantics)
        .setGrepIncludes(grepIncludes)
        .setIsStampingEnabled(isStampingEnabled)
        .setTestOrTestOnlyTarget(isTestOrTestOnlyTarget)
        .setLinkType(linkType)
        .setLinkerFiles(
            (cppConfiguration.useSpecificToolFiles()
                    && linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER)
                ? ccToolchain.getArFiles()
                : ccToolchain.getLinkerFiles())
        .setLinkArtifactFactory(linkArtifactFactory)
        .setUseTestOnlyFlags(useTestOnlyFlags)
        .addExecutionInfo(executionInfo);
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
   */
  private Artifact getLinkedArtifact(LinkTargetType linkTargetType) throws RuleErrorException {
      String maybePicName = label.getName() + linkedArtifactNameSuffix;
      if (linkTargetType.picness() == Picness.PIC) {
        maybePicName =
            CppHelper.getArtifactNameForCategory(
                ruleErrorConsumer, ccToolchain, ArtifactCategory.PIC_FILE, maybePicName);
      }
      String linkedName =
          CppHelper.getArtifactNameForCategory(
              ruleErrorConsumer, ccToolchain, linkTargetType.getLinkerOutput(), maybePicName);
      PathFragment artifactFragment =
          PathFragment.create(label.getName()).getParentDirectory().getRelative(linkedName);

    return CppHelper.getLinkedArtifact(
        label,
        actionConstructionContext,
        configuration,
        linkTargetType,
        linkedArtifactNameSuffix,
        artifactFragment);
  }

  private static List<LinkerInputs.LibraryToLink> convertLibraryToLinkListToLinkerInputList(
      NestedSet<LibraryToLink> librariesToLink, boolean staticMode, boolean forDynamicLibrary) {
    ImmutableList.Builder<LinkerInputs.LibraryToLink> librariesToLinkBuilder =
        ImmutableList.builder();
    for (LibraryToLink libraryToLink : librariesToLink.toList()) {
      LinkerInputs.LibraryToLink staticLibraryToLink =
          libraryToLink.getStaticLibrary() == null ? null : libraryToLink.getStaticLibraryToLink();
      LinkerInputs.LibraryToLink picStaticLibraryToLink =
          libraryToLink.getPicStaticLibrary() == null
              ? null
              : libraryToLink.getPicStaticLibraryToLink();
      LinkerInputs.LibraryToLink libraryToLinkToUse = null;
      if (staticMode) {
        if (forDynamicLibrary) {
          if (picStaticLibraryToLink != null) {
            libraryToLinkToUse = picStaticLibraryToLink;
          } else if (staticLibraryToLink != null) {
            libraryToLinkToUse = staticLibraryToLink;
          }
        } else {
          if (staticLibraryToLink != null) {
            libraryToLinkToUse = staticLibraryToLink;
          } else if (picStaticLibraryToLink != null) {
            libraryToLinkToUse = picStaticLibraryToLink;
          }
        }
        if (libraryToLinkToUse == null) {
          if (libraryToLink.getInterfaceLibrary() != null) {
            libraryToLinkToUse = libraryToLink.getInterfaceLibraryToLink();
          } else if (libraryToLink.getDynamicLibrary() != null) {
            libraryToLinkToUse = libraryToLink.getDynamicLibraryToLink();
          }
        }
      } else {
        if (libraryToLink.getInterfaceLibrary() != null) {
          libraryToLinkToUse = libraryToLink.getInterfaceLibraryToLink();
        } else if (libraryToLink.getDynamicLibrary() != null) {
          libraryToLinkToUse = libraryToLink.getDynamicLibraryToLink();
        }
        if (libraryToLinkToUse == null) {
          if (forDynamicLibrary) {
            if (picStaticLibraryToLink != null) {
              libraryToLinkToUse = picStaticLibraryToLink;
            } else if (staticLibraryToLink != null) {
              libraryToLinkToUse = staticLibraryToLink;
            }
          } else {
            if (staticLibraryToLink != null) {
              libraryToLinkToUse = staticLibraryToLink;
            } else if (picStaticLibraryToLink != null) {
              libraryToLinkToUse = picStaticLibraryToLink;
            }
          }
        }
      }
      Preconditions.checkNotNull(libraryToLinkToUse);
      librariesToLinkBuilder.add(libraryToLinkToUse);
    }
    return librariesToLinkBuilder.build();
  }
}
