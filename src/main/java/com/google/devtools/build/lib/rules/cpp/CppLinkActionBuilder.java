// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CPP_LINK_EXEC_GROUP;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory;
import com.google.devtools.build.lib.rules.cpp.LibrariesToLinkCollector.CollectedLibrariesToLink;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkerOrArchiver;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import javax.annotation.Nullable;

/** Builder class to construct {@link CppLinkAction}s. */
public class CppLinkActionBuilder {

  /**
   * An implementation of {@link LinkArtifactFactory} that can create artifacts anywhere.
   *
   * <p>Necessary when the LTO backend actions of libraries should be shareable, and thus cannot be
   * under the package directory.
   */
  static final LinkArtifactFactory SHAREABLE_LINK_ARTIFACT_FACTORY =
      new LinkArtifactFactory() {
        @Override
        public Artifact create(
            ActionConstructionContext actionConstructionContext,
            RepositoryName repositoryName,
            BuildConfiguration configuration,
            PathFragment rootRelativePath) {
          return actionConstructionContext.getShareableArtifact(
              rootRelativePath, configuration.getBinDirectory(repositoryName));
        }
      };

  public static final String SHARED_NONLTO_BACKEND_ROOT_PREFIX = "shared.nonlto";

  private final Artifact output;
  private final CppSemantics cppSemantics;
  @Nullable private String mnemonic;

  // can be null for CppLinkAction.createTestBuilder()
  @Nullable private final CcToolchainProvider toolchain;
  private final FdoContext fdoContext;
  private Artifact interfaceOutput;
  /** Directory where toolchain stores language-runtime libraries (libstdc++, libc++ ...) */
  private PathFragment toolchainLibrariesSolibDir;

  protected final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;
  private FeatureConfiguration featureConfiguration;

  // Morally equivalent with {@link Context}, except these are mutable.
  // Keep these in sync with {@link Context}.
  private final Set<LinkerInput> objectFiles = new LinkedHashSet<>();
  private final Set<Artifact> nonCodeInputs = new LinkedHashSet<>();
  private final NestedSetBuilder<LinkerInputs.LibraryToLink> libraries =
      NestedSetBuilder.linkOrder();
  private final NestedSetBuilder<LibraryToLink> librariesToLink = NestedSetBuilder.linkOrder();
  private NestedSet<Artifact> linkerFiles = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private Artifact runtimeMiddleman;
  private ArtifactCategory toolchainLibrariesType = null;
  private NestedSet<Artifact> toolchainLibrariesInputs =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private final ImmutableSet.Builder<Linkstamp> linkstampsBuilder = ImmutableSet.builder();
  private ImmutableList<String> additionalLinkstampDefines = ImmutableList.of();
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private Link.LinkingMode linkingMode = LinkingMode.STATIC;
  private String libraryIdentifier = null;
  private LtoCompilationContext ltoCompilationContext;
  private Artifact defFile;

  private boolean fake;
  private boolean isNativeDeps;
  private boolean useTestOnlyFlags;
  private boolean wholeArchive;
  private boolean mustKeepDebug = false;
  private LinkArtifactFactory linkArtifactFactory = CppLinkAction.DEFAULT_ARTIFACT_FACTORY;

  private boolean isLtoIndexing = false;
  private boolean usePicForLtoBackendActions = false;
  private Iterable<LtoBackendArtifacts> allLtoArtifacts = null;

  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkActionInputs = NestedSetBuilder.stableOrder();
  private final ImmutableList.Builder<Artifact> linkActionOutputs = ImmutableList.builder();

  private final ActionConstructionContext actionConstructionContext;
  private final RuleErrorConsumer ruleErrorConsumer;
  private final RepositoryName repositoryName;

  private Artifact grepIncludes;
  // TODO(plf): This is not exactly the same as `useTestOnlyFlags` but perhaps we can remove one
  //  of them.
  private boolean isTestOrTestOnlyTarget;
  private boolean isStampingEnabled;
  private final Map<String, String> executionInfo = new LinkedHashMap<>();

  /**
   * Creates a builder that builds {@link CppLinkAction}s.
   *
   * @param ruleErrorConsumer the RuleErrorConsumer of the rule being built
   * @param actionConstructionContext the ActionConstructionContext of the rule being built
   * @param label the Label of the rule being built
   * @param output the output artifact
   * @param configuration the configuration used to determine the tool chain and the default link
   *     options
   * @param toolchain the C++ toolchain provider
   * @param fdoContext the C++ FDO optimization support
   * @param cppSemantics to be used for linkstamp compiles
   */
  public CppLinkActionBuilder(
      RuleErrorConsumer ruleErrorConsumer,
      ActionConstructionContext actionConstructionContext,
      Label label,
      Artifact output,
      BuildConfiguration configuration,
      CcToolchainProvider toolchain,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration,
      CppSemantics cppSemantics) {
    this.output = Preconditions.checkNotNull(output);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
    this.toolchain = toolchain;
    this.fdoContext = fdoContext;
    if (featureConfiguration.isEnabled(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES)) {
      toolchainLibrariesSolibDir = toolchain.getDynamicRuntimeSolibDir();
    }
    this.featureConfiguration = featureConfiguration;
    this.cppSemantics = Preconditions.checkNotNull(cppSemantics);

    this.ruleErrorConsumer = ruleErrorConsumer;
    this.actionConstructionContext = actionConstructionContext;
    repositoryName = label.getPackageIdentifier().getRepository();
  }

  /** Returns the action name for purposes of querying the crosstool. */
  private String getActionName() {
    // We check that this action is not lto-indexing, or when it is, it's either for executable
    // or transitive or nodeps dynamic library.
    Preconditions.checkArgument(
        !isLtoIndexing || linkType.isExecutable() || linkType.isDynamicLibrary());
    if (isLtoIndexing && cppConfiguration.useStandaloneLtoIndexingCommandLines()) {
      if (linkType.isExecutable()) {
        return CppActionNames.LTO_INDEX_EXECUTABLE;
      } else if (linkType.isTransitiveDynamicLibrary()) {
        return CppActionNames.LTO_INDEX_DYNAMIC_LIBRARY;
      } else {
        return CppActionNames.LTO_INDEX_NODEPS_DYNAMIC_LIBRARY;
      }
    }

    return linkType.getActionName();
  }

  /** Returns linker inputs that are not libraries. */
  public Set<LinkerInput> getObjectFiles() {
    return objectFiles;
  }

  /** Returns linker inputs that are libraries. */
  public NestedSetBuilder<LinkerInputs.LibraryToLink> getLibraries() {
    return libraries;
  }

  /** Returns linkstamps for this link action. */
  public final ImmutableSet<Linkstamp> getLinkstamps() {
    return linkstampsBuilder.build();
  }

  /** Returns command line options for this link action. */
  public final List<String> getLinkopts() {
    return this.linkopts;
  }

  /** Returns the type of this link action. */
  public LinkTargetType getLinkType() {
    return this.linkType;
  }
  /** Returns the staticness of this link action. */
  public Link.LinkingMode getLinkingMode() {
    return this.linkingMode;
  }

  /** Returns true for a cc_fake_binary. */
  public boolean isFake() {
    return this.fake;
  }

  public CppLinkActionBuilder setLinkArtifactFactory(LinkArtifactFactory linkArtifactFactory) {
    this.linkArtifactFactory = linkArtifactFactory;
    return this;
  }

  /**
   * Maps bitcode object files used by the LTO backends to the corresponding minimized bitcode file
   * used as input to the LTO indexing step.
   */
  private ImmutableSet<LinkerInput> computeLtoIndexingObjectFileInputs() {
    ImmutableSet.Builder<LinkerInput> objectFileInputsBuilder = ImmutableSet.builder();
    for (LinkerInput input : objectFiles) {
      Artifact objectFile = input.getArtifact();
      objectFileInputsBuilder.add(
          LinkerInputs.simpleLinkerInput(
              this.ltoCompilationContext.getMinimizedBitcodeOrSelf(objectFile),
              ArtifactCategory.OBJECT_FILE,
              /* disableWholeArchive= */ false));
    }
    return objectFileInputsBuilder.build();
  }

  /**
   * Maps bitcode library files used by the LTO backends to the corresponding minimized bitcode file
   * used as input to the LTO indexing step.
   */
  private static NestedSet<LinkerInputs.LibraryToLink> computeLtoIndexingUniqueLibraries(
      NestedSet<LinkerInputs.LibraryToLink> originalUniqueLibraries,
      boolean includeLinkStaticInLtoIndexing) {
    NestedSetBuilder<LinkerInputs.LibraryToLink> uniqueLibrariesBuilder =
        NestedSetBuilder.linkOrder();
    for (LinkerInputs.LibraryToLink lib : originalUniqueLibraries.toList()) {
      if (!lib.containsObjectFiles()) {
        uniqueLibrariesBuilder.add(lib);
        continue;
      }
      ImmutableSet.Builder<Artifact> newObjectFilesBuilder = ImmutableSet.builder();
      for (Artifact a : lib.getObjectFiles()) {
        // If this link includes object files from another library, that library must be
        // statically linked.
        if (!includeLinkStaticInLtoIndexing) {
          Preconditions.checkNotNull(lib.getSharedNonLtoBackends());
          LtoBackendArtifacts ltoArtifacts = lib.getSharedNonLtoBackends().getOrDefault(a, null);
          // Either we have a shared LTO artifact, or this wasn't bitcode to start with.
          Preconditions.checkState(
              ltoArtifacts != null || !lib.getLtoCompilationContext().containsBitcodeFile(a));
          if (ltoArtifacts != null) {
            // Include the native object produced by the shared LTO backend in the LTO indexing
            // step instead of the bitcode file. The LTO indexing step invokes the linker which
            // must see all objects used to produce the final link output.
            newObjectFilesBuilder.add(ltoArtifacts.getObjectFile());
            continue;
          }
        }
        newObjectFilesBuilder.add(lib.getLtoCompilationContext().getMinimizedBitcodeOrSelf(a));
      }
      uniqueLibrariesBuilder.add(
          LinkerInputs.newInputLibrary(
              lib.getArtifact(),
              lib.getArtifactCategory(),
              lib.getLibraryIdentifier(),
              newObjectFilesBuilder.build(),
              lib.getLtoCompilationContext(),
              /* sharedNonLtoBackends= */ null,
              /* mustKeepDebug= */ false));
    }
    return uniqueLibrariesBuilder.build();
  }

  /**
   * Returns true if there are any LTO bitcode inputs to this link, either directly transitively via
   * library inputs.
   */
  public boolean hasLtoBitcodeInputs() {
    if (!ltoCompilationContext.isEmpty()) {
      return true;
    }
    for (LinkerInputs.LibraryToLink lib : libraries.build().toList()) {
      if (!lib.getLtoCompilationContext().isEmpty()) {
        return true;
      }
    }
    return false;
  }

  /*
   * Create an LtoBackendArtifacts object, using the appropriate constructor depending on whether
   * the associated ThinLTO link will utilize LTO indexing (therefore unique LTO backend actions),
   * or not (and therefore the library being linked will create a set of shared LTO backends).
   */
  private LtoBackendArtifacts createLtoArtifact(
      Artifact bitcodeFile,
      @Nullable BitcodeFiles allBitcode,
      PathFragment ltoOutputRootPrefix,
      boolean createSharedNonLto,
      List<String> argv)
      throws RuleErrorException {
    // Depending on whether LTO indexing is allowed, generate an LTO backend
    // that will be fed the results of the indexing step, or a dummy LTO backend
    // that simply compiles the bitcode into native code without any index-based
    // cross module optimization.
    LtoBackendArtifacts ltoArtifact =
        createSharedNonLto
            ? new LtoBackendArtifacts(
                ruleErrorConsumer,
                configuration.getOptions(),
                cppConfiguration,
                ltoOutputRootPrefix,
                bitcodeFile,
                actionConstructionContext,
                repositoryName,
                configuration,
                SHAREABLE_LINK_ARTIFACT_FACTORY,
                featureConfiguration,
                toolchain,
                fdoContext,
                usePicForLtoBackendActions,
                toolchain.shouldCreatePerObjectDebugInfo(featureConfiguration, cppConfiguration),
                argv)
            : new LtoBackendArtifacts(
                ruleErrorConsumer,
                configuration.getOptions(),
                cppConfiguration,
                ltoOutputRootPrefix,
                bitcodeFile,
                allBitcode,
                actionConstructionContext,
                repositoryName,
                configuration,
                linkArtifactFactory,
                featureConfiguration,
                toolchain,
                fdoContext,
                usePicForLtoBackendActions,
                toolchain.shouldCreatePerObjectDebugInfo(featureConfiguration, cppConfiguration),
                argv);
    return ltoArtifact;
  }

  private ImmutableList<String> collectPerFileLtoBackendOpts(Artifact objectFile) {
    return cppConfiguration.getPerFileLtoBackendOpts().stream()
        .filter(perLabelOptions -> perLabelOptions.isIncluded(objectFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(ImmutableList.toImmutableList());
  }

  private List<String> getLtoBackendUserCompileFlags(
      Artifact objectFile, ImmutableList<String> copts) {
    List<String> argv = new ArrayList<>();
    argv.addAll(cppConfiguration.getLinkopts());
    argv.addAll(copts);
    argv.addAll(cppConfiguration.getLtoBackendOptions());
    argv.addAll(collectPerFileLtoBackendOpts(objectFile));
    return argv;
  }

  private Iterable<LtoBackendArtifacts> createLtoArtifacts(
      PathFragment ltoOutputRootPrefix,
      NestedSet<LinkerInputs.LibraryToLink> uniqueLibraries,
      boolean allowLtoIndexing,
      boolean includeLinkStaticInLtoIndexing)
      throws RuleErrorException {
    Set<Artifact> compiled = new LinkedHashSet<>();
    for (LinkerInputs.LibraryToLink lib : uniqueLibraries.toList()) {
      compiled.addAll(lib.getLtoCompilationContext().getBitcodeFiles());
    }

    // Make this a NestedSet to return from LtoBackendAction.getAllowedDerivedInputs. For M binaries
    // and N .o files, this is O(M*N). If we had nested sets of bitcode files, it would be O(M + N).
    NestedSetBuilder<Artifact> allBitcode = NestedSetBuilder.stableOrder();
    // Since this link includes object files from another library, we know that library must be
    // statically linked, so we need to look at includeLinkStaticInLtoIndexing to decide whether
    // to include its objects in the LTO indexing for this target.
    if (includeLinkStaticInLtoIndexing) {
      for (LinkerInputs.LibraryToLink lib : uniqueLibraries.toList()) {
        if (!lib.containsObjectFiles()) {
          continue;
        }
        for (Artifact objectFile : lib.getObjectFiles()) {
          if (compiled.contains(objectFile)) {
            allBitcode.add(objectFile);
          }
        }
      }
    }
    for (LinkerInput input : objectFiles) {
      if (this.ltoCompilationContext.containsBitcodeFile(input.getArtifact())) {
        allBitcode.add(input.getArtifact());
      }
    }
    BitcodeFiles bitcodeFiles = new BitcodeFiles(allBitcode.build());

    ImmutableList.Builder<LtoBackendArtifacts> ltoOutputs = ImmutableList.builder();
    for (LinkerInputs.LibraryToLink lib : uniqueLibraries.toList()) {
      if (!lib.containsObjectFiles()) {
        continue;
      }
      // We will create new LTO backends whenever we are performing LTO indexing, in which case
      // each target linking this library needs a unique set of LTO backends.
      for (Artifact objectFile : lib.getObjectFiles()) {
        if (compiled.contains(objectFile)) {
          if (includeLinkStaticInLtoIndexing) {
            List<String> backendUserCompileFlags =
                getLtoBackendUserCompileFlags(
                    objectFile, lib.getLtoCompilationContext().getCopts(objectFile));
            LtoBackendArtifacts ltoArtifacts =
                createLtoArtifact(
                    objectFile,
                    bitcodeFiles,
                    ltoOutputRootPrefix,
                    /* createSharedNonLto= */ false,
                    backendUserCompileFlags);
            ltoOutputs.add(ltoArtifacts);
          } else {
            // We should have created shared LTO backends when the library was created.
            Preconditions.checkNotNull(lib.getSharedNonLtoBackends());
            LtoBackendArtifacts ltoArtifacts =
                lib.getSharedNonLtoBackends().getOrDefault(objectFile, null);
            Preconditions.checkNotNull(ltoArtifacts);
            ltoOutputs.add(ltoArtifacts);
          }
        }
      }
    }
    for (LinkerInput input : objectFiles) {
      if (this.ltoCompilationContext.containsBitcodeFile(input.getArtifact())) {
        List<String> backendUserCompileFlags =
            getLtoBackendUserCompileFlags(
                input.getArtifact(), this.ltoCompilationContext.getCopts(input.getArtifact()));
        LtoBackendArtifacts ltoArtifacts =
            createLtoArtifact(
                input.getArtifact(),
                bitcodeFiles,
                ltoOutputRootPrefix,
                !allowLtoIndexing,
                backendUserCompileFlags);
        ltoOutputs.add(ltoArtifacts);
      }
    }

    return ltoOutputs.build();
  }

  private ImmutableMap<Artifact, LtoBackendArtifacts> createSharedNonLtoArtifacts(
      boolean isLtoIndexing) throws RuleErrorException {
    // Only create the shared LTO artifacts for a statically linked library that has bitcode files.
    if (ltoCompilationContext == null
        || isLtoIndexing
        || linkType.linkerOrArchiver() != LinkerOrArchiver.ARCHIVER) {
      return ImmutableMap.<Artifact, LtoBackendArtifacts>of();
    }

    PathFragment ltoOutputRootPrefix = PathFragment.create(SHARED_NONLTO_BACKEND_ROOT_PREFIX);

    ImmutableMap.Builder<Artifact, LtoBackendArtifacts> sharedNonLtoBackends =
        ImmutableMap.builder();

    for (LinkerInput input : objectFiles) {
      if (this.ltoCompilationContext.containsBitcodeFile(input.getArtifact())) {
        List<String> backendUserCompileFlags =
            getLtoBackendUserCompileFlags(
                input.getArtifact(), this.ltoCompilationContext.getCopts(input.getArtifact()));
        LtoBackendArtifacts ltoArtifacts =
            createLtoArtifact(
                input.getArtifact(),
                /* allBitcode= */ null,
                ltoOutputRootPrefix,
                /* createSharedNonLto= */ true,
                backendUserCompileFlags);
        sharedNonLtoBackends.put(input.getArtifact(), ltoArtifacts);
      }
    }

    return sharedNonLtoBackends.build();
  }

  @VisibleForTesting
  boolean canSplitCommandLine() {
    if (fake) {
      return false;
    }

    if (toolchain == null || !toolchain.supportsParamFiles()) {
      return false;
    }

    switch (linkType) {
        // On Unix, we currently can't split dynamic library links if they have interface outputs.
        // That was probably an unintended side effect of the change that introduced interface
        // outputs.
        // On Windows, We can always split the command line when building DLL.
      case NODEPS_DYNAMIC_LIBRARY:
      case DYNAMIC_LIBRARY:
        return (interfaceOutput == null
            || featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS));
      case EXECUTABLE:
      case STATIC_LIBRARY:
      case PIC_STATIC_LIBRARY:
      case ALWAYS_LINK_STATIC_LIBRARY:
      case ALWAYS_LINK_PIC_STATIC_LIBRARY:
        return true;

      default:
        return false;
    }
  }

  private ImmutableList<LinkerInputs.LibraryToLink> collectLinkerInputs(
      NestedSet<LibraryToLink> librariesToLink) {
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
      if (linkingMode == LinkingMode.STATIC) {
        if (linkType.isDynamicLibrary()) {
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
      if (libraryToLinkToUse == null) {
        if (libraryToLink.getInterfaceLibrary() != null) {
          libraryToLinkToUse = libraryToLink.getInterfaceLibraryToLink();
        } else if (libraryToLink.getDynamicLibrary() != null) {
          libraryToLinkToUse = libraryToLink.getDynamicLibraryToLink();
        }
      }
      Preconditions.checkNotNull(libraryToLinkToUse);
      checkLibrary(libraryToLinkToUse);
      librariesToLinkBuilder.add(libraryToLinkToUse);
    }
    return librariesToLinkBuilder.build();
  }

  /** Builds the Action as configured and returns it. */
  public CppLinkAction build() throws InterruptedException, RuleErrorException {
    NestedSet<LinkerInputs.LibraryToLink> originalUniqueLibraries = null;

    if (librariesToLink.isEmpty()) {
      originalUniqueLibraries = libraries.build();
    } else {
      Preconditions.checkState(libraries.isEmpty());
      originalUniqueLibraries =
          NestedSetBuilder.<LinkerInputs.LibraryToLink>linkOrder()
              .addAll(collectLinkerInputs(librariesToLink.build()))
              .build();
    }

    // Executable links do not have library identifiers.
    boolean hasIdentifier = (libraryIdentifier != null);
    boolean isExecutable = linkType.isExecutable();
    Preconditions.checkState(hasIdentifier != isExecutable);
    Preconditions.checkNotNull(featureConfiguration);
    ImmutableSet<Linkstamp> linkstamps = linkstampsBuilder.build();
    final ImmutableMap<Linkstamp, Artifact> linkstampMap =
        mapLinkstampsToOutputs(
            linkstamps,
            actionConstructionContext,
            repositoryName,
            configuration,
            output,
            linkArtifactFactory);

    if (interfaceOutput != null && (fake || !linkType.isDynamicLibrary())) {
      throw new RuntimeException(
          "Interface output can only be used " + "with non-fake DYNAMIC_LIBRARY targets");
    }

    if (!featureConfiguration.actionIsConfigured(linkType.getActionName())) {
      ruleErrorConsumer.ruleError(
          String.format(
              "Expected action_config for '%s' to be configured", linkType.getActionName()));
    }

    final ImmutableList<Artifact> buildInfoHeaderArtifacts =
        !linkstamps.isEmpty()
            ? actionConstructionContext
                .getAnalysisEnvironment()
                .getBuildInfo(isStampingEnabled, CppBuildInfo.KEY, configuration)
            : ImmutableList.of();

    boolean needWholeArchive =
        wholeArchive
            || needWholeArchive(
                featureConfiguration, linkingMode, linkType, linkopts, cppConfiguration);
    // Disallow LTO indexing for test targets that link statically, and optionally for any
    // linkstatic target (which can be used to disable LTO indexing for non-testonly cc_binary
    // built due to data dependences for a blaze test invocation). Otherwise this will provoke
    // Blaze OOM errors in the case where multiple static tests are invoked together,
    // since each target needs a separate set of LTO Backend actions. With dynamic linking,
    // the targest share the dynamic libraries which were produced via smaller subsets of
    // LTO indexing/backends. ThinLTO on the tests will be different than the ThinLTO
    // optimizations applied to the associated main binaries anyway.
    // Even for dynamically linked tests, disallow linkstatic libraries from participating
    // in the test's LTO indexing step for similar reasons.
    boolean canIncludeAnyLinkStaticInLtoIndexing =
        !featureConfiguration.isEnabled(
            CppRuleClasses.THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS);
    boolean canIncludeAnyLinkStaticTestTargetInLtoIndexing =
        !featureConfiguration.isEnabled(
            CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS);
    boolean includeLinkStaticInLtoIndexing =
        canIncludeAnyLinkStaticInLtoIndexing
            && (canIncludeAnyLinkStaticTestTargetInLtoIndexing || !isTestOrTestOnlyTarget);
    boolean allowLtoIndexing =
        includeLinkStaticInLtoIndexing
            || (linkingMode == Link.LinkingMode.DYNAMIC && !ltoCompilationContext.isEmpty());

    PathFragment ltoOutputRootPrefix = null;
    if (isLtoIndexing) {
      Preconditions.checkState(allLtoArtifacts == null);
      ltoOutputRootPrefix =
          allowLtoIndexing
              ? FileSystemUtils.appendExtension(output.getRootRelativePath(), ".lto")
              : PathFragment.create(SHARED_NONLTO_BACKEND_ROOT_PREFIX);
      // Use the originalUniqueLibraries which contains the full bitcode files
      // needed by the LTO backends (as opposed to the minimized bitcode files
      // containing just the summaries and symbol information that can be used by
      // the LTO indexing step).
      allLtoArtifacts =
          createLtoArtifacts(
              ltoOutputRootPrefix,
              originalUniqueLibraries,
              allowLtoIndexing,
              includeLinkStaticInLtoIndexing);

      if (!allowLtoIndexing) {
        return null;
      }
    }

    // Get the set of object files and libraries containing the correct
    // inputs for this link, depending on whether this is LTO indexing or
    // a native link.
    NestedSet<LinkerInputs.LibraryToLink> uniqueLibraries;
    ImmutableSet<LinkerInput> objectFileInputs;
    ImmutableSet<LinkerInput> linkstampObjectFileInputs;
    if (isLtoIndexing) {
      objectFileInputs = computeLtoIndexingObjectFileInputs();
      uniqueLibraries =
          computeLtoIndexingUniqueLibraries(
              originalUniqueLibraries, includeLinkStaticInLtoIndexing);
      linkstampObjectFileInputs = ImmutableSet.of();
    } else {
      objectFileInputs = ImmutableSet.copyOf(objectFiles);
      linkstampObjectFileInputs =
          ImmutableSet.copyOf(LinkerInputs.linkstampLinkerInputs(linkstampMap.values()));
      uniqueLibraries = originalUniqueLibraries;
    }

    Map<Artifact, Artifact> ltoMapping = new HashMap<>();

    if (isFinalLinkOfLtoBuild()) {
      for (LtoBackendArtifacts a : allLtoArtifacts) {
        ltoMapping.put(a.getBitcodeFile(), a.getObjectFile());
      }
    }
    NestedSet<Artifact> objectArtifacts =
        getArtifactsPossiblyLtoMapped(objectFileInputs, ltoMapping);
    NestedSet<Artifact> linkstampObjectArtifacts =
        getArtifactsPossiblyLtoMapped(linkstampObjectFileInputs, ltoMapping);

    ImmutableSet<Artifact> combinedObjectArtifacts =
        ImmutableSet.<Artifact>builder()
            .addAll(objectArtifacts.toList())
            .addAll(linkstampObjectArtifacts.toList())
            .build();
    final LinkerInputs.LibraryToLink outputLibrary =
        linkType.isExecutable()
            ? null
            : LinkerInputs.newInputLibrary(
                output,
                linkType.getLinkerOutput(),
                libraryIdentifier,
                linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER
                    ? combinedObjectArtifacts
                    : ImmutableSet.of(),
                linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER
                    ? ltoCompilationContext
                    : LtoCompilationContext.EMPTY,
                createSharedNonLtoArtifacts(isLtoIndexing),
                /* mustKeepDebug= */ false);
    final LinkerInputs.LibraryToLink interfaceOutputLibrary =
        (interfaceOutput == null)
            ? null
            : LinkerInputs.newInputLibrary(
                interfaceOutput,
                ArtifactCategory.DYNAMIC_LIBRARY,
                libraryIdentifier,
                combinedObjectArtifacts,
                ltoCompilationContext,
                /* sharedNonLtoBackends= */ null,
                /* mustKeepDebug= */ false);

    @Nullable Artifact thinltoParamFile = null;
    @Nullable Artifact thinltoMergedObjectFile = null;
    PathFragment outputRootPath = output.getRootRelativePath();
    if (allowLtoIndexing && allLtoArtifacts != null) {
      // Create artifact for the file that the LTO indexing step will emit
      // object file names into for any that were included in the link as
      // determined by the linker's symbol resolution. It will be used to
      // provide the inputs for the subsequent final native object link.
      // Note that the paths emitted into this file will have their prefixes
      // replaced with the final output directory, so they will be the paths
      // of the native object files not the input bitcode files.
      PathFragment linkerParamFileRootPath = ParameterFile.derivePath(outputRootPath, "lto-final");
      thinltoParamFile =
          linkArtifactFactory.create(
              actionConstructionContext, repositoryName, configuration, linkerParamFileRootPath);

      // Create artifact for the merged object file, which is an object file that is created
      // during the LTO indexing step and needs to be passed to the final link.
      PathFragment thinltoMergedObjectFileRootPath =
          outputRootPath.replaceName(outputRootPath.getBaseName() + ".lto.merged.o");
      thinltoMergedObjectFile =
          linkArtifactFactory.create(
              actionConstructionContext,
              repositoryName,
              configuration,
              thinltoMergedObjectFileRootPath);
    }

    final ImmutableSet<Artifact> actionOutputs;
    if (isLtoIndexing) {
      ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();
      for (LtoBackendArtifacts ltoA : allLtoArtifacts) {
        ltoA.addIndexingOutputs(builder);
      }
      if (thinltoParamFile != null) {
        builder.add(thinltoParamFile);
      }
      if (thinltoMergedObjectFile != null) {
        builder.add(thinltoMergedObjectFile);
        addObjectFile(thinltoMergedObjectFile);
      }
      actionOutputs = builder.build();
    } else {
      actionOutputs =
          constructOutputs(
              output,
              linkActionOutputs.build(),
              interfaceOutputLibrary == null ? null : interfaceOutputLibrary.getArtifact());
    }

    // Linker inputs without any start/end lib expansions.
    final Iterable<LinkerInput> nonExpandedLinkerInputs =
        IterablesChain.<LinkerInput>builder()
            .add(objectFileInputs)
            .add(linkstampObjectFileInputs)
            .add(uniqueLibraries.toList())
            .add(
                // Adding toolchain libraries without whole archive no-matter-what. People don't
                // want to include whole libstdc++ in their binary ever.
                ImmutableSet.copyOf(
                    LinkerInputs.simpleLinkerInputs(
                        toolchainLibrariesInputs.toList(),
                        toolchainLibrariesType,
                        /* disableWholeArchive= */ true)))
            .build();

    PathFragment paramRootPath =
        ParameterFile.derivePath(outputRootPath, (isLtoIndexing) ? "lto-index" : "2");

    @Nullable
    final Artifact paramFile =
        canSplitCommandLine()
            ? linkArtifactFactory.create(
                actionConstructionContext, repositoryName, configuration, paramRootPath)
            : null;

    // Add build variables necessary to template link args into the crosstool.
    CcToolchainVariables.Builder buildVariablesBuilder =
        CcToolchainVariables.builder(
            toolchain.getBuildVariables(configuration.getOptions(), cppConfiguration));
    Preconditions.checkState(!isLtoIndexing || allowLtoIndexing);
    Preconditions.checkState(allowLtoIndexing || thinltoParamFile == null);
    Preconditions.checkState(allowLtoIndexing || thinltoMergedObjectFile == null);
    PathFragment solibDir =
        configuration
            .getBinDirectory(repositoryName)
            .getExecPath()
            .getRelative(toolchain.getSolibDirectory());
    LibrariesToLinkCollector librariesToLinkCollector =
        new LibrariesToLinkCollector(
            isNativeDeps,
            cppConfiguration,
            toolchain,
            toolchainLibrariesSolibDir,
            linkType,
            linkingMode,
            output,
            solibDir,
            isLtoIndexing,
            allLtoArtifacts,
            featureConfiguration,
            thinltoParamFile,
            allowLtoIndexing,
            nonExpandedLinkerInputs,
            needWholeArchive);
    CollectedLibrariesToLink collectedLibrariesToLink =
        librariesToLinkCollector.collectLibrariesToLink();

    NestedSet<Artifact> expandedLinkerArtifacts =
        getArtifactsPossiblyLtoMapped(
            collectedLibrariesToLink.getExpandedLinkerInputs().toList(), ltoMapping);

    CcToolchainVariables variables;
    try {
      ImmutableList.Builder<String> userLinkFlags =
          ImmutableList.<String>builder().addAll(linkopts).addAll(cppConfiguration.getLinkopts());

      if (isLtoIndexing && cppConfiguration.useStandaloneLtoIndexingCommandLines()) {
        userLinkFlags.addAll(cppConfiguration.getLtoIndexOptions());
      }

      variables =
          LinkBuildVariables.setupVariables(
              getLinkType().linkerOrArchiver().equals(LinkerOrArchiver.LINKER),
              configuration.getBinDirectory().getExecPath(),
              output.getExecPathString(),
              linkType.equals(LinkTargetType.DYNAMIC_LIBRARY),
              paramFile != null ? paramFile.getExecPathString() : null,
              thinltoParamFile != null ? thinltoParamFile.getExecPathString() : null,
              thinltoMergedObjectFile != null ? thinltoMergedObjectFile.getExecPathString() : null,
              mustKeepDebug,
              toolchain,
              cppConfiguration,
              configuration.getOptions(),
              featureConfiguration,
              useTestOnlyFlags,
              isLtoIndexing,
              userLinkFlags.build(),
              toolchain.getInterfaceSoBuilder().getExecPathString(),
              interfaceOutput != null ? interfaceOutput.getExecPathString() : null,
              ltoOutputRootPrefix,
              defFile != null ? defFile.getExecPathString() : null,
              fdoContext,
              collectedLibrariesToLink.getRuntimeLibrarySearchDirectories(),
              collectedLibrariesToLink.getLibrariesToLink(),
              collectedLibrariesToLink.getLibrarySearchDirectories(),
              /* addIfsoRelatedVariables= */ true);
    } catch (EvalException e) {
      ruleErrorConsumer.ruleError(e.getMessage());
      variables = CcToolchainVariables.EMPTY;
    }
    buildVariablesBuilder.addAllNonTransitive(variables);
    for (VariablesExtension extraVariablesExtension : variablesExtensions) {
      extraVariablesExtension.addVariables(buildVariablesBuilder);
    }

    CcToolchainVariables buildVariables = buildVariablesBuilder.build();

    Preconditions.checkArgument(
        linkType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
        "you can't link an interface dynamic library directly");
    if (!linkType.isDynamicLibrary()) {
      Preconditions.checkArgument(
          interfaceOutput == null,
          "interface output may only be non-null for dynamic library links");
    }
    if (linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER) {
      // solib dir must be null for static links
      toolchainLibrariesSolibDir = null;

      Preconditions.checkArgument(
          linkingMode == Link.LinkingMode.STATIC, "static library link must be static");
      Preconditions.checkArgument(
          !isNativeDeps, "the native deps flag must be false for static links");
      Preconditions.checkArgument(
          !needWholeArchive, "the need whole archive flag must be false for static links");
    }

    LinkCommandLine.Builder linkCommandLineBuilder =
        new LinkCommandLine.Builder()
            .setActionName(getActionName())
            .setLinkerInputArtifacts(
                NestedSetBuilder.<Artifact>stableOrder()
                    .addTransitive(expandedLinkerArtifacts)
                    .addTransitive(linkstampObjectArtifacts)
                    .build())
            .setLinkTargetType(linkType)
            .setLinkingMode(linkingMode)
            .setToolchainLibrariesSolibDir(
                linkType.linkerOrArchiver() == LinkerOrArchiver.ARCHIVER
                    ? null
                    : toolchainLibrariesSolibDir)
            .setNativeDeps(isNativeDeps)
            .setUseTestOnlyFlags(useTestOnlyFlags)
            .setParamFile(paramFile)
            .setFeatureConfiguration(featureConfiguration);

    // TODO(b/62693279): Cleanup once internal crosstools specify ifso building correctly.
    if (shouldUseLinkDynamicLibraryTool()) {
      linkCommandLineBuilder.forceToolPath(
          toolchain.getLinkDynamicLibraryTool().getExecPathString());
    }

    if (!isLtoIndexing) {
      linkCommandLineBuilder.setBuildInfoHeaderArtifacts(buildInfoHeaderArtifacts);
    }

    linkCommandLineBuilder.setBuildVariables(buildVariables);
    if (CppHelper.doNotSplitLinkingCmdLine(
        actionConstructionContext.getAnalysisEnvironment().getStarlarkSemantics(), toolchain)) {
      linkCommandLineBuilder.doNotSplitLinkingCmdLine();
    }
    LinkCommandLine linkCommandLine = linkCommandLineBuilder.build();

    // Compute the set of inputs - we only need stable order here.
    NestedSetBuilder<Artifact> dependencyInputsBuilder = NestedSetBuilder.stableOrder();
    dependencyInputsBuilder.addTransitive(linkerFiles);
    dependencyInputsBuilder.addTransitive(linkActionInputs.build());
    // TODO(b/62693279): Cleanup once internal crosstools specify ifso building correctly.
    if (shouldUseLinkDynamicLibraryTool()) {
      dependencyInputsBuilder.add(toolchain.getLinkDynamicLibraryTool());
    }
    if (runtimeMiddleman != null) {
      dependencyInputsBuilder.add(runtimeMiddleman);
    }
    if (defFile != null) {
      dependencyInputsBuilder.add(defFile);
    }

    NestedSet<Artifact> nonCodeInputsAsNestedSet =
        NestedSetBuilder.wrap(Order.STABLE_ORDER, nonCodeInputs);

    // getPrimaryInput returns the first element, and that is a public interface - therefore the
    // order here is important.
    NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(expandedLinkerArtifacts)
            .addTransitive(nonCodeInputsAsNestedSet)
            .addTransitive(dependencyInputsBuilder.build());

    if (thinltoParamFile != null && !isLtoIndexing) {
      inputsBuilder.add(thinltoParamFile);
    }
    if (linkCommandLine.getParamFile() != null) {
      inputsBuilder.add(linkCommandLine.getParamFile());
      // Pass along tree artifacts, so they can be properly expanded.
      NestedSet<Artifact> paramFileActionInputs =
          NestedSetBuilder.wrap(
              Order.STABLE_ORDER,
              Iterables.filter(expandedLinkerArtifacts.toList(), Artifact::isTreeArtifact));

      Action parameterFileWriteAction =
          new ParameterFileWriteAction(
              getOwner(),
              paramFileActionInputs,
              paramFile,
              linkCommandLine.paramCmdLine(),
              ParameterFile.ParameterFileType.UNQUOTED);
      actionConstructionContext.registerAction(parameterFileWriteAction);
    }

    ImmutableMap<String, String> toolchainEnv =
        CppHelper.getEnvironmentVariables(
            ruleErrorConsumer, featureConfiguration, buildVariables, getActionName());

    // If the crosstool uses action_configs to configure cc compilation, collect execution info
    // from there, otherwise, use no execution info.
    // TODO(b/27903698): Assert that the crosstool has an action_config for this action.

    if (featureConfiguration.actionIsConfigured(getActionName())) {
      for (String req : featureConfiguration.getToolRequirementsForAction(getActionName())) {
        executionInfo.put(req, "");
      }
    }
    configuration.modifyExecutionInfo(
        executionInfo, CppLinkAction.getMnemonic(mnemonic, isLtoIndexing));

    if (!isLtoIndexing) {
      for (Map.Entry<Linkstamp, Artifact> linkstampEntry : linkstampMap.entrySet()) {
        actionConstructionContext.registerAction(
            CppLinkstampCompileHelper.createLinkstampCompileAction(
                ruleErrorConsumer,
                actionConstructionContext,
                grepIncludes,
                configuration,
                linkstampEntry.getKey().getArtifact(),
                linkstampEntry.getValue(),
                linkstampEntry.getKey().getDeclaredIncludeSrcs(),
                nonCodeInputsAsNestedSet,
                inputsBuilder.build(),
                buildInfoHeaderArtifacts,
                additionalLinkstampDefines,
                toolchain,
                configuration.isCodeCoverageEnabled(),
                cppConfiguration,
                CppHelper.getFdoBuildStamp(cppConfiguration, fdoContext, featureConfiguration),
                featureConfiguration,
                cppConfiguration.forcePic()
                    || (linkType.isDynamicLibrary()
                        && toolchain.usePicForDynamicLibraries(
                            cppConfiguration, featureConfiguration)),
                Matcher.quoteReplacement(
                    isNativeDeps && cppConfiguration.shareNativeDeps()
                        ? output.getExecPathString()
                        : Label.print(getOwner().getLabel())),
                Matcher.quoteReplacement(output.getExecPathString()),
                cppSemantics));
      }

      inputsBuilder.addAll(linkstampMap.values());
    }

    inputsBuilder.addTransitive(linkstampObjectArtifacts);

    ImmutableSet<Artifact> fakeLinkerInputArtifacts =
        collectedLibrariesToLink.getExpandedLinkerInputs().toList().stream()
            .filter(LinkerInput::isFake)
            .map(LinkerInput::getArtifact)
            .collect(ImmutableSet.toImmutableSet());

    return new CppLinkAction(
        getOwner(),
        mnemonic,
        inputsBuilder.build(),
        actionOutputs,
        outputLibrary,
        output,
        interfaceOutputLibrary,
        fake,
        fakeLinkerInputArtifacts,
        isLtoIndexing,
        linkstampMap,
        linkCommandLine,
        configuration.getActionEnvironment(),
        toolchainEnv,
        ImmutableMap.copyOf(executionInfo),
        toolchain.getToolPathFragment(Tool.LD, ruleErrorConsumer),
        toolchain.getHostSystemName(),
        toolchain.getTargetCpu());
  }

  /** We're doing 4-phased lto build, and this is the final link action (4-th phase). */
  private boolean isFinalLinkOfLtoBuild() {
    return !isLtoIndexing && allLtoArtifacts != null;
  }

  private static NestedSet<Artifact> getArtifactsPossiblyLtoMapped(
      Iterable<LinkerInput> inputs, Map<Artifact, Artifact> ltoMapping) {
    Preconditions.checkNotNull(ltoMapping);
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    Iterable<Artifact> artifacts = LinkerInputs.toLibraryArtifacts(inputs);
    for (Artifact a : artifacts) {
      Artifact renamed = ltoMapping.get(a);
      result.add(renamed == null ? a : renamed);
    }
    return result.build();
  }

  private boolean shouldUseLinkDynamicLibraryTool() {
    return linkType.isDynamicLibrary()
        && toolchain.supportsInterfaceSharedLibraries(featureConfiguration)
        && !featureConfiguration.hasConfiguredLinkerPathInActionConfig();
  }

  /** The default heuristic on whether we need to use whole-archive for the link. */
  private static boolean needWholeArchive(
      FeatureConfiguration featureConfiguration,
      LinkingMode linkingMode,
      LinkTargetType type,
      Collection<String> linkopts,
      CppConfiguration cppConfig) {
    boolean sharedLinkopts =
        type.isDynamicLibrary() || linkopts.contains("-shared") || cppConfig.hasSharedLinkOption();
    // Fasten your seat belt, the logic below doesn't make perfect sense and it's full of obviously
    // missed corner cases. The world still stands and depends on this behavior, so ¯\_(ツ)_/¯.
    if (!sharedLinkopts) {
      // We are not producing shared library, there is no reason to use --whole-archive with
      // executable (if the executable doesn't use the symbols, nobody else will, so --whole-archive
      // is not needed).
      return false;
    }
    if (cppConfig.removeLegacyWholeArchive()) {
      // --incompatible_remove_legacy_whole_archive has been flipped, no --whole-archive for the
      // entire build.
      return false;
    }
    if (linkingMode != LinkingMode.STATIC) {
      // legacy whole archive only applies to static linking mode.
      return false;
    }
    if (featureConfiguration.getRequestedFeatures().contains(CppRuleClasses.LEGACY_WHOLE_ARCHIVE)) {
      // --incompatible_remove_legacy_whole_archive has not been flipped, and this target requested
      // --whole-archive using features.
      return true;
    }
    if (cppConfig.legacyWholeArchive()) {
      // --incompatible_remove_legacy_whole_archive has not been flipped, so whether to
      // use --whole-archive depends on --legacy_whole_archive.
      return true;
    }
    // Hopefully future default.
    return false;
  }

  private static ImmutableSet<Artifact> constructOutputs(
      Artifact primaryOutput, Iterable<Artifact> outputList, Artifact... outputs) {
    return new ImmutableSet.Builder<Artifact>()
        .add(primaryOutput)
        .addAll(outputList)
        .addAll(CollectionUtils.asSetWithoutNulls(outputs))
        .build();
  }

  /**
   * Translates a collection of {@link Linkstamp} instances to an immutable mapping from linkstamp
   * to object files. In other words, given a set of source files, this method determines the output
   * path to which each file should be compiled.
   *
   * @param linkstamps set of {@link Linkstamp}s
   * @param actionConstructionContext of the rule for which this link is being performed
   * @param repositoryName of the rule for which this link is being performed
   * @param outputBinary the binary output path for this link
   * @return an immutable map that pairs each source file with the corresponding object file that
   *     should be fed into the link
   */
  public static ImmutableMap<Linkstamp, Artifact> mapLinkstampsToOutputs(
      ImmutableSet<Linkstamp> linkstamps,
      ActionConstructionContext actionConstructionContext,
      RepositoryName repositoryName,
      BuildConfiguration configuration,
      Artifact outputBinary,
      LinkArtifactFactory linkArtifactFactory) {
    ImmutableMap.Builder<Linkstamp, Artifact> mapBuilder = ImmutableMap.builder();

    PathFragment outputBinaryPath = outputBinary.getRootRelativePath();
    PathFragment stampOutputDirectory =
        outputBinaryPath
            .getParentDirectory()
            .getRelative(CppHelper.OBJS)
            .getRelative(outputBinaryPath.getBaseName());

    for (Linkstamp linkstamp : linkstamps) {
      PathFragment stampOutputPath =
          stampOutputDirectory.getRelative(
              FileSystemUtils.replaceExtension(
                  linkstamp.getArtifact().getRootRelativePath(), ".o"));
      mapBuilder.put(
          linkstamp,
          // Note that link stamp actions can be shared between link actions that output shared
          // native dep libraries.
          linkArtifactFactory.create(
              actionConstructionContext, repositoryName, configuration, stampOutputPath));
    }
    return mapBuilder.build();
  }

  protected ActionOwner getOwner() {
    ActionOwner execGroupOwner = actionConstructionContext.getActionOwner(CPP_LINK_EXEC_GROUP);
    return execGroupOwner == null ? actionConstructionContext.getActionOwner() : execGroupOwner;
  }

  /** Sets the mnemonic for the link action. */
  public CppLinkActionBuilder setMnemonic(String mnemonic) {
    this.mnemonic = mnemonic;
    return this;
  }

  /** Set the crosstool inputs required for the action. */
  public CppLinkActionBuilder setLinkerFiles(NestedSet<Artifact> linkerFiles) {
    this.linkerFiles = linkerFiles;
    return this;
  }

  /** Returns the set of LTO artifacts created during build() */
  public Iterable<LtoBackendArtifacts> getAllLtoBackendArtifacts() {
    return allLtoArtifacts;
  }

  /**
   * This is the LTO indexing step, rather than the real link.
   *
   * <p>When using this, build() will store allLtoArtifacts as a side-effect so the next build()
   * call can emit the real link. Do not call addInput() between the two build() calls.
   */
  public CppLinkActionBuilder setLtoIndexing(boolean ltoIndexing) {
    this.isLtoIndexing = ltoIndexing;
    return this;
  }

  /** Sets flag for using PIC in any scheduled LTO Backend actions. */
  public CppLinkActionBuilder setUsePicForLtoBackendActions(boolean usePic) {
    this.usePicForLtoBackendActions = usePic;
    return this;
  }

  /** Sets the C++ runtime library inputs for the action. */
  public CppLinkActionBuilder setRuntimeInputs(
      ArtifactCategory runtimeType, Artifact middleman, NestedSet<Artifact> inputs) {
    this.toolchainLibrariesType = runtimeType;
    this.runtimeMiddleman = middleman;
    this.toolchainLibrariesInputs = inputs;
    return this;
  }

  /** Adds a variables extension to template the toolchain for this link action. */
  public CppLinkActionBuilder addVariablesExtension(VariablesExtension variablesExtension) {
    this.variablesExtensions.add(variablesExtension);
    return this;
  }

  /** Adds variables extensions to template the toolchain for this link action. */
  public CppLinkActionBuilder addVariablesExtensions(List<VariablesExtension> variablesExtensions) {
    for (VariablesExtension variablesExtension : variablesExtensions) {
      addVariablesExtension(variablesExtension);
    }
    return this;
  }

  /**
   * Sets the interface output of the link. A non-null argument can only be provided if the link
   * type is {@code NODEPS_DYNAMIC_LIBRARY} and fake is false.
   */
  public CppLinkActionBuilder setInterfaceOutput(Artifact interfaceOutput) {
    this.interfaceOutput = interfaceOutput;
    return this;
  }

  public CppLinkActionBuilder addLtoCompilationContext(
      LtoCompilationContext ltoCompilationContext) {
    Preconditions.checkState(this.ltoCompilationContext == null);
    this.ltoCompilationContext = ltoCompilationContext;
    return this;
  }

  public CppLinkActionBuilder setDefFile(Artifact defFile) {
    this.defFile = defFile;
    return this;
  }

  private void addObjectFile(LinkerInput input) {
    // We skip file extension checks for TreeArtifacts because they represent directory artifacts
    // without a file extension.
    String name = input.getArtifact().getFilename();
    Preconditions.checkArgument(
        input.getArtifact().isTreeArtifact() || Link.OBJECT_FILETYPES.matches(name), name);
    this.objectFiles.add(input);
    if (input.isMustKeepDebug()) {
      this.mustKeepDebug = true;
    }
  }

  /** Adds a single object file to the set of inputs. */
  public CppLinkActionBuilder addObjectFile(Artifact input) {
    addObjectFile(
        LinkerInputs.simpleLinkerInput(
            input, ArtifactCategory.OBJECT_FILE, /* disableWholeArchive= */ false));
    return this;
  }

  /** Adds object files to the linker action. */
  public CppLinkActionBuilder addObjectFiles(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addObjectFile(
          LinkerInputs.simpleLinkerInput(
              input, ArtifactCategory.OBJECT_FILE, /* disableWholeArchive= */ false));
    }
    return this;
  }

  /**
   * Adds non-code files to the set of inputs. They will not be passed to the linker command line
   * unless that is explicitly modified, too.
   */
  public CppLinkActionBuilder addNonCodeInputs(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addNonCodeInput(input);
    }

    return this;
  }

  /**
   * Adds a single non-code file to the set of inputs. It will not be passed to the linker command
   * line unless that is explicitly modified, too.
   */
  public CppLinkActionBuilder addNonCodeInput(Artifact input) {
    String basename = input.getFilename();
    Preconditions.checkArgument(!Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename), basename);
    Preconditions.checkArgument(!Link.SHARED_LIBRARY_FILETYPES.matches(basename), basename);
    Preconditions.checkArgument(!Link.OBJECT_FILETYPES.matches(basename), basename);

    this.nonCodeInputs.add(input);
    return this;
  }

  public CppLinkActionBuilder addFakeObjectFiles(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addObjectFile(LinkerInputs.fakeLinkerInput(input));
    }
    return this;
  }

  private void checkLibrary(LinkerInputs.LibraryToLink input) {
    String name = input.getArtifact().getFilename();
    Preconditions.checkArgument(
        Link.ARCHIVE_LIBRARY_FILETYPES.matches(name) || Link.SHARED_LIBRARY_FILETYPES.matches(name),
        "'%s' is not a library file",
        input);
  }

  /**
   * Adds a single artifact to the set of inputs. The artifact must be an archive or a shared
   * library. Note that all directly added libraries are implicitly ordered before all nested sets
   * added with {@link #addLibraries}, even if added in the opposite order.
   */
  public CppLinkActionBuilder addLibrary(LinkerInputs.LibraryToLink input) {
    checkLibrary(input);
    libraries.add(input);
    if (input.isMustKeepDebug()) {
      mustKeepDebug = true;
    }
    return this;
  }

  /**
   * Adds multiple artifact to the set of inputs. The artifacts must be archives or shared
   * libraries.
   */
  public CppLinkActionBuilder addLibraries(Collection<LinkerInputs.LibraryToLink> inputs) {
    Preconditions.checkState(librariesToLink.isEmpty());
    for (LinkerInputs.LibraryToLink input : inputs) {
      checkLibrary(input);
      if (input.isMustKeepDebug()) {
        mustKeepDebug = true;
      }
    }
    this.libraries.addAll(inputs);
    return this;
  }

  public CppLinkActionBuilder addLibrariesToLink(Iterable<LibraryToLink> inputs) {
    Preconditions.checkState(libraries.isEmpty());
    for (LibraryToLink input : inputs) {
      if (input.getMustKeepDebug()) {
        mustKeepDebug = true;
      }
    }
    this.librariesToLink.addAll(inputs);
    return this;
  }

  /**
   * Sets the type of ELF file to be created (.a, .so, .lo, executable). The default is {@link
   * LinkTargetType#STATIC_LIBRARY}.
   */
  public CppLinkActionBuilder setLinkType(LinkTargetType linkType) {
    this.linkType = linkType;
    return this;
  }

  /**
   * Sets the degree of "staticness" of the link: fully static (static binding of all symbols),
   * mostly static (use dynamic binding only for symbols from glibc), dynamic (use dynamic binding
   * wherever possible). The default is {@link LinkingMode#STATIC}.
   */
  public CppLinkActionBuilder setLinkingMode(Link.LinkingMode linkingMode) {
    this.linkingMode = linkingMode;
    return this;
  }

  /**
   * Sets the identifier of the library produced by the action. See {@link
   * LinkerInputs.LibraryToLink#getLibraryIdentifier()}
   */
  public CppLinkActionBuilder setLibraryIdentifier(String libraryIdentifier) {
    this.libraryIdentifier = libraryIdentifier;
    return this;
  }

  /**
   * Adds {@link Linkstamp}s.
   *
   * <p>This is used to embed various values from the build system into binaries to identify their
   * provenance.
   *
   * <p>Linkstamp object files are also automatically added to the inputs of the link action.
   */
  public CppLinkActionBuilder addLinkstamps(Iterable<Linkstamp> linkstamps) {
    this.linkstampsBuilder.addAll(linkstamps);
    return this;
  }

  public CppLinkActionBuilder setAdditionalLinkstampDefines(
      ImmutableList<String> additionalLinkstampDefines) {
    this.additionalLinkstampDefines = Preconditions.checkNotNull(additionalLinkstampDefines);
    return this;
  }

  /** Adds an additional linker option. */
  public CppLinkActionBuilder addLinkopt(String linkopt) {
    this.linkopts.add(linkopt);
    return this;
  }

  /**
   * Adds multiple linker options at once.
   *
   * @see #addLinkopt(String)
   */
  public CppLinkActionBuilder addLinkopts(Collection<String> linkopts) {
    this.linkopts.addAll(linkopts);
    return this;
  }

  /**
   * Merges the given link params into this builder by calling {@link #addLinkopts}, {@link
   * #addLibraries}, and {@link #addLinkstamps}.
   */
  public CppLinkActionBuilder addLinkParams(
      List<LinkerInputs.LibraryToLink> libraries,
      List<String> userLinkFlags,
      List<Linkstamp> linkstamps,
      List<Artifact> nonCodeInputs,
      RuleErrorConsumer errorListener) {
    addLinkopts(userLinkFlags);
    addLibraries(libraries);
    if (nonCodeInputs != null) {
      addNonCodeInputs(nonCodeInputs);
    }
    CppHelper.checkLinkstampsUnique(errorListener, linkstamps);
    addLinkstamps(linkstamps);
    return this;
  }

  /** Sets whether this link action will be used for a cc_fake_binary; false by default. */
  public CppLinkActionBuilder setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  /** Sets whether this link action is used for a native dependency library. */
  public CppLinkActionBuilder setNativeDeps(boolean isNativeDeps) {
    this.isNativeDeps = isNativeDeps;
    return this;
  }

  /**
   * Setting this to true overrides the default whole-archive computation and force-enables whole
   * archives for every archive in the link. This is only necessary for linking executable binaries
   * that are supposed to export symbols.
   *
   * <p>Usually, the link action while use whole archives for dynamic libraries that are native deps
   * (or the legacy whole archive flag is enabled), and that are not dynamically linked.
   *
   * <p>(Note that it is possible to build dynamic libraries with cc_binary rules by specifying
   * linkshared = 1, and giving the rule a name that matches the pattern {@code
   * lib&lt;name&gt;.so}.)
   */
  public CppLinkActionBuilder setWholeArchive(boolean wholeArchive) {
    this.wholeArchive = wholeArchive;
    return this;
  }

  /**
   * Sets whether this link action should use test-specific flags (e.g. $EXEC_ORIGIN instead of
   * $ORIGIN for the solib search path or lazy binding); false by default.
   */
  public CppLinkActionBuilder setUseTestOnlyFlags(boolean useTestOnlyFlags) {
    this.useTestOnlyFlags = useTestOnlyFlags;
    return this;
  }

  /** Used to set the runfiles path for tests' dynamic libraries. */
  public CppLinkActionBuilder setTestOrTestOnlyTarget(boolean isTestOrTestOnlyTarget) {
    this.isTestOrTestOnlyTarget = isTestOrTestOnlyTarget;
    return this;
  }

  /**
   * Used to test the grep-includes tool. This is needing during linking because of linkstamping.
   */
  public CppLinkActionBuilder setGrepIncludes(Artifact grepIncludes) {
    this.grepIncludes = grepIncludes;
    return this;
  }

  /** Whether linkstamping is enabled. */
  public CppLinkActionBuilder setIsStampingEnabled(boolean isStampingEnabled) {
    this.isStampingEnabled = isStampingEnabled;
    return this;
  }

  /**
   * Sets the name of the directory where the solib symlinks for the dynamic runtime libraries live.
   * This is usually automatically set from the cc_toolchain.
   */
  public CppLinkActionBuilder setToolchainLibrariesSolibDir(
      PathFragment toolchainLibrariesSolibDir) {
    this.toolchainLibrariesSolibDir = toolchainLibrariesSolibDir;
    return this;
  }

  /** Adds an extra input artifact to the link action. */
  public CppLinkActionBuilder addActionInput(Artifact input) {
    this.linkActionInputs.add(input);
    return this;
  }

  /** Adds extra input artifacts to the link action. */
  public CppLinkActionBuilder addActionInputs(Iterable<Artifact> inputs) {
    this.linkActionInputs.addAll(inputs);
    return this;
  }

  /** Adds extra input artifacts to the link actions. */
  public CppLinkActionBuilder addTransitiveActionInputs(NestedSet<Artifact> inputs) {
    this.linkActionInputs.addTransitive(inputs);
    return this;
  }

  /** Adds an extra output artifact to the link action. */
  public CppLinkActionBuilder addActionOutput(Artifact output) {
    this.linkActionOutputs.add(output);
    return this;
  }

  public CppLinkActionBuilder addExecutionInfo(Map<String, String> executionInfo) {
    this.executionInfo.putAll(executionInfo);
    return this;
  }
}
