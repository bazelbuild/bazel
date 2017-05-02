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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.ImmutableIterable;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.Context;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** Builder class to construct {@link CppLinkAction}s. */
public class CppLinkActionBuilder {

  /** A build variable for entries in the linker runtime search path (usually set by -rpath flag) */
  public static final String RUNTIME_LIBRARY_SEARCH_DIRECTORIES_VARIABLE =
      "runtime_library_search_directories";

  public static final String LIBRARY_SEARCH_DIRECTORIES_VARIABLE = "library_search_directories";

  /** A build variable for flags providing files to link as inputs in the linker invocation */
  public static final String LIBRARIES_TO_LINK_VARIABLE = "libraries_to_link";

  /**
   * A build variable for thinlto param file produced by thinlto-indexing action and consumed by
   * normal linking actions.
   */
  public static final String THINLTO_PARAM_FILE_VARIABLE = "thinlto_param_file";

  /**
   * A build variable to let thinlto know where it should write linker flags when indexing.
   */
  public static final String THINLTO_INDEXING_PARAM_FILE_VARIABLE = "thinlto_indexing_param_file";

  public static final String THINLTO_PREFIX_REPLACE_VARIABLE = "thinlto_prefix_replace";

  /**
   * A build variable to let the LTO indexing step know how to map from the minimized bitcode file
   * to the full bitcode file used by the LTO Backends.
   */
  public static final String THINLTO_OBJECT_SUFFIX_REPLACE_VARIABLE =
      "thinlto_object_suffix_replace";

  /**
   * A build variable for linker param file created by Bazel to overcome the command line length
   * limit.
   */
  public static final String LINKER_PARAM_FILE_VARIABLE = "linker_param_file";

  /** A build variable for the execpath of the output of the linker. */
  public static final String OUTPUT_EXECPATH_VARIABLE = "output_execpath";

  /** A build variable setting if interface library should be generated. */
  public static final String GENERATE_INTERFACE_LIBRARY_VARIABLE = "generate_interface_library";

  /** A build variable for the path to the interface library builder tool. */
  public static final String INTERFACE_LIBRARY_BUILDER_VARIABLE = "interface_library_builder_path";

  /** A build variable for the input for the interface library builder tool. */
  public static final String INTERFACE_LIBRARY_INPUT_VARIABLE = "interface_library_input_path";

  /** A build variable for the path where to generate interface library using the builder tool. */
  public static final String INTERFACE_LIBRARY_OUTPUT_VARIABLE = "interface_library_output_path";

  /** A build variable for hard-coded linker flags currently only known by bazel. */
  public static final String LEGACY_LINK_FLAGS_VARIABLE = "legacy_link_flags";
  /**
   * A build variable that is set to indicate a mostly static linking for which the linked binary
   * should be piped to /dev/null.
   */
  public static final String SKIP_MOSTLY_STATIC_VARIABLE = "skip_mostly_static";

  /** A build variable giving a path to which to write symbol counts. */
  public static final String SYMBOL_COUNTS_OUTPUT_VARIABLE = "symbol_counts_output";

  /** A build variable giving linkstamp paths. */
  public static final String LINKSTAMP_PATHS_VARIABLE = "linkstamp_paths";

  /** A build variable whose presence indicates that PIC code should be generated. */
  public static final String FORCE_PIC_VARIABLE = "force_pic";

  /** A build variable whose presence indicates that the debug symbols should be stripped. */
  public static final String STRIP_DEBUG_SYMBOLS_VARIABLE = "strip_debug_symbols";

  /** A build variable whose presence indicates that this action is a cc_test linking action. */
  public static final String IS_CC_TEST_LINK_ACTION_VARIABLE = "is_cc_test_link_action";

  /**
   *  A build variable whose presence indicates that files were compiled with fission (debug
   *  info is in .dwo files instead of .o files and linker needs to know).
   */
  public static final String IS_USING_FISSION_VARIABLE = "is_using_fission";

  /**
   * A (temporary) build variable whose presence indicates that this action is not a cc_test linking
   * action.
   */
  public static final String IS_NOT_CC_TEST_LINK_ACTION_VARIABLE = "is_not_cc_test_link_action";

  // Builder-only
  // Null when invoked from tests (e.g. via createTestBuilder).
  @Nullable private final RuleContext ruleContext;
  private final AnalysisEnvironment analysisEnvironment;
  private final Artifact output;
  @Nullable private String mnemonic;

  // can be null for CppLinkAction.createTestBuilder()
  @Nullable private final CcToolchainProvider toolchain;
  private final FdoSupportProvider fdoSupport;
  private Artifact interfaceOutput;
  private Artifact symbolCounts;
  private PathFragment runtimeSolibDir;
  protected final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;
  private FeatureConfiguration featureConfiguration;

  // Morally equivalent with {@link Context}, except these are mutable.
  // Keep these in sync with {@link Context}.
  private final Set<LinkerInput> objectFiles = new LinkedHashSet<>();
  private final Set<Artifact> nonCodeInputs = new LinkedHashSet<>();
  private final NestedSetBuilder<LibraryToLink> libraries = NestedSetBuilder.linkOrder();
  private NestedSet<Artifact> crosstoolInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private Artifact runtimeMiddleman;
  private ArtifactCategory runtimeType = null;
  private NestedSet<Artifact> runtimeInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private final NestedSetBuilder<Artifact> compilationInputs = NestedSetBuilder.stableOrder();
  private final Set<Artifact> linkstamps = new LinkedHashSet<>();
  private List<String> linkstampOptions = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private LinkStaticness linkStaticness = LinkStaticness.FULLY_STATIC;
  private String libraryIdentifier = null;
  private ImmutableMap<Artifact, Artifact> ltoBitcodeFiles;

  private boolean fake;
  private boolean isNativeDeps;
  private boolean useTestOnlyFlags;
  private boolean wholeArchive;
  private LinkArtifactFactory linkArtifactFactory = CppLinkAction.DEFAULT_ARTIFACT_FACTORY;

  private boolean isLTOIndexing = false;
  private boolean usePicForLTOBackendActions = false;
  private boolean useFissionForLTOBackendActions = false;
  private Iterable<LTOBackendArtifacts> allLTOArtifacts = null;
  
  private final List<VariablesExtension> variablesExtensions = new ArrayList<>();
  private final NestedSetBuilder<Artifact> linkActionInputs = NestedSetBuilder.stableOrder();
  private final ImmutableList.Builder<Artifact> linkActionOutputs = ImmutableList.builder();

  /**
   * Creates a builder that builds {@link CppLinkAction} instances.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   * @param toolchain the C++ toolchain provider
   * @param fdoSupport the C++ FDO optimization support
   */
  public CppLinkActionBuilder(
      RuleContext ruleContext,
      Artifact output,
      CcToolchainProvider toolchain,
      FdoSupportProvider fdoSupport) {
    this(
        ruleContext,
        output,
        ruleContext.getConfiguration(),
        ruleContext.getAnalysisEnvironment(),
        toolchain,
        fdoSupport);
  }

  /**
   * Creates a builder that builds {@link CppLinkAction} instances.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   * @param configuration build configuration
   * @param toolchain C++ toolchain provider
   * @param fdoSupport the C++ FDO optimization support
   */
  public CppLinkActionBuilder(
      RuleContext ruleContext,
      Artifact output,
      BuildConfiguration configuration,
      CcToolchainProvider toolchain,
      FdoSupportProvider fdoSupport) {
    this(ruleContext, output, configuration, ruleContext.getAnalysisEnvironment(), toolchain,
        fdoSupport);
  }

  /**
   * Creates a builder that builds {@link CppLinkAction}s.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   * @param configuration the configuration used to determine the tool chain and the default link
   *     options
   * @param toolchain the C++ toolchain provider
   * @param fdoSupport the C++ FDO optimization support
   */
  private CppLinkActionBuilder(
      @Nullable RuleContext ruleContext,
      Artifact output,
      BuildConfiguration configuration,
      AnalysisEnvironment analysisEnvironment,
      CcToolchainProvider toolchain,
      FdoSupportProvider fdoSupport) {
    this.ruleContext = ruleContext;
    this.analysisEnvironment = Preconditions.checkNotNull(analysisEnvironment);
    this.output = Preconditions.checkNotNull(output);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
    this.toolchain = toolchain;
    this.fdoSupport = fdoSupport;
    if (cppConfiguration.supportsEmbeddedRuntimes() && toolchain != null) {
      runtimeSolibDir = toolchain.getDynamicRuntimeSolibDir();
    }
  }

  /**
   * Given a Context, creates a Builder that builds {@link CppLinkAction}s. Note well: Keep the
   * Builder->Context and Context->Builder transforms consistent!
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   * @param linkContext an immutable CppLinkAction.Context from the original builder
   * @param configuration build configuration
   * @param toolchain the C++ toolchain provider
   * @param fdoSupport the C++ FDO optimization support
   */
  public CppLinkActionBuilder(
      RuleContext ruleContext,
      Artifact output,
      Context linkContext,
      BuildConfiguration configuration,
      CcToolchainProvider toolchain,
      FdoSupportProvider fdoSupport) {
    // These Builder-only fields get set in the constructor:
    //   ruleContext, analysisEnvironment, outputPath, configuration, runtimeSolibDir
    this(
        ruleContext,
        output,
        configuration,
        ruleContext.getAnalysisEnvironment(),
        toolchain,
        fdoSupport);
    Preconditions.checkNotNull(linkContext);

    // All linkContext fields should be transferred to this Builder.
    this.objectFiles.addAll(linkContext.objectFiles);
    this.nonCodeInputs.addAll(linkContext.nonCodeInputs);
    this.libraries.addTransitive(linkContext.libraries);
    this.crosstoolInputs = linkContext.crosstoolInputs;
    this.ltoBitcodeFiles = linkContext.ltoBitcodeFiles;
    this.runtimeMiddleman = linkContext.runtimeMiddleman;
    this.runtimeInputs = linkContext.runtimeInputs;
    this.runtimeType = linkContext.runtimeType;
    this.compilationInputs.addTransitive(linkContext.compilationInputs);
    this.linkstamps.addAll(linkContext.linkstamps);
    this.linkopts.addAll(linkContext.linkopts);
    this.linkType = linkContext.linkType;
    this.linkStaticness = linkContext.linkStaticness;
    this.fake = linkContext.fake;
    this.isNativeDeps = linkContext.isNativeDeps;
    this.useTestOnlyFlags = linkContext.useTestOnlyFlags;
  }

  /** Returns the action name for purposes of querying the crosstool. */
  private String getActionName() {
    return linkType.getActionName();
  }
  
  /** Returns linker inputs that are not libraries. */
  public Set<LinkerInput> getObjectFiles() {
    return objectFiles;
  }

  public Set<Artifact> getNonCodeInputs() {
    return nonCodeInputs;
  }

  /**
   * Returns linker inputs that are libraries.
   */
  public NestedSetBuilder<LibraryToLink> getLibraries() {
    return libraries;
  }

  /**
   * Returns inputs arising from the crosstool.
   */
  public NestedSet<Artifact> getCrosstoolInputs() {
    return this.crosstoolInputs;
  }
  
  /**
   * Returns the runtime middleman artifact.
   */
  public Artifact getRuntimeMiddleman() {
    return this.runtimeMiddleman;
  }
  
  /**
   * Returns runtime inputs for this link action.
   */
  public NestedSet<Artifact> getRuntimeInputs() {
    return this.runtimeInputs;
  }

  public ArtifactCategory getRuntimeType() {
    return runtimeType;
  }

  /**
   * Returns compilation inputs for this link action.
   */
  public final NestedSetBuilder<Artifact> getCompilationInputs() {
    return this.compilationInputs;
  }
  
  /**
   * Returns linkstamps for this link action.
   */
  public final Set<Artifact> getLinkstamps() {
    return this.linkstamps;
  }
  /**
   * Returns linkstamp options for this link action.
   */
  public List<String> getLinkstampOptions() {
    return this.linkstampOptions;
  }

  /**
   * Returns command line options for this link action.
   */
  public final List<String> getLinkopts() {
    return this.linkopts;
  }
  
  /**
   * Returns the type of this link action.
   */
  public LinkTargetType getLinkType() {
    return this.linkType;
  }
  /**
   * Returns the staticness of this link action.
   */
  public LinkStaticness getLinkStaticness() {
    return this.linkStaticness;
  }
  /**
   * Returns linker inputs that are lto bitcode files in a map from the full bitcode file used by
   * the LTO Backend to the minimized bitcode used by the LTO indexing.
   */
  public ImmutableMap<Artifact, Artifact> getLtoBitcodeFiles() {
    return this.ltoBitcodeFiles;
  }

  /**
   * Returns true for a cc_fake_binary.
   */
  public boolean isFake() {
    return this.fake;
  }
  
  /**
   * Returns true for native dependencies of another language.
   */
  public boolean isNativeDeps() {
    return this.isNativeDeps;
  }
 
  public CppLinkActionBuilder setLinkArtifactFactory(LinkArtifactFactory linkArtifactFactory) {
    this.linkArtifactFactory = linkArtifactFactory;
    return this;
  }
  
  /**
   * Returns true if this link action uses test only flags.
   */
  public boolean useTestOnlyFlags() {
    return this.useTestOnlyFlags;
  }

  /**
   * Maps bitcode object files used by the LTO backends to the corresponding minimized bitcode file
   * used as input to the LTO indexing step.
   */
  private ImmutableSet<LinkerInput> computeLTOIndexingObjectFileInputs() {
    ImmutableSet.Builder<LinkerInput> objectFileInputsBuilder = ImmutableSet.<LinkerInput>builder();
    for (LinkerInput input : objectFiles) {
      Artifact objectFile = input.getArtifact();
      objectFileInputsBuilder.add(
          LinkerInputs.simpleLinkerInput(
              this.ltoBitcodeFiles.getOrDefault(objectFile, objectFile),
              ArtifactCategory.OBJECT_FILE));
    }
    return objectFileInputsBuilder.build();
  }

  /**
   * Maps bitcode library files used by the LTO backends to the corresponding minimized bitcode file
   * used as input to the LTO indexing step.
   */
  private static NestedSet<LibraryToLink> computeLTOIndexingUniqueLibraries(
      NestedSet<LibraryToLink> originalUniqueLibraries) {
    NestedSetBuilder<LibraryToLink> uniqueLibrariesBuilder = NestedSetBuilder.linkOrder();
    for (LibraryToLink lib : originalUniqueLibraries) {
      if (!lib.containsObjectFiles()) {
        uniqueLibrariesBuilder.add(lib);
        continue;
      }
      ImmutableSet.Builder<Artifact> newObjectFilesBuilder = ImmutableSet.<Artifact>builder();
      for (Artifact a : lib.getObjectFiles()) {
        newObjectFilesBuilder.add(lib.getLTOBitcodeFiles().getOrDefault(a, a));
      }
      uniqueLibrariesBuilder.add(
          LinkerInputs.newInputLibrary(
              lib.getArtifact(),
              lib.getArtifactCategory(),
              lib.getLibraryIdentifier(),
              newObjectFilesBuilder.build(),
              lib.getLTOBitcodeFiles()));
    }
    return uniqueLibrariesBuilder.build();
  }

  private Iterable<LTOBackendArtifacts> createLTOArtifacts(
      PathFragment ltoOutputRootPrefix, NestedSet<LibraryToLink> uniqueLibraries) {
    Set<Artifact> compiled = new LinkedHashSet<>();
    for (LibraryToLink lib : uniqueLibraries) {
      compiled.addAll(lib.getLTOBitcodeFiles().keySet());
    }

    // This flattens the set of object files, so for M binaries and N .o files,
    // this is O(M*N). If we had a nested set of .o files, we could have O(M + N) instead.
    Map<PathFragment, Artifact> allBitcode = new HashMap<>();
    for (LibraryToLink lib : uniqueLibraries) {
      if (!lib.containsObjectFiles()) {
        continue;
      }
      for (Artifact a : lib.getObjectFiles()) {
        if (compiled.contains(a)) {
          allBitcode.put(a.getExecPath(), a);
        }
      }
    }
    for (LinkerInput input : objectFiles) {
      if (this.ltoBitcodeFiles.containsKey(input.getArtifact())) {
        allBitcode.put(input.getArtifact().getExecPath(), input.getArtifact());
      }
    }

    ImmutableList.Builder<LTOBackendArtifacts> ltoOutputs = ImmutableList.builder();
    for (Artifact a : allBitcode.values()) {
      LTOBackendArtifacts ltoArtifacts =
          new LTOBackendArtifacts(
              ltoOutputRootPrefix, a, allBitcode, ruleContext, configuration, linkArtifactFactory);
      ltoOutputs.add(ltoArtifacts);
    }
    return ltoOutputs.build();
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
        // We currently can't split dynamic library links if they have interface outputs. That was
        // probably an unintended side effect of the change that introduced interface outputs.
      case DYNAMIC_LIBRARY:
        return interfaceOutput == null;
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

  /** Builds the Action as configured and returns it. */
  public CppLinkAction build() throws InterruptedException {
    // Executable links do not have library identifiers.
    boolean hasIdentifier = (libraryIdentifier != null);
    boolean isExecutable = linkType.isExecutable();
    Preconditions.checkState(hasIdentifier != isExecutable);    

    if (interfaceOutput != null && (fake || linkType != LinkTargetType.DYNAMIC_LIBRARY)) {
      throw new RuntimeException(
          "Interface output can only be used " + "with non-fake DYNAMIC_LIBRARY targets");
    }

    final ImmutableList<Artifact> buildInfoHeaderArtifacts =
        !linkstamps.isEmpty()
            ? analysisEnvironment.getBuildInfo(ruleContext, CppBuildInfo.KEY, configuration)
            : ImmutableList.<Artifact>of();

    boolean needWholeArchive =
        wholeArchive
            || needWholeArchive(linkStaticness, linkType, linkopts, isNativeDeps, cppConfiguration);

    NestedSet<LibraryToLink> originalUniqueLibraries = libraries.build();

    // Get the set of object files and libraries containing the correct
    // inputs for this link, depending on whether this is LTO indexing or
    // a native link.
    NestedSet<LibraryToLink> uniqueLibraries;
    ImmutableSet<LinkerInput> objectFileInputs;
    if (isLTOIndexing) {
      objectFileInputs = computeLTOIndexingObjectFileInputs();
      uniqueLibraries = computeLTOIndexingUniqueLibraries(originalUniqueLibraries);
    } else {
      objectFileInputs = ImmutableSet.copyOf(objectFiles);
      uniqueLibraries = originalUniqueLibraries;
    }
    final Iterable<Artifact> objectArtifacts = LinkerInputs.toLibraryArtifacts(objectFileInputs);

    final Iterable<LinkerInput> linkerInputs =
        IterablesChain.<LinkerInput>builder()
            .add(objectFileInputs)
            .add(
                ImmutableIterable.from(
                    Link.mergeInputsCmdLine(
                        uniqueLibraries, needWholeArchive, cppConfiguration.archiveType())))
            .build();

    // ruleContext can only be null during testing. This is kind of ugly.
    final ImmutableSet<String> features =
        (ruleContext == null) ? ImmutableSet.<String>of() : ruleContext.getFeatures();

    // For backwards compatibility, and for tests, we permit the link action to be
    // instantiated without a feature configuration.
    if (featureConfiguration == null) {
      if (toolchain != null) {
        featureConfiguration =
            CcCommon.configureFeatures(ruleContext, toolchain, CcLibraryHelper.SourceCategory.CC);
      } else {
        featureConfiguration = CcCommon.configureFeatures(ruleContext, toolchain);
      }
    }

    final LibraryToLink outputLibrary = linkType.isExecutable()
        ? null
        : LinkerInputs.newInputLibrary(output,
            linkType.getLinkerOutput(),
            libraryIdentifier,
            objectArtifacts, this.ltoBitcodeFiles);
    final LibraryToLink interfaceOutputLibrary =
        (interfaceOutput == null)
            ? null
            : LinkerInputs.newInputLibrary(interfaceOutput,
                ArtifactCategory.DYNAMIC_LIBRARY,
                libraryIdentifier,
                objectArtifacts, this.ltoBitcodeFiles);

    final ImmutableMap<Artifact, Artifact> linkstampMap =
        mapLinkstampsToOutputs(linkstamps, ruleContext, configuration, output, linkArtifactFactory);

    PathFragment ltoOutputRootPrefix = null;
    if (isLTOIndexing && allLTOArtifacts == null) {
      ltoOutputRootPrefix =
          FileSystemUtils.appendExtension(
              output.getRootRelativePath(), ".lto");
      // Use the originalUniqueLibraries which contains the full bitcode files
      // needed by the LTO backends (as opposed to the minimized bitcode files
      // that can be used by the LTO indexing step).
      allLTOArtifacts = createLTOArtifacts(ltoOutputRootPrefix, originalUniqueLibraries);
    }

    PathFragment linkerParamFileRootPath = null;
    @Nullable Artifact thinltoParamFile = null;
    if (allLTOArtifacts != null) {
      // Create artifact for the file that the LTO indexing step will emit
      // object file names into for any that were included in the link as
      // determined by the linker's symbol resolution. It will be used to
      // provide the inputs for the subsequent final native object link.
      // Note that the paths emitted into this file will have their prefixes
      // replaced with the final output directory, so they will be the paths
      // of the native object files not the input bitcode files.
      linkerParamFileRootPath =
          ParameterFile.derivePath(output.getRootRelativePath(), "lto-final");
      thinltoParamFile =
          linkArtifactFactory.create(ruleContext, configuration, linkerParamFileRootPath);
    }

    final ImmutableList<Artifact> actionOutputs;
    if (isLTOIndexing) {
      ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
      for (LTOBackendArtifacts ltoA : allLTOArtifacts) {
        ltoA.addIndexingOutputs(builder);
      }
      if (thinltoParamFile != null) {
        builder.add(thinltoParamFile);
      }
      actionOutputs = builder.build();
    } else {
      actionOutputs =
          constructOutputs(
              output,
              Iterables.concat(linkstampMap.values(), linkActionOutputs.build()),
              interfaceOutputLibrary == null ? null : interfaceOutputLibrary.getArtifact(),
              symbolCounts);
    }

    ImmutableList<LinkerInput> runtimeLinkerInputs =
        ImmutableList.copyOf(LinkerInputs.simpleLinkerInputs(runtimeInputs, runtimeType));

    PathFragment paramRootPath =
        ParameterFile.derivePath(output.getRootRelativePath(), (isLTOIndexing) ? "lto-index" : "2");

    @Nullable
    final Artifact paramFile =
        canSplitCommandLine()
            ? linkArtifactFactory.create(ruleContext, configuration, paramRootPath)
            : null;

    // Add build variables necessary to template link args into the crosstool.
    Variables.Builder buildVariablesBuilder = new Variables.Builder();
    CppLinkVariablesExtension variablesExtension =
        isLTOIndexing
            ? new CppLinkVariablesExtension(
                configuration,
                ImmutableMap.<Artifact, Artifact>of(),
                needWholeArchive,
                linkerInputs,
                runtimeLinkerInputs,
                null,
                paramFile,
                thinltoParamFile,
                ltoOutputRootPrefix,
                null,
                null)
            : new CppLinkVariablesExtension(
                configuration,
                linkstampMap,
                needWholeArchive,
                linkerInputs,
                runtimeLinkerInputs,
                output,
                paramFile,
                thinltoParamFile,
                PathFragment.EMPTY_FRAGMENT,
                toolchain.getInterfaceSoBuilder(),
                interfaceOutput);
    variablesExtension.addVariables(buildVariablesBuilder);
    for (VariablesExtension extraVariablesExtension : variablesExtensions) {
      extraVariablesExtension.addVariables(buildVariablesBuilder);
    }
    Variables buildVariables = buildVariablesBuilder.build();

    Preconditions.checkArgument(
        linkType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
        "you can't link an interface dynamic library directly");
    if (linkType != LinkTargetType.DYNAMIC_LIBRARY) {
      Preconditions.checkArgument(
          interfaceOutput == null,
          "interface output may only be non-null for dynamic library links");
    }
    if (linkType.staticness() == Staticness.STATIC) {
      // solib dir must be null for static links
      runtimeSolibDir = null;

      Preconditions.checkArgument(
          linkStaticness == LinkStaticness.FULLY_STATIC, "static library link must be static");
      Preconditions.checkArgument(
          symbolCounts == null, "the symbol counts output must be null for static links");
      Preconditions.checkArgument(
          !isNativeDeps, "the native deps flag must be false for static links");
      Preconditions.checkArgument(
          !needWholeArchive, "the need whole archive flag must be false for static links");
    }

    LinkCommandLine.Builder linkCommandLineBuilder =
        new LinkCommandLine.Builder(configuration, getOwner(), ruleContext)
            .setLinkerInputs(linkerInputs)
            .setRuntimeInputs(runtimeLinkerInputs)
            .setLinkTargetType(linkType)
            .setLinkStaticness(linkStaticness)
            .setFeatures(features)
            .setRuntimeSolibDir(linkType.staticness() == Staticness.STATIC ? null : runtimeSolibDir)
            .setNativeDeps(isNativeDeps)
            .setUseTestOnlyFlags(useTestOnlyFlags)
            .setParamFile(paramFile)
            .setToolchain(toolchain)
            .setFdoSupport(fdoSupport.getFdoSupport())
            .setBuildVariables(buildVariables)
            .setToolPath(getToolPath())
            .setFeatureConfiguration(featureConfiguration);

    if (!isLTOIndexing) {
      linkCommandLineBuilder
          .setOutput(output)
          .setBuildInfoHeaderArtifacts(buildInfoHeaderArtifacts)
          .setLinkstamps(linkstampMap)
          .setLinkopts(ImmutableList.copyOf(linkopts))
          .addLinkstampCompileOptions(linkstampOptions);
    } else {
      List<String> opts = new ArrayList<>(linkopts);
      opts.addAll(featureConfiguration.getCommandLine("lto-indexing", buildVariables));
      opts.addAll(cppConfiguration.getLTOIndexOptions());
      linkCommandLineBuilder.setLinkopts(ImmutableList.copyOf(opts));
    }

    LinkCommandLine linkCommandLine = linkCommandLineBuilder.build();

    // Compute the set of inputs - we only need stable order here.
    NestedSetBuilder<Artifact> dependencyInputsBuilder = NestedSetBuilder.stableOrder();
    dependencyInputsBuilder.addTransitive(crosstoolInputs);
    dependencyInputsBuilder.add(toolchain.getLinkDynamicLibraryTool());
    dependencyInputsBuilder.addTransitive(linkActionInputs.build());
    if (runtimeMiddleman != null) {
      dependencyInputsBuilder.add(runtimeMiddleman);
    }
    if (!isLTOIndexing) {
      dependencyInputsBuilder.addAll(buildInfoHeaderArtifacts);
      dependencyInputsBuilder.addAll(linkstamps);
      dependencyInputsBuilder.addTransitive(compilationInputs.build());
    }

    Iterable<Artifact> expandedInputs =
        LinkerInputs.toLibraryArtifacts(
            Link.mergeInputsDependencies(
                uniqueLibraries, needWholeArchive, cppConfiguration.archiveType()));
    Iterable<Artifact> expandedNonLibraryInputs = LinkerInputs.toLibraryArtifacts(objectFileInputs);

    if (!isLTOIndexing && allLTOArtifacts != null) {
      // We are doing LTO, and this is the real link, so substitute
      // the LTO bitcode files with the real object files they were translated into.
      Map<Artifact, Artifact> ltoMapping = new HashMap<>();
      for (LTOBackendArtifacts a : allLTOArtifacts) {
        ltoMapping.put(a.getBitcodeFile(), a.getObjectFile());
      }

      // Handle libraries.
      List<Artifact> renamedInputs = new ArrayList<>();
      for (Artifact a : expandedInputs) {
        Artifact renamed = ltoMapping.get(a);
        renamedInputs.add(renamed == null ? a : renamed);
      }
      expandedInputs = renamedInputs;

      // Handle non-libraries.
      List<Artifact> renamedNonLibraryInputs = new ArrayList<>();
      for (Artifact a : expandedNonLibraryInputs) {
        Artifact renamed = ltoMapping.get(a);
        renamedNonLibraryInputs.add(renamed == null ? a : renamed);
      }
      expandedNonLibraryInputs = renamedNonLibraryInputs;
    } else if (isLTOIndexing && allLTOArtifacts != null) {
      for (LTOBackendArtifacts a : allLTOArtifacts) {
        List<String> argv = new ArrayList<>();
        argv.addAll(cppConfiguration.getLinkOptions());
        argv.addAll(cppConfiguration.getCompilerOptions(features));
        a.setCommandLine(argv);

        a.scheduleLTOBackendAction(
            ruleContext,
            featureConfiguration,
            toolchain,
            fdoSupport,
            usePicForLTOBackendActions,
            useFissionForLTOBackendActions);
      }
    }

    // getPrimaryInput returns the first element, and that is a public interface - therefore the
    // order here is important.
    IterablesChain.Builder<Artifact> inputsBuilder =
        IterablesChain.<Artifact>builder()
            .add(ImmutableList.copyOf(expandedNonLibraryInputs))
            .add(ImmutableList.copyOf(nonCodeInputs))
            .add(dependencyInputsBuilder.build())
            .add(ImmutableIterable.from(expandedInputs));

    if (thinltoParamFile != null && !isLTOIndexing) {
      inputsBuilder.add(ImmutableList.of(thinltoParamFile));
    }
    if (linkCommandLine.getParamFile() != null) {
      inputsBuilder.add(ImmutableList.of(linkCommandLine.getParamFile()));
      Action parameterFileWriteAction =
          new ParameterFileWriteAction(
              getOwner(),
              paramFile,
              linkCommandLine.paramCmdLine(),
              ParameterFile.ParameterFileType.UNQUOTED,
              ISO_8859_1);
      analysisEnvironment.registerAction(parameterFileWriteAction);
    }

    ImmutableMap<String, String> toolchainEnv =
        featureConfiguration.getEnvironmentVariables(getActionName(), buildVariables);

    // If the crosstool uses action_configs to configure cc compilation, collect execution info
    // from there, otherwise, use no execution info.
    // TODO(b/27903698): Assert that the crosstool has an action_config for this action.
    ImmutableSet.Builder<String> executionRequirements = ImmutableSet.<String>builder();
    if (featureConfiguration.actionIsConfigured(getActionName())) {
      executionRequirements.addAll(
          featureConfiguration.getToolForAction(getActionName()).getExecutionRequirements());
    }

    return new CppLinkAction(
        getOwner(),
        mnemonic,
        inputsBuilder.deduplicate().build(),
        actionOutputs,
        cppConfiguration,
        outputLibrary,
        output,
        interfaceOutputLibrary,
        fake,
        isLTOIndexing,
        allLTOArtifacts,
        linkCommandLine,
        configuration.getVariableShellEnvironment(),
        configuration.getLocalShellEnvironment(),
        toolchainEnv,
        executionRequirements.build());
  }

  /**
   * Returns the tool path from feature configuration, if the tool in the configuration is sane, or
   * builtin tool, if configuration has a dummy value.
   */
  private String getToolPath() {
    if (!featureConfiguration.actionIsConfigured(linkType.getActionName())) {
      return null;
    }
    String toolPath =
        featureConfiguration
            .getToolForAction(linkType.getActionName())
            .getToolPath(cppConfiguration.getCrosstoolTopPathFragment())
            .getPathString();
    if (linkType.equals(LinkTargetType.DYNAMIC_LIBRARY)
        && !featureConfiguration.hasConfiguredLinkerPathInActionConfig()) {
      toolPath = toolchain.getLinkDynamicLibraryTool().getExecPathString();
    }
    return toolPath;
  }

  /** The default heuristic on whether we need to use whole-archive for the link. */
  private static boolean needWholeArchive(
      LinkStaticness staticness,
      LinkTargetType type,
      Collection<String> linkopts,
      boolean isNativeDeps,
      CppConfiguration cppConfig) {
    boolean fullyStatic = (staticness == LinkStaticness.FULLY_STATIC);
    boolean mostlyStatic = (staticness == LinkStaticness.MOSTLY_STATIC);
    boolean sharedLinkopts =
        type == LinkTargetType.DYNAMIC_LIBRARY
            || linkopts.contains("-shared")
            || cppConfig.getLinkOptions().contains("-shared");
    return (isNativeDeps || cppConfig.legacyWholeArchive())
        && (fullyStatic || mostlyStatic)
        && sharedLinkopts;
  }

  private static ImmutableList<Artifact> constructOutputs(
      Artifact primaryOutput, Iterable<Artifact> outputList, Artifact... outputs) {
    return new ImmutableList.Builder<Artifact>()
        .add(primaryOutput)
        .addAll(outputList)
        .addAll(CollectionUtils.asListWithoutNulls(outputs))
        .build();
  }

  /**
   * Translates a collection of linkstamp source files to an immutable mapping from source files to
   * object files. In other words, given a set of source files, this method determines the output
   * path to which each file should be compiled.
   *
   * @param linkstamps collection of linkstamp source files
   * @param ruleContext the rule for which this link is being performed
   * @param outputBinary the binary output path for this link
   * @return an immutable map that pairs each source file with the corresponding object file that
   *     should be fed into the link
   */
  public static ImmutableMap<Artifact, Artifact> mapLinkstampsToOutputs(
      Collection<Artifact> linkstamps,
      RuleContext ruleContext,
      BuildConfiguration configuration,
      Artifact outputBinary,
      LinkArtifactFactory linkArtifactFactory) {
    ImmutableMap.Builder<Artifact, Artifact> mapBuilder = ImmutableMap.builder();

    PathFragment outputBinaryPath = outputBinary.getRootRelativePath();
    PathFragment stampOutputDirectory =
        outputBinaryPath
            .getParentDirectory()
            .getRelative(CppHelper.OBJS)
            .getRelative(outputBinaryPath.getBaseName());

    for (Artifact linkstamp : linkstamps) {
      PathFragment stampOutputPath =
          stampOutputDirectory.getRelative(
              FileSystemUtils.replaceExtension(linkstamp.getRootRelativePath(), ".o"));
      mapBuilder.put(
          linkstamp,
          // Note that link stamp actions can be shared between link actions that output shared
          // native dep libraries.
          linkArtifactFactory.create(ruleContext, configuration, stampOutputPath));
    }
    return mapBuilder.build();
  }

  protected ActionOwner getOwner() {
    return ruleContext.getActionOwner();
  }
  
  /** Sets the mnemonic for the link action. */
  public CppLinkActionBuilder setMnemonic(String mnemonic) {
    this.mnemonic = mnemonic;
    return this;
  }

  /** Set the crosstool inputs required for the action. */
  public CppLinkActionBuilder setCrosstoolInputs(NestedSet<Artifact> inputs) {
    this.crosstoolInputs = inputs;
    return this;
  }

  /** Sets the feature configuration for the action. */
  public CppLinkActionBuilder setFeatureConfiguration(FeatureConfiguration featureConfiguration) {
    this.featureConfiguration = featureConfiguration;
    return this;
  }

  /**
   * This is the LTO indexing step, rather than the real link.
   *
   * <p>When using this, build() will store allLTOArtifacts as a side-effect so the next build()
   * call can emit the real link. Do not call addInput() between the two build() calls.
   */
  public CppLinkActionBuilder setLTOIndexing(boolean ltoIndexing) {
    this.isLTOIndexing = ltoIndexing;
    return this;
  }

  /** Sets flag for using PIC in any scheduled LTO Backend actions. */
  public CppLinkActionBuilder setUsePicForLTOBackendActions(boolean usePic) {
    this.usePicForLTOBackendActions = usePic;
    return this;
  }

  /** Sets flag for using Fission in any scheduled LTO Backend actions. */
  public CppLinkActionBuilder setUseFissionForLTOBackendActions(boolean useFission) {
    this.useFissionForLTOBackendActions = useFission;
    return this;
  }

  /** Sets the C++ runtime library inputs for the action. */
  public CppLinkActionBuilder setRuntimeInputs(
      ArtifactCategory runtimeType, Artifact middleman, NestedSet<Artifact> inputs) {
    Preconditions.checkArgument((middleman == null) == inputs.isEmpty());
    this.runtimeType = runtimeType;
    this.runtimeMiddleman = middleman;
    this.runtimeInputs = inputs;
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
   * type is {@code DYNAMIC_LIBRARY} and fake is false.
   */
  public CppLinkActionBuilder setInterfaceOutput(Artifact interfaceOutput) {
    this.interfaceOutput = interfaceOutput;
    return this;
  }

  public CppLinkActionBuilder setSymbolCountsOutput(Artifact symbolCounts) {
    this.symbolCounts = symbolCounts;
    return this;
  }

  /**
   * Add additional inputs needed for the linkstamp compilation that is being done as part of the
   * link.
   */
  public CppLinkActionBuilder addCompilationInputs(Iterable<Artifact> inputs) {
    this.compilationInputs.addAll(inputs);
    return this;
  }

  public CppLinkActionBuilder addTransitiveCompilationInputs(NestedSet<Artifact> inputs) {
    this.compilationInputs.addTransitive(inputs);
    return this;
  }

  private void addObjectFile(LinkerInput input) {
    // We skip file extension checks for TreeArtifacts because they represent directory artifacts
    // without a file extension.
    String name = input.getArtifact().getFilename();
    Preconditions.checkArgument(
        input.getArtifact().isTreeArtifact() || Link.OBJECT_FILETYPES.matches(name), name);
    this.objectFiles.add(input);
  }

  public CppLinkActionBuilder addLTOBitcodeFiles(ImmutableMap<Artifact, Artifact> files) {
    Preconditions.checkState(ltoBitcodeFiles == null);
    ltoBitcodeFiles = files;
    return this;
  }

  /**
   * Adds a single object file to the set of inputs.
   */
  public CppLinkActionBuilder addObjectFile(Artifact input) {
    addObjectFile(LinkerInputs.simpleLinkerInput(input, ArtifactCategory.OBJECT_FILE));
    return this;
  }

  /**
   * Adds object files to the linker action.
   */
  public CppLinkActionBuilder addObjectFiles(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addObjectFile(LinkerInputs.simpleLinkerInput(input, ArtifactCategory.OBJECT_FILE));
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

  private void checkLibrary(LibraryToLink input) {
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
  public CppLinkActionBuilder addLibrary(LibraryToLink input) {
    checkLibrary(input);
    libraries.add(input);
    return this;
  }

  /**
   * Adds multiple artifact to the set of inputs. The artifacts must be archives or shared
   * libraries.
   */
  public CppLinkActionBuilder addLibraries(NestedSet<LibraryToLink> inputs) {
    for (LibraryToLink input : inputs) {
      checkLibrary(input);
    }
    this.libraries.addTransitive(inputs);
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
   * wherever possible). The default is {@link LinkStaticness#FULLY_STATIC}.
   */
  public CppLinkActionBuilder setLinkStaticness(LinkStaticness linkStaticness) {
    this.linkStaticness = linkStaticness;
    return this;
  }

  /**
   * Sets the identifier of the library produced by the action. See
   * {@link LinkerInputs.LibraryToLink#getLibraryIdentifier()}
   */
  public CppLinkActionBuilder setLibraryIdentifier(String libraryIdentifier) {
    this.libraryIdentifier = libraryIdentifier;
    return this;
  }

  /**
   * Adds a C++ source file which will be compiled at link time. This is used to embed various
   * values from the build system into binaries to identify their provenance.
   *
   * <p>Link stamps are also automatically added to the inputs.
   */
  public CppLinkActionBuilder addLinkstamps(Map<Artifact, NestedSet<Artifact>> linkstamps) {
    this.linkstamps.addAll(linkstamps.keySet());
    // Add inputs for linkstamping.
    if (!linkstamps.isEmpty()) {
      for (Map.Entry<Artifact, NestedSet<Artifact>> entry : linkstamps.entrySet()) {
        addCompilationInputs(entry.getValue());
      }
    }
    return this;
  }

  public CppLinkActionBuilder addLinkstampCompilerOptions(ImmutableList<String> linkstampOptions) {
    this.linkstampOptions = linkstampOptions;
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
      CcLinkParams linkParams, RuleErrorConsumer errorListener) throws InterruptedException {
    addLinkopts(linkParams.flattenedLinkopts());
    addLibraries(linkParams.getLibraries());
    ExtraLinkTimeLibraries extraLinkTimeLibraries = linkParams.getExtraLinkTimeLibraries();
    if (extraLinkTimeLibraries != null) {
      for (ExtraLinkTimeLibrary extraLibrary : extraLinkTimeLibraries.getExtraLibraries()) {
        addLibraries(extraLibrary.buildLibraries(ruleContext));
      }
    }
    addLinkstamps(CppHelper.resolveLinkstamps(errorListener, linkParams));
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

  /**
   * Sets the name of the directory where the solib symlinks for the dynamic runtime libraries live.
   * This is usually automatically set from the cc_toolchain.
   */
  public CppLinkActionBuilder setRuntimeSolibDir(PathFragment runtimeSolibDir) {
    this.runtimeSolibDir = runtimeSolibDir;
    return this;
  }
  
  /**
   * Adds an extra input artifact to the link action.
   */
  public CppLinkActionBuilder addActionInput(Artifact input) {
    this.linkActionInputs.add(input);
    return this;
  }
  
  /**
   * Adds extra input artifacts to the link action.
   */
  public CppLinkActionBuilder addActionInputs(Iterable<Artifact> inputs) {
    this.linkActionInputs.addAll(inputs);
    return this;
  }
  
  /**
   * Adds extra input artifacts to the link actions.
   */
  public CppLinkActionBuilder addTransitiveActionInputs(NestedSet<Artifact> inputs) {
    this.linkActionInputs.addTransitive(inputs);
    return this;
  }

  /** Adds an extra output artifact to the link action. */
  public CppLinkActionBuilder addActionOutput(Artifact output) {
    this.linkActionOutputs.add(output);
    return this;
  }

  private static class LinkArgCollector {
    ImmutableSet<String> runtimeLibrarySearchDirectories;
    ImmutableSet<String> librarySearchDirectories;
    SequenceBuilder librariesToLink;

    public void setRuntimeLibrarySearchDirectories(
        ImmutableSet<String> runtimeLibrarySearchDirectories) {
      this.runtimeLibrarySearchDirectories = runtimeLibrarySearchDirectories;
    }

    public void setLibrariesToLink(SequenceBuilder librariesToLink) {
      this.librariesToLink = librariesToLink;
    }

    public void setLibrarySearchDirectories(ImmutableSet<String> librarySearchDirectories) {
      this.librarySearchDirectories = librarySearchDirectories;
    }

    public ImmutableSet<String> getRuntimeLibrarySearchDirectories() {
      return runtimeLibrarySearchDirectories;
    }

    public SequenceBuilder getLibrariesToLink() {
      return librariesToLink;
    }

    public ImmutableSet<String> getLibrarySearchDirectories() {
      return librarySearchDirectories;
    }

  }

  private class CppLinkVariablesExtension implements VariablesExtension {

    private final BuildConfiguration configuration;
    private final ImmutableMap<Artifact, Artifact> linkstampMap;
    private final boolean needWholeArchive;
    private final Iterable<LinkerInput> linkerInputs;
    private final ImmutableList<LinkerInput> runtimeLinkerInputs;
    private final Artifact outputArtifact;
    private final Artifact interfaceLibraryBuilder;
    private final Artifact interfaceLibraryOutput;
    private final Artifact paramFile;
    private final Artifact thinltoParamFile;
    private final PathFragment ltoOutputRootPrefix;

    private final LinkArgCollector linkArgCollector = new LinkArgCollector();

    public CppLinkVariablesExtension(
        BuildConfiguration configuration,
        ImmutableMap<Artifact, Artifact> linkstampMap,
        boolean needWholeArchive,
        Iterable<LinkerInput> linkerInputs,
        ImmutableList<LinkerInput> runtimeLinkerInputs,
        Artifact output,
        Artifact paramFile,
        Artifact thinltoParamFile,
        PathFragment ltoOutputRootPrefix,
        Artifact interfaceLibraryBuilder,
        Artifact interfaceLibraryOutput) {
      this.configuration = configuration;
      this.linkstampMap = linkstampMap;
      this.needWholeArchive = needWholeArchive;
      this.linkerInputs = linkerInputs;
      this.runtimeLinkerInputs = runtimeLinkerInputs;
      this.outputArtifact = output;
      this.interfaceLibraryBuilder = interfaceLibraryBuilder;
      this.interfaceLibraryOutput = interfaceLibraryOutput;
      this.paramFile = paramFile;
      this.thinltoParamFile = thinltoParamFile;
      this.ltoOutputRootPrefix = ltoOutputRootPrefix;

      addInputFileLinkOptions(linkArgCollector);
    }

    @Override
    public void addVariables(Variables.Builder buildVariables) {

      // symbol counting
      if (symbolCounts != null) {
        buildVariables.addStringVariable(
            SYMBOL_COUNTS_OUTPUT_VARIABLE, symbolCounts.getExecPathString());
      }

      // linkstamp
      ImmutableSet.Builder<String> linkstampPaths = ImmutableSet.<String>builder();
      for (Artifact linkstampOutput : linkstampMap.values()) {
        linkstampPaths.add(linkstampOutput.getExecPathString());
      }

      buildVariables.addStringSequenceVariable(
          LINKSTAMP_PATHS_VARIABLE, linkstampPaths.build());

      // pic
      if (cppConfiguration.forcePic()) {
        buildVariables.addStringVariable(FORCE_PIC_VARIABLE, "");
      }

      if (cppConfiguration.shouldStripBinaries()) {
        buildVariables.addStringVariable(STRIP_DEBUG_SYMBOLS_VARIABLE, "");
      }

      if (getLinkType().staticness().equals(Staticness.DYNAMIC) && cppConfiguration.useFission()) {
        buildVariables.addStringVariable(IS_USING_FISSION_VARIABLE, "");
      }

      if (useTestOnlyFlags()) {
        buildVariables.addStringVariable(IS_CC_TEST_LINK_ACTION_VARIABLE, "");
      } else {
        buildVariables.addStringVariable(IS_NOT_CC_TEST_LINK_ACTION_VARIABLE, "");
      }

      if (linkArgCollector.getRuntimeLibrarySearchDirectories() != null) {
        buildVariables.addStringSequenceVariable(
            RUNTIME_LIBRARY_SEARCH_DIRECTORIES_VARIABLE,
            linkArgCollector.getRuntimeLibrarySearchDirectories());
      }

      buildVariables.addCustomBuiltVariable(
          LIBRARIES_TO_LINK_VARIABLE, linkArgCollector.getLibrariesToLink());

      buildVariables.addStringSequenceVariable(
          LIBRARY_SEARCH_DIRECTORIES_VARIABLE, linkArgCollector.getLibrarySearchDirectories());

      if (paramFile != null) {
        buildVariables.addStringVariable(LINKER_PARAM_FILE_VARIABLE, paramFile.getExecPathString());
      }

      // mostly static
      if (linkStaticness == LinkStaticness.MOSTLY_STATIC && cppConfiguration.skipStaticOutputs()) {
        buildVariables.addStringVariable(SKIP_MOSTLY_STATIC_VARIABLE, "");
      }

      // output exec path
      if (outputArtifact != null) {
        buildVariables.addStringVariable(
            OUTPUT_EXECPATH_VARIABLE, outputArtifact.getExecPathString());
      }

      if (isLTOIndexing()) {
        if (thinltoParamFile != null) {
          // This is a lto-indexing action and we want it to populate param file.
          buildVariables.addStringVariable(
              THINLTO_INDEXING_PARAM_FILE_VARIABLE, thinltoParamFile.getExecPathString());
          // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
          buildVariables.addStringVariable(
              "thinlto_optional_params_file", "=" + thinltoParamFile.getExecPathString());
        } else {
          buildVariables.addStringVariable(THINLTO_INDEXING_PARAM_FILE_VARIABLE, "");
          // TODO(b/33846234): Remove once all the relevant crosstools don't depend on the variable.
          buildVariables.addStringVariable("thinlto_optional_params_file", "");
        }
        buildVariables.addStringVariable(
            THINLTO_PREFIX_REPLACE_VARIABLE,
            configuration.getBinDirectory().getExecPathString()
                + ";"
                + configuration.getBinDirectory().getExecPath().getRelative(ltoOutputRootPrefix));
        buildVariables.addStringVariable(
            THINLTO_OBJECT_SUFFIX_REPLACE_VARIABLE,
            Iterables.getOnlyElement(CppFileTypes.LTO_INDEXING_OBJECT_FILE.getExtensions())
                + ";"
                + Iterables.getOnlyElement(CppFileTypes.OBJECT_FILE.getExtensions()));
      } else {
        if (thinltoParamFile != null) {
          // This is a normal link action and we need to use param file created by lto-indexing.
          buildVariables.addStringVariable(
              THINLTO_PARAM_FILE_VARIABLE, thinltoParamFile.getExecPathString());
        }
      }
      boolean shouldGenerateInterfaceLibrary =
          outputArtifact != null
              && interfaceLibraryBuilder != null
              && interfaceLibraryOutput != null;
      buildVariables.addStringVariable(
          GENERATE_INTERFACE_LIBRARY_VARIABLE, shouldGenerateInterfaceLibrary ? "yes" : "no");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_BUILDER_VARIABLE,
          shouldGenerateInterfaceLibrary ? interfaceLibraryBuilder.getExecPathString() : "ignored");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_INPUT_VARIABLE,
          shouldGenerateInterfaceLibrary ? outputArtifact.getExecPathString() : "ignored");
      buildVariables.addStringVariable(
          INTERFACE_LIBRARY_OUTPUT_VARIABLE,
          shouldGenerateInterfaceLibrary ? interfaceLibraryOutput.getExecPathString() : "ignored");

      // Variables arising from the toolchain
      buildVariables
          .addAllStringVariables(toolchain.getBuildVariables())
          .build();
      fdoSupport.getFdoSupport().getLinkOptions(featureConfiguration, buildVariables);
    }

    private boolean isLTOIndexing() {
      return !ltoOutputRootPrefix.equals(PathFragment.EMPTY_FRAGMENT);
    }

    private boolean isSharedNativeLibrary() {
      return isNativeDeps && cppConfiguration.shareNativeDeps();
    }

    /**
     * When linking a shared library fully or mostly static then we need to link in *all* dependent
     * files, not just what the shared library needs for its own code. This is done by wrapping all
     * objects/libraries with -Wl,-whole-archive and -Wl,-no-whole-archive. For this case the
     * globalNeedWholeArchive parameter must be set to true. Otherwise only library objects (.lo)
     * need to be wrapped with -Wl,-whole-archive and -Wl,-no-whole-archive.
     *
     * <p>TODO: Factor out of the bazel binary into build variables for crosstool action_configs.
     */
    private void addInputFileLinkOptions(LinkArgCollector linkArgCollector) {
      ImmutableSet.Builder<String> librarySearchDirectories = ImmutableSet.builder();
      ImmutableSet.Builder<String> runtimeRpathRoots = ImmutableSet.builder();
      ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps = ImmutableSet.builder();

      // List of command line parameters that need to be placed *outside* of
      // --whole-archive ... --no-whole-archive.
      SequenceBuilder librariesToLink = new SequenceBuilder();

      PathFragment solibDir =
          configuration
              .getBinDirectory(ruleContext.getRule().getRepository())
              .getExecPath()
              .getRelative(cppConfiguration.getSolibDirectory());
      String runtimeSolibName = runtimeSolibDir != null ? runtimeSolibDir.getBaseName() : null;
      boolean runtimeRpath =
          runtimeSolibDir != null
              && (linkType == LinkTargetType.DYNAMIC_LIBRARY
                  || (linkType == LinkTargetType.EXECUTABLE
                      && linkStaticness == LinkStaticness.DYNAMIC));

      if (runtimeRpath) {
        if (isNativeDeps) {
          runtimeRpathRoots.add(".");
        }
        runtimeRpathRoots.add(runtimeSolibName + "/");
      }

      String rpathRoot;
      // Calculate the correct relative value for the "-rpath" link option (which sets
      // the search path for finding shared libraries).
      if (isSharedNativeLibrary()) {
        // For shared native libraries, special symlinking is applied to ensure C++
        // runtimes are available under $ORIGIN/_solib_[arch]. So we set the RPATH to find
        // them.
        //
        // Note that we have to do this because $ORIGIN points to different paths for
        // different targets. In other words, blaze-bin/d1/d2/d3/a_shareddeps.so and
        // blaze-bin/d4/b_shareddeps.so have different path depths. The first could
        // reference a standard blaze-bin/_solib_[arch] via $ORIGIN/../../../_solib[arch],
        // and the second could use $ORIGIN/../_solib_[arch]. But since this is a shared
        // artifact, both are symlinks to the same place, so
        // there's no *one* RPATH setting that fits all targets involved in the sharing.
        rpathRoot = cppConfiguration.getSolibDirectory() + "/";
        if (runtimeRpath) {
          runtimeRpathRoots.add("../" + runtimeSolibName + "/");
        }
      } else {
        // For all other links, calculate the relative path from the output file to _solib_[arch]
        // (the directory where all shared libraries are stored, which resides under the blaze-bin
        // directory. In other words, given blaze-bin/my/package/binary, rpathRoot would be
        // "../../_solib_[arch]".
        if (runtimeRpath) {
          runtimeRpathRoots.add(
              Strings.repeat("../", output.getRootRelativePath().segmentCount() - 1)
                  + runtimeSolibName
                  + "/");
        }

        rpathRoot =
            Strings.repeat("../", output.getRootRelativePath().segmentCount() - 1)
                + cppConfiguration.getSolibDirectory()
                + "/";

        if (isNativeDeps) {
          // We also retain the $ORIGIN/ path to solibs that are in _solib_<arch>, as opposed to
          // the package directory)
          if (runtimeRpath) {
            runtimeRpathRoots.add("../" + runtimeSolibName + "/");
          }
        }
      }

      Map<Artifact, Artifact> ltoMap = generateLtoMap();
      boolean includeSolibDir =
          addLinkerInputs(
              librarySearchDirectories,
              rpathRootsForExplicitSoDeps,
              librariesToLink,
              solibDir,
              rpathRoot,
              ltoMap);
      boolean includeRuntimeSolibDir =
          addRuntimeLinkerInputs(
              librarySearchDirectories,
              rpathRootsForExplicitSoDeps,
              librariesToLink,
              solibDir,
              rpathRoot,
              ltoMap);
      Preconditions.checkState(
          ltoMap == null || ltoMap.isEmpty(), "Still have LTO objects left: %s", ltoMap);

      ImmutableSet.Builder<String> runtimeLibrarySearchDirectories = ImmutableSet.builder();
      // rpath ordering matters for performance; first add the one where most libraries are found.
      if (includeSolibDir) {
        runtimeLibrarySearchDirectories.add(rpathRoot);
      }
      runtimeLibrarySearchDirectories.addAll(rpathRootsForExplicitSoDeps.build());
      if (includeRuntimeSolibDir) {
        runtimeLibrarySearchDirectories.addAll(runtimeRpathRoots.build());
      }

      linkArgCollector.setLibrarySearchDirectories(librarySearchDirectories.build());
      linkArgCollector.setRuntimeLibrarySearchDirectories(runtimeLibrarySearchDirectories.build());
      linkArgCollector.setLibrariesToLink(librariesToLink);
    }

    private Map<Artifact, Artifact> generateLtoMap() {
      if (isLTOIndexing || allLTOArtifacts == null) {
        return null;
      }
      // TODO(bazel-team): The LTO final link can only work if there are individual .o files on
      // the command line. Rather than crashing, this should issue a nice error. We will get
      // this by
      // 1) moving supports_start_end_lib to a toolchain feature
      // 2) having thin_lto require start_end_lib
      // As a bonus, we can rephrase --nostart_end_lib as --features=-start_end_lib and get rid
      // of a command line option.

      Preconditions.checkState(cppConfiguration.useStartEndLib());
      Map<Artifact, Artifact> ltoMap = new HashMap<>();
      for (LTOBackendArtifacts l : allLTOArtifacts) {
        ltoMap.put(l.getBitcodeFile(), l.getObjectFile());
      }
      return ltoMap;
    }

    private boolean addRuntimeLinkerInputs(
        Builder<String> librarySearchDirectories,
        Builder<String> rpathRootsForExplicitSoDeps,
        SequenceBuilder librariesToLink,
        PathFragment solibDir,
        String rpathRoot,
        Map<Artifact, Artifact> ltoMap) {
      boolean includeRuntimeSolibDir = false;
      for (LinkerInput input : runtimeLinkerInputs) {
        if (input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY) {
          PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
          Preconditions.checkState(
              runtimeSolibDir != null && libDir.equals(runtimeSolibDir),
              "Artifact '%s' is not under directory '%s'.",
              input.getArtifact(),
              solibDir);
          includeRuntimeSolibDir = true;
          addDynamicInputLinkOptions(
              input,
              librariesToLink,
              librarySearchDirectories,
              rpathRootsForExplicitSoDeps,
              solibDir,
              rpathRoot);
        } else {
          addStaticInputLinkOptions(input, librariesToLink, true, ltoMap);
        }
      }
      return includeRuntimeSolibDir;
    }

    private boolean addLinkerInputs(
        Builder<String> librarySearchDirectories,
        Builder<String> rpathEntries,
        SequenceBuilder librariesToLink,
        PathFragment solibDir,
        String rpathRoot,
        Map<Artifact, Artifact> ltoMap) {
      boolean includeSolibDir = false;
      for (LinkerInput input : linkerInputs) {
        if (input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY) {
          PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
          Preconditions.checkState(
              libDir.startsWith(solibDir),
              "Artifact '%s' is not under directory '%s'.",
              input.getArtifact(),
              solibDir);
          if (libDir.equals(solibDir)) {
            includeSolibDir = true;
          }
          addDynamicInputLinkOptions(
              input,
              librariesToLink,
              librarySearchDirectories,
              rpathEntries,
              solibDir,
              rpathRoot);
        } else {
          addStaticInputLinkOptions(input, librariesToLink, false, ltoMap);
        }
      }
      return includeSolibDir;
    }

    /**
     * Adds command-line options for a dynamic library input file into options and libOpts.
     *
     * @param librariesToLink - a collection that will be exposed as a build variable.
     */
    private void addDynamicInputLinkOptions(
        LinkerInput input,
        SequenceBuilder librariesToLink,
        ImmutableSet.Builder<String> librarySearchDirectories,
        ImmutableSet.Builder<String> rpathRootsForExplicitSoDeps,
        PathFragment solibDir,
        String rpathRoot) {
      Preconditions.checkState(input.getArtifactCategory() == ArtifactCategory.DYNAMIC_LIBRARY);
      Preconditions.checkState(!Link.useStartEndLib(input, cppConfiguration.archiveType()));

      Artifact inputArtifact = input.getArtifact();
      PathFragment libDir = inputArtifact.getExecPath().getParentDirectory();
      if (!libDir.equals(solibDir)
          && (runtimeSolibDir == null || !runtimeSolibDir.equals(libDir))) {
        String dotdots = "";
        PathFragment commonParent = solibDir;
        while (!libDir.startsWith(commonParent)) {
          dotdots += "../";
          commonParent = commonParent.getParentDirectory();
        }

        rpathRootsForExplicitSoDeps.add(
            rpathRoot + dotdots + libDir.relativeTo(commonParent).getPathString());
      }

      librarySearchDirectories.add(
          inputArtifact.getExecPath().getParentDirectory().getPathString());

      String name = inputArtifact.getFilename();
      if (CppFileTypes.SHARED_LIBRARY.matches(name)) {
        // Use normal shared library resolution rules for shared libraries.
        String libName = name.replaceAll("(^lib|\\.(so|dylib)$)", "");
        librariesToLink.addValue(
            LibraryToLinkValue.forDynamicLibrary(libName));
      } else if (CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name)) {
        // Versioned shared libraries require the exact library filename, e.g.:
        // -lfoo -> libfoo.so
        // -l:libfoo.so.1 -> libfoo.so.1
        librariesToLink.addValue(
            LibraryToLinkValue.forVersionedDynamicLibrary(name));
      } else {
        // Interface shared objects have a non-standard extension
        // that the linker won't be able to find.  So use the
        // filename directly rather than a -l option.  Since the
        // library has an SONAME attribute, this will work fine.
        librariesToLink.addValue(
            LibraryToLinkValue.forInterfaceLibrary(inputArtifact.getExecPathString()));
      }
    }

    /**
     * Adds command-line options for a static library or non-library input into options.
     *
     * @param librariesToLink - a collection that will be exposed as a build variable.
     * @param ltoMap is a mutable list of exec paths that should be on the command-line, which must
     */
    private void addStaticInputLinkOptions(
        LinkerInput input,
        SequenceBuilder librariesToLink,
        boolean isRuntimeLinkerInput,
        @Nullable Map<Artifact, Artifact> ltoMap) {
      ArtifactCategory artifactCategory = input.getArtifactCategory();
      Preconditions.checkState(artifactCategory != ArtifactCategory.DYNAMIC_LIBRARY);
      // If we had any LTO artifacts, ltoMap whould be non-null. In that case,
      // we should have created a thinltoParamFile which the LTO indexing
      // step will populate with the exec paths that correspond to the LTO
      // artifacts that the linker decided to include based on symbol resolution.
      // Those files will be included directly in the link (and not wrapped
      // in --start-lib/--end-lib) to ensure consistency between the two link
      // steps.
      Preconditions.checkState(ltoMap == null || thinltoParamFile != null);

      // start-lib/end-lib library: adds its input object files.
      if (Link.useStartEndLib(input, cppConfiguration.archiveType())) {
        Iterable<Artifact> archiveMembers = input.getObjectFiles();
        if (!Iterables.isEmpty(archiveMembers)) {
          ImmutableList.Builder<String> nonLTOArchiveMembersBuilder = ImmutableList.builder();
          for (Artifact member : archiveMembers) {
            if (ltoMap != null && ltoMap.remove(member) != null) {
              // The LTO artifacts that should be included in the final link
              // are listed in the thinltoParamFile. When ltoMap is non-null
              // the backend artifact may be missing due to libraries that list .o
              // files explicitly, or generate .o files from assembler.
              continue;
            }
            nonLTOArchiveMembersBuilder.add(member.getExecPathString());
          }
          ImmutableList<String> nonLTOArchiveMembers = nonLTOArchiveMembersBuilder.build();
          if (!nonLTOArchiveMembers.isEmpty()) {
            boolean inputIsWholeArchive = !isRuntimeLinkerInput && needWholeArchive;
            librariesToLink.addValue(
                LibraryToLinkValue.forObjectFileGroup(nonLTOArchiveMembers, inputIsWholeArchive));
          }
        }
      } else {
        Preconditions.checkArgument(
            artifactCategory.equals(ArtifactCategory.OBJECT_FILE)
                || artifactCategory.equals(ArtifactCategory.STATIC_LIBRARY)
                || artifactCategory.equals(ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY));
        boolean isAlwaysLinkStaticLibrary =
            artifactCategory == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY;
        boolean inputIsWholeArchive =
            (!isRuntimeLinkerInput && (isAlwaysLinkStaticLibrary || needWholeArchive))
                || (isRuntimeLinkerInput && isAlwaysLinkStaticLibrary && !needWholeArchive);

        Artifact inputArtifact = input.getArtifact();
        if (ltoMap != null && ltoMap.remove(inputArtifact) != null) {
          // The LTO artifacts that should be included in the final link
          // are listed in the thinltoParamFile.
          return;
        }

        String name;
        if (input.isFake()) {
          name = Link.FAKE_OBJECT_PREFIX + inputArtifact.getExecPathString();
        } else {
          name = inputArtifact.getExecPathString();
        }

        librariesToLink.addValue(
            artifactCategory.equals(ArtifactCategory.OBJECT_FILE)
                ? LibraryToLinkValue.forObjectFile(name, inputIsWholeArchive)
                : LibraryToLinkValue.forStaticLibrary(name, inputIsWholeArchive));
      }
    }
  }
}
