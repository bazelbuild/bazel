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
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.Context;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
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

  /** A build variable for clang flags that set the root of the linker search path. */
  public static final String RUNTIME_ROOT_FLAGS_VARIABLE = "runtime_root_flags";

  /** A build variable for entries in the linker search path. */
  public static final String RUNTIME_ROOT_ENTRIES_VARIABLE = "runtime_root_entries";

  /**
   * A build variable for options applying to specific libraries in the linker invocation that
   * either identify a library to be linked or add a directory to the runtime library search path.
   */
  public static final String LIBOPTS_VARIABLE = "libopts";

  /**
   * A build variable for flags providing files to link as inputs in the linker invocation that
   * should not go in a -whole_archive block.
   */
  public static final String LINKER_INPUT_PARAMS_VARIABLE = "linker_input_params";

  /**
   * A build variable for flags providing files to link as inputs in the linker invocation that
   * should go in a -whole_archive block.
   */
  public static final String WHOLE_ARCHIVE_LINKER_INPUT_PARAMS_VARIABLE =
      "whole_archive_linker_params";

  /** A build variable whose presence indicates that whole archive flags should be applied. */
  public static final String GLOBAL_WHOLE_ARCHIVE_VARIABLE = "global_whole_archive";

  /** A build variable for the execpath of the output of the linker. */
  public static final String OUTPUT_EXECPATH_VARIABLE = "output_execpath";

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

  // Builder-only
  // Null when invoked from tests (e.g. via createTestBuilder).
  @Nullable private final RuleContext ruleContext;
  private final AnalysisEnvironment analysisEnvironment;
  private final Artifact output;

  // can be null for CppLinkAction.createTestBuilder()
  @Nullable private final CcToolchainProvider toolchain;
  private Artifact interfaceOutput;
  private Artifact symbolCounts;
  private PathFragment runtimeSolibDir;
  protected final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;
  private FeatureConfiguration featureConfiguration;

  // Morally equivalent with {@link Context}, except these are mutable.
  // Keep these in sync with {@link Context}.
  private final Set<LinkerInput> nonLibraries = new LinkedHashSet<>();
  private final NestedSetBuilder<LibraryToLink> libraries = NestedSetBuilder.linkOrder();
  private NestedSet<Artifact> crosstoolInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private Artifact runtimeMiddleman;
  private NestedSet<Artifact> runtimeInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private final NestedSetBuilder<Artifact> compilationInputs = NestedSetBuilder.stableOrder();
  private final Set<Artifact> linkstamps = new LinkedHashSet<>();
  private List<String> linkstampOptions = new ArrayList<>();
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private LinkStaticness linkStaticness = LinkStaticness.FULLY_STATIC;
  private String libraryIdentifier = null;
  private List<Artifact> ltoBitcodeFiles = new ArrayList<>();

  private boolean fake;
  private boolean isNativeDeps;
  private boolean useTestOnlyFlags;
  private boolean wholeArchive;
  private LinkArtifactFactory linkArtifactFactory = CppLinkAction.DEFAULT_ARTIFACT_FACTORY;

  private boolean isLTOIndexing = false;
  private Iterable<LTOBackendArtifacts> allLTOArtifacts = null;

  /**
   * Creates a builder that builds {@link CppLinkAction} instances.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   */
  public CppLinkActionBuilder(RuleContext ruleContext, Artifact output) {
    this(
        ruleContext,
        output,
        ruleContext.getConfiguration(),
        ruleContext.getAnalysisEnvironment(),
        CppHelper.getToolchain(ruleContext));
  }

  /**
   * Creates a builder that builds {@link CppLinkAction} instances.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   */
  public CppLinkActionBuilder(
      RuleContext ruleContext,
      Artifact output,
      BuildConfiguration configuration,
      CcToolchainProvider toolchain) {
    this(ruleContext, output, configuration, ruleContext.getAnalysisEnvironment(), toolchain);
  }

  /**
   * Creates a builder that builds {@link CppLinkAction}s.
   *
   * @param ruleContext the rule that owns the action
   * @param output the output artifact
   * @param configuration the configuration used to determine the tool chain and the default link
   *     options
   */
  private CppLinkActionBuilder(
      @Nullable RuleContext ruleContext,
      Artifact output,
      BuildConfiguration configuration,
      AnalysisEnvironment analysisEnvironment,
      CcToolchainProvider toolchain) {
    this.ruleContext = ruleContext;
    this.analysisEnvironment = Preconditions.checkNotNull(analysisEnvironment);
    this.output = Preconditions.checkNotNull(output);
    this.configuration = Preconditions.checkNotNull(configuration);
    this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
    this.toolchain = toolchain;
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
   */
  public CppLinkActionBuilder(
      RuleContext ruleContext,
      Artifact output,
      Context linkContext,
      BuildConfiguration configuration) {
    // These Builder-only fields get set in the constructor:
    //   ruleContext, analysisEnvironment, outputPath, configuration, runtimeSolibDir
    this(
        ruleContext,
        output,
        configuration,
        ruleContext.getAnalysisEnvironment(),
        CppHelper.getToolchain(ruleContext));
    Preconditions.checkNotNull(linkContext);

    // All linkContext fields should be transferred to this Builder.
    this.nonLibraries.addAll(linkContext.nonLibraries);
    this.libraries.addTransitive(linkContext.libraries);
    this.crosstoolInputs = linkContext.crosstoolInputs;
    this.runtimeMiddleman = linkContext.runtimeMiddleman;
    this.runtimeInputs = linkContext.runtimeInputs;
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
  public Set<LinkerInput> getNonLibraries() {
    return nonLibraries;
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
   * Returns lto bitcode files for this link action.
   */
  public List<Artifact> getLtoBitcodeFiles() {
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

  private Iterable<LTOBackendArtifacts> createLTOArtifacts(
      PathFragment ltoOutputRootPrefix, NestedSet<LibraryToLink> uniqueLibraries) {
    Set<Artifact> compiled = new LinkedHashSet<>();
    for (LibraryToLink lib : uniqueLibraries) {
      Iterables.addAll(compiled, lib.getLTOBitcodeFiles());
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
    for (LinkerInput input : nonLibraries) {
      // This relies on file naming conventions. It would be less fragile to have a dedicated
      // field for non-library .o files.
      if (CppFileTypes.OBJECT_FILE.matches(input.getArtifact().getExecPath())
          || CppFileTypes.PIC_OBJECT_FILE.matches(input.getArtifact().getExecPath())) {
        if (this.ltoBitcodeFiles.contains(input.getArtifact())) {
          allBitcode.put(input.getArtifact().getExecPath(), input.getArtifact());
        }
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
  public CppLinkAction build() {
    Preconditions.checkState(
        (libraryIdentifier == null) == (linkType == LinkTargetType.EXECUTABLE));
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

    NestedSet<LibraryToLink> uniqueLibraries = libraries.build();
    final Iterable<Artifact> filteredNonLibraryArtifacts =
        CppLinkAction.filterLinkerInputArtifacts(LinkerInputs.toLibraryArtifacts(nonLibraries));

    final Iterable<LinkerInput> linkerInputs =
        IterablesChain.<LinkerInput>builder()
            .add(ImmutableList.copyOf(CppLinkAction.filterLinkerInputs(nonLibraries)))
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
        featureConfiguration = CcCommon.configureFeatures(ruleContext);
      }
    }

    final LibraryToLink outputLibrary = LinkerInputs.newInputLibrary(
        output, libraryIdentifier, filteredNonLibraryArtifacts, this.ltoBitcodeFiles);
    final LibraryToLink interfaceOutputLibrary =
        (interfaceOutput == null)
            ? null
            : LinkerInputs.newInputLibrary(interfaceOutput, libraryIdentifier,
                filteredNonLibraryArtifacts, this.ltoBitcodeFiles);

    final ImmutableMap<Artifact, Artifact> linkstampMap =
        mapLinkstampsToOutputs(linkstamps, ruleContext, configuration, output, linkArtifactFactory);

    PathFragment ltoOutputRootPrefix = null;
    if (isLTOIndexing && allLTOArtifacts == null) {
      ltoOutputRootPrefix =
          FileSystemUtils.appendExtension(
              output.getRootRelativePath(), ".lto");
      allLTOArtifacts = createLTOArtifacts(ltoOutputRootPrefix, uniqueLibraries);
    }

    final ImmutableList<Artifact> actionOutputs;
    if (isLTOIndexing) {
      ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
      for (LTOBackendArtifacts ltoA : allLTOArtifacts) {
        ltoA.addIndexingOutputs(builder);
      }
      actionOutputs = builder.build();
    } else {
      actionOutputs =
          constructOutputs(
              output,
              linkstampMap.values(),
              interfaceOutputLibrary == null ? null : interfaceOutputLibrary.getArtifact(),
              symbolCounts);
    }

    ImmutableList<LinkerInput> runtimeLinkerInputs =
        ImmutableList.copyOf(LinkerInputs.simpleLinkerInputs(runtimeInputs));

    // Add build variables necessary to template link args into the crosstool.
    Variables.Builder buildVariablesBuilder = new Variables.Builder();
    CppLinkVariablesExtension variablesExtension =
        isLTOIndexing
            ? new CppLinkVariablesExtension(
                linkstampMap, needWholeArchive, linkerInputs, runtimeLinkerInputs, null)
            : new CppLinkVariablesExtension(
                linkstampMap,
                needWholeArchive,
                linkerInputs,
                runtimeLinkerInputs,
                output);
    variablesExtension.addVariables(buildVariablesBuilder);
    Variables buildVariables = buildVariablesBuilder.build();

    PathFragment paramRootPath =
        ParameterFile.derivePath(
            output.getRootRelativePath(), (isLTOIndexing) ? "lto" : "2");

    @Nullable
    final Artifact paramFile =
        canSplitCommandLine()
            ? linkArtifactFactory.create(ruleContext, configuration, paramRootPath)
            : null;

    Preconditions.checkArgument(
        linkType != LinkTargetType.INTERFACE_DYNAMIC_LIBRARY,
        "you can't link an interface dynamic library directly");
    if (linkType != LinkTargetType.DYNAMIC_LIBRARY) {
      Preconditions.checkArgument(
          interfaceOutput == null,
          "interface output may only be non-null for dynamic library links");
    }
    if (linkType.isStaticLibraryLink()) {
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
            .setRuntimeSolibDir(linkType.isStaticLibraryLink() ? null : runtimeSolibDir)
            .setNativeDeps(isNativeDeps)
            .setUseTestOnlyFlags(useTestOnlyFlags)
            .setParamFile(paramFile)
            .setToolchain(toolchain)
            .setBuildVariables(buildVariables)
            .setFeatureConfiguration(featureConfiguration);

    // TODO(b/30228443): Refactor noWholeArchiveInputs into action_configs, and remove this.
    if (needWholeArchive) {
      linkCommandLineBuilder.setNoWholeArchiveFlags(variablesExtension.getNoWholeArchiveInputs());
    }

    if (!isLTOIndexing) {
      linkCommandLineBuilder
          .setOutput(output)
          .setInterfaceOutput(interfaceOutput)
          .setBuildInfoHeaderArtifacts(buildInfoHeaderArtifacts)
          .setInterfaceSoBuilder(getInterfaceSoBuilder())
          .setLinkstamps(linkstampMap)
          .setLinkopts(ImmutableList.copyOf(linkopts))
          .addLinkstampCompileOptions(linkstampOptions);
    } else {
      // TODO(bazel-team): once the LLVM compiler patches have been finalized, this should
      // be converted to a crosstool feature configuration instead.
      List<String> opts = new ArrayList<>(linkopts);
      opts.add("-flto=thin");
      opts.add("-Wl,-plugin-opt,thinlto-index-only");
      opts.add("-Wl,-plugin-opt,thinlto-emit-imports-files");
      opts.add(
          "-Wl,-plugin-opt,thinlto-prefix-replace="
              + configuration.getBinDirectory().getExecPathString()
              + ";"
              + configuration
                  .getBinDirectory()
                  .getExecPath()
                  .getRelative(ltoOutputRootPrefix)
                  .toString());
      linkCommandLineBuilder.setLinkopts(ImmutableList.copyOf(opts));
    }

    LinkCommandLine linkCommandLine = linkCommandLineBuilder.build();

    // Compute the set of inputs - we only need stable order here.
    NestedSetBuilder<Artifact> dependencyInputsBuilder = NestedSetBuilder.stableOrder();
    dependencyInputsBuilder.addTransitive(crosstoolInputs);
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
    Iterable<Artifact> expandedNonLibraryInputs = LinkerInputs.toLibraryArtifacts(nonLibraries);
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
        argv.addAll(featureConfiguration.getCommandLine(getActionName(), Variables.EMPTY));
        argv.addAll(cppConfiguration.getCompilerOptions(features));
        a.setCommandLine(argv);
      }
    }

    // getPrimaryInput returns the first element, and that is a public interface - therefore the
    // order here is important.
    IterablesChain.Builder<Artifact> inputsBuilder =
        IterablesChain.<Artifact>builder()
            .add(ImmutableList.copyOf(expandedNonLibraryInputs))
            .add(dependencyInputsBuilder.build())
            .add(ImmutableIterable.from(expandedInputs));

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

    Map<String, String> toolchainEnv =
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
        inputsBuilder.deduplicate().build(),
        actionOutputs,
        cppConfiguration,
        outputLibrary,
        interfaceOutputLibrary,
        fake,
        isLTOIndexing,
        allLTOArtifacts,
        linkCommandLine,
        toolchainEnv,
        executionRequirements.build());
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
      Artifact primaryOutput, Collection<Artifact> outputList, Artifact... outputs) {
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
            .getRelative("_objs")
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

  protected Artifact getInterfaceSoBuilder() {
    return analysisEnvironment.getEmbeddedToolArtifact(CppRuleClasses.BUILD_INTERFACE_SO);
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

  /** Sets the C++ runtime library inputs for the action. */
  public CppLinkActionBuilder setRuntimeInputs(Artifact middleman, NestedSet<Artifact> inputs) {
    Preconditions.checkArgument((middleman == null) == inputs.isEmpty());
    this.runtimeMiddleman = middleman;
    this.runtimeInputs = inputs;
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

  private void addNonLibraryInput(LinkerInput input) {
    String name = input.getArtifact().getFilename();
    Preconditions.checkArgument(
        !Link.ARCHIVE_LIBRARY_FILETYPES.matches(name)
            && !Link.SHARED_LIBRARY_FILETYPES.matches(name),
        "'%s' is a library file",
        input);
    this.nonLibraries.add(input);
  }

  public CppLinkActionBuilder addLTOBitcodeFiles(Iterable<Artifact> files) {
    for (Artifact a : files) {
      ltoBitcodeFiles.add(a);
    }
    return this;
  }

  /**
   * Adds a single artifact to the set of inputs (C++ source files, header files, etc). Artifacts
   * that are not of recognized types will be used for dependency checking but will not be passed to
   * the linker. The artifact must not be an archive or a shared library.
   */
  public CppLinkActionBuilder addNonLibraryInput(Artifact input) {
    addNonLibraryInput(LinkerInputs.simpleLinkerInput(input));
    return this;
  }

  /**
   * Adds multiple artifacts to the set of inputs (C++ source files, header files, etc). Artifacts
   * that are not of recognized types will be used for dependency checking but will not be passed to
   * the linker. The artifacts must not be archives or shared libraries.
   */
  public CppLinkActionBuilder addNonLibraryInputs(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addNonLibraryInput(LinkerInputs.simpleLinkerInput(input));
    }
    return this;
  }

  public CppLinkActionBuilder addFakeNonLibraryInputs(Iterable<Artifact> inputs) {
    for (Artifact input : inputs) {
      addNonLibraryInput(LinkerInputs.fakeLinkerInput(input));
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
      CcLinkParams linkParams, RuleErrorConsumer errorListener) {
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

  private static class LinkArgCollector {
    String rpathRoot;
    List<String> rpathEntries;
    Set<String> libopts;
    List<String> linkerInputParams;
    List<String> wholeArchiveLinkerInputParams;
    List<String> noWholeArchiveInputs;

    public void setRpathRoot(String rPathRoot) {
      this.rpathRoot = rPathRoot;
    }

    public void setRpathEntries(List<String> rpathEntries) {
      this.rpathEntries = rpathEntries;
    }

    public void setLibopts(Set<String> libopts) {
      this.libopts = libopts;
    }

    public void setLinkerInputParams(List<String> linkerInputParams) {
      this.linkerInputParams = linkerInputParams;
    }

    public void setWholeArchiveLinkerInputParams(List<String> wholeArchiveInputParams) {
      this.wholeArchiveLinkerInputParams = wholeArchiveInputParams;
    }

    public void setNoWholeArchiveInputs(List<String> noWholeArchiveInputs) {
      this.noWholeArchiveInputs = noWholeArchiveInputs;
    }

    public String getRpathRoot() {
      return rpathRoot;
    }

    public List<String> getRpathEntries() {
      return rpathEntries;
    }

    public Set<String> getLibopts() {
      return libopts;
    }

    public List<String> getLinkerInputParams() {
      return linkerInputParams;
    }

    public List<String> getWholeArchiveLinkerInputParams() {
      return wholeArchiveLinkerInputParams;
    }

    public List<String> getNoWholeArchiveInputs() {
      return noWholeArchiveInputs;
    }
  }

  private class CppLinkVariablesExtension implements VariablesExtension {

    private final ImmutableMap<Artifact, Artifact> linkstampMap;
    private final boolean needWholeArchive;
    private final Iterable<LinkerInput> linkerInputs;
    private final ImmutableList<LinkerInput> runtimeLinkerInputs;
    private final Artifact outputArtifact;

    private final LinkArgCollector linkArgCollector = new LinkArgCollector();

    public CppLinkVariablesExtension(
        ImmutableMap<Artifact, Artifact> linkstampMap,
        boolean needWholeArchive,
        Iterable<LinkerInput> linkerInputs,
        ImmutableList<LinkerInput> runtimeLinkerInputs,
        Artifact output) {
      this.linkstampMap = linkstampMap;
      this.needWholeArchive = needWholeArchive;
      this.linkerInputs = linkerInputs;
      this.runtimeLinkerInputs = runtimeLinkerInputs;
      this.outputArtifact = output;

      addInputFileLinkOptions(linkArgCollector);
    }

    /**
     * Returns linker parameters indicating libraries that should not be linked inside a
     * --whole_archive block.
     *
     * <p>TODO(b/30228443): Refactor into action configs
     */
    public List<String> getNoWholeArchiveInputs() {
      return linkArgCollector.getNoWholeArchiveInputs();
    }

    @Override
    public void addVariables(Variables.Builder buildVariables) {

      // symbol counting
      if (symbolCounts != null) {
        buildVariables.addVariable(SYMBOL_COUNTS_OUTPUT_VARIABLE, symbolCounts.getExecPathString());
      }

      // linkstamp
      ImmutableSet.Builder<String> linkstampPaths = ImmutableSet.<String>builder();
      for (Artifact linkstampOutput : linkstampMap.values()) {
        linkstampPaths.add(linkstampOutput.getExecPathString());
      }

      buildVariables.addSequenceVariable(LINKSTAMP_PATHS_VARIABLE, linkstampPaths.build());

      // pic
      boolean forcePic = cppConfiguration.forcePic();
      if (forcePic) {
        buildVariables.addVariable(FORCE_PIC_VARIABLE, "");
      }

      // rpath
      if (linkArgCollector.getRpathRoot() != null) {
        buildVariables.addVariable(RUNTIME_ROOT_FLAGS_VARIABLE, linkArgCollector.getRpathRoot());
      }

      if (linkArgCollector.getRpathEntries() != null) {
        buildVariables.addSequenceVariable(
            RUNTIME_ROOT_ENTRIES_VARIABLE, linkArgCollector.getRpathEntries());
      }

      buildVariables.addSequenceVariable(LIBOPTS_VARIABLE, linkArgCollector.getLibopts());
      buildVariables.addSequenceVariable(
          LINKER_INPUT_PARAMS_VARIABLE, linkArgCollector.getLinkerInputParams());
      buildVariables.addSequenceVariable(
          WHOLE_ARCHIVE_LINKER_INPUT_PARAMS_VARIABLE,
          linkArgCollector.getWholeArchiveLinkerInputParams());

      // global archive
      if (needWholeArchive) {
        buildVariables.addVariable(GLOBAL_WHOLE_ARCHIVE_VARIABLE, "");
      }

      // mostly static
      if (linkStaticness == LinkStaticness.MOSTLY_STATIC && cppConfiguration.skipStaticOutputs()) {
        buildVariables.addVariable(SKIP_MOSTLY_STATIC_VARIABLE, "");
      }

      // output exec path
      if (this.outputArtifact != null) {
        buildVariables.addVariable(
            OUTPUT_EXECPATH_VARIABLE, this.outputArtifact.getExecPathString());
      }

      // Variables arising from the toolchain
      buildVariables
          .addAllVariables(CppHelper.getToolchain(ruleContext).getBuildVariables())
          .build();
      CppHelper.getFdoSupport(ruleContext).getLinkOptions(featureConfiguration, buildVariables);
    }

    private boolean isDynamicLibrary(LinkerInput linkInput) {
      Artifact libraryArtifact = linkInput.getArtifact();
      String name = libraryArtifact.getFilename();
      return Link.SHARED_LIBRARY_FILETYPES.matches(name) && name.startsWith("lib");
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

      // Used to collect -L and -Wl,-rpath options, ensuring that each used only once.
      Set<String> libOpts = new LinkedHashSet<>();

      // List of command line parameters to link input files (either directly or using -l).
      List<String> linkerInputParameters = new ArrayList<>();

      // List of command line parameters that need to be placed *outside* of
      // --whole-archive ... --no-whole-archive.
      List<String> noWholeArchiveInputs = new ArrayList<>();

      PathFragment solibDir =
          configuration
              .getBinDirectory()
              .getExecPath()
              .getRelative(cppConfiguration.getSolibDirectory());
      String runtimeSolibName = runtimeSolibDir != null ? runtimeSolibDir.getBaseName() : null;
      boolean runtimeRpath =
          runtimeSolibDir != null
              && (linkType == LinkTargetType.DYNAMIC_LIBRARY
                  || (linkType == LinkTargetType.EXECUTABLE
                      && linkStaticness == LinkStaticness.DYNAMIC));

      String rpathRoot = null;
      List<String> runtimeRpathEntries = new ArrayList<>();

      if (output != null) {
        String origin =
            useTestOnlyFlags && cppConfiguration.supportsExecOrigin()
                ? "$EXEC_ORIGIN/"
                : "$ORIGIN/";
        if (runtimeRpath) {
          runtimeRpathEntries.add("-Wl,-rpath," + origin + runtimeSolibName + "/");
        }

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
          rpathRoot =
              "-Wl,-rpath," + origin + ":" + origin + cppConfiguration.getSolibDirectory() + "/";
          if (runtimeRpath) {
            runtimeRpathEntries.add("-Wl,-rpath," + origin + "../" + runtimeSolibName + "/");
          }
        } else {
          // For all other links, calculate the relative path from the output file to _solib_[arch]
          // (the directory where all shared libraries are stored, which resides under the blaze-bin
          // directory. In other words, given blaze-bin/my/package/binary, rpathRoot would be
          // "../../_solib_[arch]".
          if (runtimeRpath) {
            runtimeRpathEntries.add(
                "-Wl,-rpath,"
                    + origin
                    + Strings.repeat("../", output.getRootRelativePath().segmentCount() - 1)
                    + runtimeSolibName
                    + "/");
          }

          rpathRoot =
              "-Wl,-rpath,"
                  + origin
                  + Strings.repeat("../", output.getRootRelativePath().segmentCount() - 1)
                  + cppConfiguration.getSolibDirectory()
                  + "/";

          if (isNativeDeps) {
            // We also retain the $ORIGIN/ path to solibs that are in _solib_<arch>, as opposed to
            // the package directory)
            if (runtimeRpath) {
              runtimeRpathEntries.add("-Wl,-rpath," + origin + "../" + runtimeSolibName + "/");
            }
            rpathRoot += ":" + origin;
          }
        }
      }

      boolean includeSolibDir = false;

      Map<Artifact, Artifact> ltoMap = null;
      if (!isLTOIndexing && (allLTOArtifacts != null)) {
        // TODO(bazel-team): The LTO final link can only work if there are individual .o files on
        // the command line. Rather than crashing, this should issue a nice error. We will get
        // this by
        // 1) moving supports_start_end_lib to a toolchain feature
        // 2) having thin_lto require start_end_lib
        // As a bonus, we can rephrase --nostart_end_lib as --features=-start_end_lib and get rid
        // of a command line option.

        Preconditions.checkState(cppConfiguration.useStartEndLib());
        ltoMap = new HashMap<>();
        for (LTOBackendArtifacts l : allLTOArtifacts) {
          ltoMap.put(l.getBitcodeFile(), l.getObjectFile());
        }
      }

      for (LinkerInput input : linkerInputs) {
        if (isDynamicLibrary(input)) {
          PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
          Preconditions.checkState(
              libDir.startsWith(solibDir),
              "Artifact '%s' is not under directory '%s'.",
              input.getArtifact(),
              solibDir);
          if (libDir.equals(solibDir)) {
            includeSolibDir = true;
          }
          addDynamicInputLinkOptions(input, linkerInputParameters, libOpts, solibDir, rpathRoot);
        } else {
          addStaticInputLinkOptions(input, linkerInputParameters, ltoMap);
        }
      }

      boolean includeRuntimeSolibDir = false;

      for (LinkerInput input : runtimeLinkerInputs) {
        List<String> optionsList = needWholeArchive ? noWholeArchiveInputs : linkerInputParameters;

        if (isDynamicLibrary(input)) {
          PathFragment libDir = input.getArtifact().getExecPath().getParentDirectory();
          Preconditions.checkState(
              runtimeSolibDir != null && libDir.equals(runtimeSolibDir),
              "Artifact '%s' is not under directory '%s'.",
              input.getArtifact(),
              solibDir);
          includeRuntimeSolibDir = true;
          addDynamicInputLinkOptions(input, optionsList, libOpts, solibDir, rpathRoot);
        } else {
          addStaticInputLinkOptions(input, optionsList, ltoMap);
        }
      }

      // rpath ordering matters for performance; first add the one where most libraries are found.
      if (includeSolibDir && rpathRoot != null) {
        linkArgCollector.setRpathRoot(rpathRoot);
      }
      if (includeRuntimeSolibDir) {
        linkArgCollector.setRpathEntries(runtimeRpathEntries);
      }

      linkArgCollector.setLibopts(libOpts);

      ImmutableList.Builder<String> wholeArchiveInputParams = ImmutableList.builder();
      ImmutableList.Builder<String> standardArchiveInputParams = ImmutableList.builder();
      for (String param : linkerInputParameters) {
        if (!wholeArchive && Link.LINK_LIBRARY_FILETYPES.matches(param) && !needWholeArchive) {
          wholeArchiveInputParams.add(param);
        } else {
          standardArchiveInputParams.add(param);
        }
      }

      linkArgCollector.setLinkerInputParams(standardArchiveInputParams.build());
      linkArgCollector.setWholeArchiveLinkerInputParams(wholeArchiveInputParams.build());
      linkArgCollector.setNoWholeArchiveInputs(noWholeArchiveInputs);

      if (ltoMap != null) {
        Preconditions.checkState(ltoMap.isEmpty(), "Still have LTO objects left: %s", ltoMap);
      }
    }

    /** Adds command-line options for a dynamic library input file into options and libOpts. */
    private void addDynamicInputLinkOptions(
        LinkerInput input,
        List<String> options,
        Set<String> libOpts,
        PathFragment solibDir,
        String rpathRoot) {
      Preconditions.checkState(isDynamicLibrary(input));
      Preconditions.checkState(!Link.useStartEndLib(input, cppConfiguration.archiveType()));

      Artifact inputArtifact = input.getArtifact();
      PathFragment libDir = inputArtifact.getExecPath().getParentDirectory();
      if (rpathRoot != null
          && !libDir.equals(solibDir)
          && (runtimeSolibDir == null || !runtimeSolibDir.equals(libDir))) {
        String dotdots = "";
        PathFragment commonParent = solibDir;
        while (!libDir.startsWith(commonParent)) {
          dotdots += "../";
          commonParent = commonParent.getParentDirectory();
        }

        libOpts.add(rpathRoot + dotdots + libDir.relativeTo(commonParent).getPathString());
      }

      libOpts.add("-L" + inputArtifact.getExecPath().getParentDirectory().getPathString());

      String name = inputArtifact.getFilename();
      if (CppFileTypes.SHARED_LIBRARY.matches(name)) {
        String libName = name.replaceAll("(^lib|\\.(so|dylib)$)", "");
        options.add("-l" + libName);
      } else {
        // Interface shared objects have a non-standard extension
        // that the linker won't be able to find.  So use the
        // filename directly rather than a -l option.  Since the
        // library has an SONAME attribute, this will work fine.
        options.add(inputArtifact.getExecPathString());
      }
    }

    /**
     * Adds command-line options for a static library or non-library input into options.
     *
     * @param ltoMap is a mutable list of exec paths that should be on the command-line, which must
     *     be supplied for LTO final links.
     */
    private void addStaticInputLinkOptions(
        LinkerInput input, List<String> options, @Nullable Map<Artifact, Artifact> ltoMap) {
      Preconditions.checkState(!isDynamicLibrary(input));

      // start-lib/end-lib library: adds its input object files.
      if (Link.useStartEndLib(input, cppConfiguration.archiveType())) {
        Iterable<Artifact> archiveMembers = input.getObjectFiles();
        if (!Iterables.isEmpty(archiveMembers)) {
          options.add("-Wl,--start-lib");
          for (Artifact member : archiveMembers) {
            if (ltoMap != null) {
              Artifact backend = ltoMap.remove(member);

              if (backend != null) {
                // If the backend artifact is missing, we can't print a warning because this may
                // happen normally, due libraries that list .o files explicitly, or generate .o
                // files from assembler.
                member = backend;
              }
            }

            options.add(member.getExecPathString());
          }
          options.add("-Wl,--end-lib");
        }
      } else {
        // For anything else, add the input directly.
        Artifact inputArtifact = input.getArtifact();

        if (ltoMap != null) {
          Artifact ltoArtifact = ltoMap.remove(inputArtifact);
          if (ltoArtifact != null) {
            inputArtifact = ltoArtifact;
          }
        }

        if (input.isFake()) {
          options.add(Link.FAKE_OBJECT_PREFIX + inputArtifact.getExecPathString());
        } else {
          options.add(inputArtifact.getExecPathString());
        }
      }
    }
  }
}
