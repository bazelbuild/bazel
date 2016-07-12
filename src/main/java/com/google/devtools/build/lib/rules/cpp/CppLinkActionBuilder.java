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
  private CcToolchainFeatures.Variables buildVariables =
      new CcToolchainFeatures.Variables.Builder().build();

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

  /**
   * Returns linker inputs that are not libraries.
   */
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

    final LibraryToLink outputLibrary =
        LinkerInputs.newInputLibrary(output, filteredNonLibraryArtifacts, this.ltoBitcodeFiles);
    final LibraryToLink interfaceOutputLibrary =
        (interfaceOutput == null)
            ? null
            : LinkerInputs.newInputLibrary(
                interfaceOutput, filteredNonLibraryArtifacts, this.ltoBitcodeFiles);

    final ImmutableMap<Artifact, Artifact> linkstampMap =
        mapLinkstampsToOutputs(linkstamps, ruleContext, configuration, output, linkArtifactFactory);

    PathFragment ltoOutputRootPrefix = null;
    if (isLTOIndexing && allLTOArtifacts == null) {
      ltoOutputRootPrefix =
          FileSystemUtils.appendExtension(
              outputLibrary.getArtifact().getRootRelativePath(), ".lto");
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
              outputLibrary.getArtifact(),
              linkstampMap.values(),
              interfaceOutputLibrary == null ? null : interfaceOutputLibrary.getArtifact(),
              symbolCounts);
    }

    PathFragment paramRootPath =
        ParameterFile.derivePath(
            outputLibrary.getArtifact().getRootRelativePath(), (isLTOIndexing) ? "lto" : "2");

    @Nullable
    final Artifact paramFile =
        canSplitCommandLine()
            ? linkArtifactFactory.create(ruleContext, configuration, paramRootPath)
            : null;

    LinkCommandLine.Builder linkCommandLineBuilder =
        new LinkCommandLine.Builder(configuration, getOwner(), ruleContext)
            .setActionName(getActionName())
            .setLinkerInputs(linkerInputs)
            .setRuntimeInputs(ImmutableList.copyOf(LinkerInputs.simpleLinkerInputs(runtimeInputs)))
            .setLinkTargetType(linkType)
            .setLinkStaticness(linkStaticness)
            .setFeatures(features)
            .setRuntimeSolibDir(linkType.isStaticLibraryLink() ? null : runtimeSolibDir)
            .setNativeDeps(isNativeDeps)
            .setUseTestOnlyFlags(useTestOnlyFlags)
            .setNeedWholeArchive(needWholeArchive)
            .setParamFile(paramFile)
            .setAllLTOArtifacts(isLTOIndexing ? null : allLTOArtifacts)
            .setToolchain(toolchain)
            .setFeatureConfiguration(featureConfiguration);

    if (!isLTOIndexing) {
      linkCommandLineBuilder
          .setOutput(outputLibrary.getArtifact())
          .setInterfaceOutput(interfaceOutput)
          .setSymbolCountsOutput(symbolCounts)
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

    // For backwards compatibility, and for tests, we permit the link action to be
    // instantiated without a feature configuration. In this case, an empty feature
    // configuration is used.
    if (featureConfiguration == null) {
      this.featureConfiguration = new FeatureConfiguration();
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
        argv.addAll(featureConfiguration.getCommandLine(getActionName(), buildVariables));
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

  /** Sets the build variables that will be used to template the crosstool. */
  public CppLinkActionBuilder setBuildVariables(CcToolchainFeatures.Variables buildVariables) {
    this.buildVariables = buildVariables;
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
   * Adds a C++ source file which will be compiled at link time. This is used to embed various
   * values from the build system into binaries to identify their provenance.
   *
   * <p>Link stamps are also automatically added to the inputs.
   */
  public CppLinkActionBuilder addLinkstamps(Map<Artifact, NestedSet<Artifact>> linkstamps) {
    this.linkstamps.addAll(linkstamps.keySet());
    // Add inputs for linkstamping.
    if (!linkstamps.isEmpty()) {
      addTransitiveCompilationInputs(toolchain.getCompile());
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
}
