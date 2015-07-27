// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.extra.CppLinkInfo;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.ImmutableIterable;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Action that represents an ELF linking step.
 */
@ThreadCompatible
public final class CppLinkAction extends AbstractAction {
  private static final String LINK_GUID = "58ec78bd-1176-4e36-8143-439f656b181d";
  private static final String FAKE_LINK_GUID = "da36f819-5a15-43a9-8a45-e01b60e10c8b";

  private final CppConfiguration cppConfiguration;
  private final LibraryToLink outputLibrary;
  private final LibraryToLink interfaceOutputLibrary;

  private final LinkCommandLine linkCommandLine;

  /** True for cc_fake_binary targets. */
  private final boolean fake;

  private final Iterable<Artifact> mandatoryInputs;

  // Linking uses a lot of memory; estimate 1 MB per input file, min 1.5 Gib.
  // It is vital to not underestimate too much here,
  // because running too many concurrent links can
  // thrash the machine to the point where it stops
  // responding to keystrokes or mouse clicks.
  // CPU and IO do not scale similarly and still use the static minimum estimate.
  public static final ResourceSet LINK_RESOURCES_PER_INPUT =
      ResourceSet.createWithRamCpuIo(1, 0, 0);

  // This defines the minimum of each resource that will be reserved.
  public static final ResourceSet MIN_STATIC_LINK_RESOURCES =
      ResourceSet.createWithRamCpuIo(1536, 1, 0.3);

  // Dynamic linking should be cheaper than static linking.
  public static final ResourceSet MIN_DYNAMIC_LINK_RESOURCES =
      ResourceSet.createWithRamCpuIo(1024, 0.3, 0.2);

  /**
   * Use {@link Builder} to create instances of this class. Also see there for
   * the documentation of all parameters.
   *
   * <p>This constructor is intentionally private and is only to be called from
   * {@link Builder#build()}.
   */
  private CppLinkAction(ActionOwner owner,
                        Iterable<Artifact> inputs,
                        ImmutableList<Artifact> outputs,
                        CppConfiguration cppConfiguration,
                        LibraryToLink outputLibrary,
                        LibraryToLink interfaceOutputLibrary,
                        boolean fake,
                        LinkCommandLine linkCommandLine) {
    super(owner, inputs, outputs);
    this.mandatoryInputs = inputs;
    this.cppConfiguration = cppConfiguration;
    this.outputLibrary = outputLibrary;
    this.interfaceOutputLibrary = interfaceOutputLibrary;
    this.fake = fake;

    this.linkCommandLine = linkCommandLine;
  }

  private static Iterable<LinkerInput> filterLinkerInputs(Iterable<LinkerInput> inputs) {
    return Iterables.filter(inputs, new Predicate<LinkerInput>() {
      @Override
      public boolean apply(LinkerInput input) {
        return Link.VALID_LINKER_INPUTS.matches(input.getArtifact().getFilename());
      }
    });
  }

  private static Iterable<Artifact> filterLinkerInputArtifacts(Iterable<Artifact> inputs) {
    return Iterables.filter(inputs, new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        return Link.VALID_LINKER_INPUTS.matches(input.getFilename());
      }
    });
  }

  private CppConfiguration getCppConfiguration() {
    return cppConfiguration;
  }

  @VisibleForTesting
  public String getTargetCpu() {
    return getCppConfiguration().getTargetCpu();
  }

  public String getHostSystemName() {
    return getCppConfiguration().getHostSystemName();
  }

  /**
   * Returns the link configuration; for correctness you should not call this method during
   * execution - only the argv is part of the action cache key, and we therefore don't guarantee
   * that the action will be re-executed if the contents change in a way that does not affect the
   * argv.
   */
  @VisibleForTesting
  public LinkCommandLine getLinkCommandLine() {
    return linkCommandLine;
  }

  public LibraryToLink getOutputLibrary() {
    return outputLibrary;
  }

  public LibraryToLink getInterfaceOutputLibrary() {
    return interfaceOutputLibrary;
  }

  /**
   * Returns the path to the output artifact produced by the linker.
   */
  public Path getOutputFile() {
    return outputLibrary.getArtifact().getPath();
  }

  @VisibleForTesting
  public List<String> getRawLinkArgv() {
    return linkCommandLine.getRawLinkArgv();
  }

  @VisibleForTesting
  public List<String> getArgv() {
    return linkCommandLine.arguments();
  }

  /**
   * Returns the command line specification for this link, included any required linkstamp
   * compilation steps. The command line may refer to a .params file.
   *
   * @return a finalized command line suitable for execution
   */
  public final List<String> getCommandLine()
      throws ExecException {
    return linkCommandLine.getCommandLine();
  }

  @Override
  @ThreadCompatible
  public void execute(
      ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException, InterruptedException {
    if (fake) {
      executeFake();
    } else {
      Executor executor = actionExecutionContext.getExecutor();

      try {
        executor.getContext(CppLinkActionContext.class).exec(
            this, actionExecutionContext);
      } catch (ExecException e) {
        throw e.toActionExecutionException("Linking of rule '" + getOwner().getLabel() + "'",
            executor.getVerboseFailures(), this);
      }
    }
  }

  @Override
  public String describeStrategy(Executor executor) {
    return fake
        ? "fake,local"
        : executor.getContext(CppLinkActionContext.class).strategyLocality(this);
  }

  // Don't forget to update FAKE_LINK_GUID if you modify this method.
  @ThreadCompatible
  private void executeFake()
      throws ActionExecutionException {
    // The uses of getLinkConfiguration in this method may not be consistent with the computed key.
    // I.e., this may be incrementally incorrect.
    final Collection<Artifact> linkstampOutputs = getLinkCommandLine().getLinkstamps().values();

    // Prefix all fake output files in the command line with $TEST_TMPDIR/.
    final String outputPrefix = "$TEST_TMPDIR/";
    List<String> escapedLinkArgv = escapeLinkArgv(linkCommandLine.getRawLinkArgv(),
        linkstampOutputs, outputPrefix);
    // Write the commands needed to build the real target to the fake target
    // file.
    StringBuilder s = new StringBuilder();
    Joiner.on('\n').appendTo(s,
        "# This is a fake target file, automatically generated.",
        "# Do not edit by hand!",
        "echo $0 is a fake target file and not meant to be executed.",
        "exit 0",
        "EOS",
        "",
        "makefile_dir=.",
        "");

    try {
      // Concatenate all the (fake) .o files into the result.
      for (LinkerInput linkerInput : getLinkCommandLine().getLinkerInputs()) {
        Artifact objectFile = linkerInput.getArtifact();
        if (CppFileTypes.OBJECT_FILE.matches(objectFile.getFilename())
            && linkerInput.isFake()) {
          s.append(FileSystemUtils.readContentAsLatin1(objectFile.getPath())); // (IOException)
        }
      }

      s.append(getOutputFile().getBaseName()).append(": ");
      for (Artifact linkstamp : linkstampOutputs) {
        s.append("mkdir -p " + outputPrefix +
            linkstamp.getExecPath().getParentDirectory() + " && ");
      }
      Joiner.on(' ').appendTo(s,
          ShellEscaper.escapeAll(linkCommandLine.finalizeAlreadyEscapedWithLinkstampCommands(
              escapedLinkArgv, outputPrefix)));
      s.append('\n');
      if (getOutputFile().exists()) {
        getOutputFile().setWritable(true); // (IOException)
      }
      FileSystemUtils.writeContent(getOutputFile(), ISO_8859_1, s.toString());
      getOutputFile().setExecutable(true); // (IOException)
      for (Artifact linkstamp : linkstampOutputs) {
        FileSystemUtils.touchFile(linkstamp.getPath());
      }
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create fake link command for rule '" +
                                         getOwner().getLabel() + ": " + e.getMessage(),
                                         this, false);
    }
  }

  /**
   * Shell-escapes the raw link command line.
   *
   * @param rawLinkArgv raw link command line
   * @param linkstampOutputs linkstamp artifacts
   * @param outputPrefix to be prepended to any outputs
   * @return escaped link command line
   */
  private List<String> escapeLinkArgv(List<String> rawLinkArgv,
      final Collection<Artifact> linkstampOutputs, final String outputPrefix) {
    final List<String> linkstampExecPaths = Artifact.asExecPaths(linkstampOutputs);
    ImmutableList.Builder<String> escapedArgs = ImmutableList.builder();
    for (String rawArg : rawLinkArgv) {
      String escapedArg;
      if (rawArg.equals(getPrimaryOutput().getExecPathString())
          || linkstampExecPaths.contains(rawArg)) {
        escapedArg = outputPrefix + ShellEscaper.escapeString(rawArg);
      } else if (rawArg.startsWith(Link.FAKE_OBJECT_PREFIX)) {
        escapedArg = outputPrefix + ShellEscaper.escapeString(
            rawArg.substring(Link.FAKE_OBJECT_PREFIX.length()));
      } else {
        escapedArg = ShellEscaper.escapeString(rawArg);
      }
      escapedArgs.add(escapedArg);
    }
    return escapedArgs.build();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    // The uses of getLinkConfiguration in this method may not be consistent with the computed key.
    // I.e., this may be incrementally incorrect.
    CppLinkInfo.Builder info = CppLinkInfo.newBuilder();
    info.addAllInputFile(Artifact.toExecPaths(
        LinkerInputs.toLibraryArtifacts(getLinkCommandLine().getLinkerInputs())));
    info.addAllInputFile(Artifact.toExecPaths(
        LinkerInputs.toLibraryArtifacts(getLinkCommandLine().getRuntimeInputs())));
    info.setOutputFile(getPrimaryOutput().getExecPathString());
    if (interfaceOutputLibrary != null) {
      info.setInterfaceOutputFile(interfaceOutputLibrary.getArtifact().getExecPathString());
    }
    info.setLinkTargetType(getLinkCommandLine().getLinkTargetType().name());
    info.setLinkStaticness(getLinkCommandLine().getLinkStaticness().name());
    info.addAllLinkStamp(Artifact.toExecPaths(getLinkCommandLine().getLinkstamps().values()));
    info.addAllBuildInfoHeaderArtifact(
        Artifact.toExecPaths(getLinkCommandLine().getBuildInfoHeaderArtifacts()));
    info.addAllLinkOpt(getLinkCommandLine().getLinkopts());

    return super.getExtraActionInfo()
        .setExtension(CppLinkInfo.cppLinkInfo, info.build());
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(fake ? FAKE_LINK_GUID : LINK_GUID);
    f.addString(getCppConfiguration().getLdExecutable().getPathString());
    f.addStrings(linkCommandLine.arguments());
    // TODO(bazel-team): For correctness, we need to ensure the invariant that all values accessed
    // during the execution phase are also covered by the key. Above, we add the argv to the key,
    // which covers most cases. Unfortunately, the extra action and fake support methods above also
    // sometimes directly access settings from the link configuration that may or may not affect the
    // key. We either need to change the code to cover them in the key computation, or change the
    // LinkConfiguration to disallow the combinations where the value of a setting does not affect
    // the argv.
    f.addBoolean(linkCommandLine.isNativeDeps());
    f.addBoolean(linkCommandLine.useTestOnlyFlags());
    if (linkCommandLine.getRuntimeSolibDir() != null) {
      f.addPath(linkCommandLine.getRuntimeSolibDir());
    }
    return f.hexDigestAndReset();
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    if (fake) {
      message.append("Fake ");
    }
    message.append(getProgressMessage());
    message.append('\n');
    message.append("  Command: ");
    message.append(ShellEscaper.escapeString(
        getCppConfiguration().getLdExecutable().getPathString()));
    message.append('\n');
    // Outputting one argument per line makes it easier to diff the results.
    for (String argument : ShellEscaper.escapeAll(linkCommandLine.arguments())) {
      message.append("  Argument: ");
      message.append(argument);
      message.append('\n');
    }
    return message.toString();
  }

  @Override
  public String getMnemonic() { return "CppLink"; }

  @Override
  protected String getRawProgressMessage() {
    return "Linking " + outputLibrary.getArtifact().prettyPrint();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return executor.getContext(CppLinkActionContext.class).estimateResourceConsumption(this);
  }

  /**
   * Estimate the resources consumed when this action is run locally.
   */
  public ResourceSet estimateResourceConsumptionLocal() {
    // It's ok if this behaves differently even if the key is identical.
    ResourceSet minLinkResources =
        getLinkCommandLine().getLinkStaticness() == Link.LinkStaticness.DYNAMIC
        ? MIN_DYNAMIC_LINK_RESOURCES
        : MIN_STATIC_LINK_RESOURCES;

    final int inputSize = Iterables.size(getLinkCommandLine().getLinkerInputs())
        + Iterables.size(getLinkCommandLine().getRuntimeInputs());

    return ResourceSet.createWithRamCpuIo(
        Math.max(inputSize * LINK_RESOURCES_PER_INPUT.getMemoryMb(),
            minLinkResources.getMemoryMb()),
        Math.max(inputSize * LINK_RESOURCES_PER_INPUT.getCpuUsage(),
            minLinkResources.getCpuUsage()),
        Math.max(inputSize * LINK_RESOURCES_PER_INPUT.getIoUsage(),
            minLinkResources.getIoUsage())
    );
  }

  @Override
  public Iterable<Artifact> getMandatoryInputs() {
    return mandatoryInputs;
  }

  /**
   * Determines whether or not this link should output a symbol counts file.
   */
  private static boolean enableSymbolsCounts(CppConfiguration cppConfiguration, boolean fake,
      LinkTargetType linkType) {
    return cppConfiguration.getSymbolCounts()
        && cppConfiguration.supportsGoldLinker()
        && linkType == LinkTargetType.EXECUTABLE
        && !fake;
  }

  /**
   * Builder class to construct {@link CppLinkAction}s.
   */
  public static class Builder {
    // Builder-only
    // Null when invoked from tests (e.g. via createTestBuilder).
    @Nullable private final RuleContext ruleContext;
    private final AnalysisEnvironment analysisEnvironment;
    private final PathFragment outputPath;

    // can be null for CppLinkAction.createTestBuilder()
    @Nullable private final CcToolchainProvider toolchain;
    private PathFragment interfaceOutputPath;
    private PathFragment runtimeSolibDir;
    protected final BuildConfiguration configuration;
    private final CppConfiguration cppConfiguration;

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
    private boolean fake;
    private boolean isNativeDeps;
    private boolean useTestOnlyFlags;
    private boolean wholeArchive;

    /**
     * Creates a builder that builds {@link CppLinkAction} instances.
     *
     * @param ruleContext the rule that owns the action
     * @param outputPath the path of the ELF file to be created, relative to the
     *        'bin' directory
     */
    public Builder(RuleContext ruleContext, PathFragment outputPath) {
      this(ruleContext, outputPath, ruleContext.getConfiguration(),
          ruleContext.getAnalysisEnvironment(), CppHelper.getToolchain(ruleContext));
    }

    /**
     * Creates a builder that builds {@link CppLinkAction} instances.
     *
     * @param ruleContext the rule that owns the action
     * @param outputPath the path of the ELF file to be created, relative to the
     *        'bin' directory
     */
    public Builder(RuleContext ruleContext, PathFragment outputPath,
        BuildConfiguration configuration, CcToolchainProvider toolchain) {
      this(ruleContext, outputPath, configuration,
          ruleContext.getAnalysisEnvironment(), toolchain);
    }

    /**
     * Creates a builder that builds {@link CppLinkAction}s.
     *
     * @param ruleContext the rule that owns the action
     * @param outputPath the path of the ELF file to be created, relative to the
     *        'bin' directory
     * @param configuration the configuration used to determine the tool chain
     *        and the default link options
     */
    private Builder(@Nullable RuleContext ruleContext, PathFragment outputPath,
        BuildConfiguration configuration, AnalysisEnvironment analysisEnvironment,
        CcToolchainProvider toolchain) {
      this.ruleContext = ruleContext;
      this.analysisEnvironment = Preconditions.checkNotNull(analysisEnvironment);
      this.outputPath = Preconditions.checkNotNull(outputPath);
      this.configuration = Preconditions.checkNotNull(configuration);
      this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
      this.toolchain = toolchain;
      if (cppConfiguration.supportsEmbeddedRuntimes() && toolchain != null) {
        runtimeSolibDir = toolchain.getDynamicRuntimeSolibDir();
      }
    }

    /**
     * Given a Context, creates a Builder that builds {@link CppLinkAction}s.
     * Note well: Keep the Builder->Context and Context->Builder transforms consistent!
     * @param ruleContext the rule that owns the action
     * @param outputPath the path of the ELF file to be created, relative to the
     *        'bin' directory
     * @param linkContext an immutable CppLinkAction.Context from the original builder
     */
    public Builder(RuleContext ruleContext, PathFragment outputPath, Context linkContext,
        BuildConfiguration configuration) {
      // These Builder-only fields get set in the constructor:
      //   ruleContext, analysisEnvironment, outputPath, configuration, runtimeSolibDir
      this(ruleContext, outputPath, configuration, ruleContext.getAnalysisEnvironment(),
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

    @VisibleForTesting
    boolean canSplitCommandLine() {
      if (toolchain == null || !toolchain.supportsParamFiles()) {
        return false;
      }

      switch (linkType) {
          // We currently can't split dynamic library links if they have interface outputs. That was
          // probably an unintended side effect of the change that introduced interface outputs.
        case DYNAMIC_LIBRARY:
          return (interfaceOutputPath == null);
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

    /**
     * Builds the Action as configured and returns it.
     *
     * <p>This method may only be called once.
     */
    public CppLinkAction build() {
      if (interfaceOutputPath != null && (fake || linkType != LinkTargetType.DYNAMIC_LIBRARY)) {
        throw new RuntimeException("Interface output can only be used "
                                   + "with non-fake DYNAMIC_LIBRARY targets");
      }

      final Artifact output = createArtifact(outputPath);
      final Artifact interfaceOutput = (interfaceOutputPath != null)
          ? createArtifact(interfaceOutputPath)
          : null;

      final ImmutableList<Artifact> buildInfoHeaderArtifacts = !linkstamps.isEmpty()
          ? ruleContext.getBuildInfo(CppBuildInfo.KEY)
          : ImmutableList.<Artifact>of();

      final Artifact symbolCountOutput = enableSymbolsCounts(cppConfiguration, fake, linkType)
          ? createArtifact(output.getRootRelativePath().replaceName(
              output.getExecPath().getBaseName() + ".sc"))
          : null;

      boolean needWholeArchive = wholeArchive || needWholeArchive(
          linkStaticness, linkType, linkopts, isNativeDeps, cppConfiguration);

      NestedSet<LibraryToLink> uniqueLibraries = libraries.build();
      final Iterable<Artifact> filteredNonLibraryArtifacts =
          filterLinkerInputArtifacts(LinkerInputs.toLibraryArtifacts(nonLibraries));
      final Iterable<LinkerInput> linkerInputs = IterablesChain.<LinkerInput>builder()
          .add(ImmutableList.copyOf(filterLinkerInputs(nonLibraries)))
          .add(ImmutableIterable.from(Link.mergeInputsCmdLine(
              uniqueLibraries, needWholeArchive, cppConfiguration.archiveType())))
          .build();

      // ruleContext can only be null during testing. This is kind of ugly.
      final ImmutableSet<String> features = (ruleContext == null)
          ? ImmutableSet.<String>of()
          : ruleContext.getFeatures();

      final LibraryToLink outputLibrary =
          LinkerInputs.newInputLibrary(output, filteredNonLibraryArtifacts);
      final LibraryToLink interfaceOutputLibrary = interfaceOutput == null ? null :
          LinkerInputs.newInputLibrary(interfaceOutput, filteredNonLibraryArtifacts);

      final ImmutableMap<Artifact, Artifact> linkstampMap =
          mapLinkstampsToOutputs(linkstamps, ruleContext, output);

      final ImmutableList<Artifact> actionOutputs =
          constructOutputs(
              outputLibrary.getArtifact(),
              linkstampMap.values(),
              interfaceOutputLibrary == null ? null : interfaceOutputLibrary.getArtifact(),
              symbolCountOutput);

      @Nullable
      final Artifact paramFile =
          canSplitCommandLine()
              ? createArtifact(
                  ParameterFile.derivePath(outputLibrary.getArtifact().getRootRelativePath()))
              : null;

      LinkCommandLine linkCommandLine =
          new LinkCommandLine.Builder(configuration, getOwner(), ruleContext)
              .setOutput(outputLibrary.getArtifact())
              .setInterfaceOutput(interfaceOutput)
              .setSymbolCountsOutput(symbolCountOutput)
              .setBuildInfoHeaderArtifacts(buildInfoHeaderArtifacts)
              .setLinkerInputs(linkerInputs)
              .setRuntimeInputs(
                  ImmutableList.copyOf(LinkerInputs.simpleLinkerInputs(runtimeInputs)))
              .setLinkTargetType(linkType)
              .setLinkStaticness(linkStaticness)
              .setLinkopts(ImmutableList.copyOf(linkopts))
              .setFeatures(features)
              .setLinkstamps(linkstampMap)
              .addLinkstampCompileOptions(linkstampOptions)
              .setRuntimeSolibDir(linkType.isStaticLibraryLink() ? null : runtimeSolibDir)
              .setNativeDeps(isNativeDeps)
              .setUseTestOnlyFlags(useTestOnlyFlags)
              .setNeedWholeArchive(needWholeArchive)
              .setInterfaceSoBuilder(getInterfaceSoBuilder())
              .setParamFile(paramFile)
              .build();

      // Compute the set of inputs - we only need stable order here.
      NestedSetBuilder<Artifact> dependencyInputsBuilder = NestedSetBuilder.stableOrder();
      dependencyInputsBuilder.addAll(buildInfoHeaderArtifacts);
      dependencyInputsBuilder.addAll(linkstamps);
      dependencyInputsBuilder.addTransitive(crosstoolInputs);
      if (runtimeMiddleman != null) {
        dependencyInputsBuilder.add(runtimeMiddleman);
      }
      dependencyInputsBuilder.addTransitive(compilationInputs.build());

      Iterable<Artifact> expandedInputs =
          LinkerInputs.toLibraryArtifacts(Link.mergeInputsDependencies(uniqueLibraries,
              needWholeArchive, cppConfiguration.archiveType()));
      // getPrimaryInput returns the first element, and that is a public interface - therefore the
      // order here is important.
      IterablesChain.Builder<Artifact> inputsBuilder = IterablesChain.<Artifact>builder()
          .add(ImmutableList.copyOf(LinkerInputs.toLibraryArtifacts(nonLibraries)))
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

      return new CppLinkAction(
          getOwner(),
          inputsBuilder.deduplicate().build(),
          actionOutputs,
          cppConfiguration,
          outputLibrary,
          interfaceOutputLibrary,
          fake,
          linkCommandLine);
    }

    /**
     * The default heuristic on whether we need to use whole-archive for the link.
     */
    private static boolean needWholeArchive(LinkStaticness staticness,
        LinkTargetType type, Collection<String> linkopts, boolean isNativeDeps,
        CppConfiguration cppConfig) {
      boolean fullyStatic = (staticness == LinkStaticness.FULLY_STATIC);
      boolean mostlyStatic = (staticness == LinkStaticness.MOSTLY_STATIC);
      boolean sharedLinkopts = type == LinkTargetType.DYNAMIC_LIBRARY
          || linkopts.contains("-shared")
          || cppConfig.getLinkOptions().contains("-shared");
      return (isNativeDeps || cppConfig.legacyWholeArchive())
          && (fullyStatic || mostlyStatic)
          && sharedLinkopts;
    }

    private static ImmutableList<Artifact> constructOutputs(Artifact primaryOutput,
        Collection<Artifact> outputList, Artifact... outputs) {
      return new ImmutableList.Builder<Artifact>()
          .add(primaryOutput)
          .addAll(outputList)
          .addAll(CollectionUtils.asListWithoutNulls(outputs))
          .build();
    }

    /**
     * Translates a collection of linkstamp source files to an immutable
     * mapping from source files to object files. In other words, given a
     * set of source files, this method determines the output path to which
     * each file should be compiled.
     *
     * @param linkstamps collection of linkstamp source files
     * @param ruleContext the rule for which this link is being performed
     * @param outputBinary the binary output path for this link
     * @return an immutable map that pairs each source file with the
     *         corresponding object file that should be fed into the link
     */
    public static ImmutableMap<Artifact, Artifact> mapLinkstampsToOutputs(
        Collection<Artifact> linkstamps, RuleContext ruleContext, Artifact outputBinary) {
      ImmutableMap.Builder<Artifact, Artifact> mapBuilder = ImmutableMap.builder();

      PathFragment outputBinaryPath = outputBinary.getRootRelativePath();
      PathFragment stampOutputDirectory = outputBinaryPath.getParentDirectory().
          getRelative("_objs").getRelative(outputBinaryPath.getBaseName());

      for (Artifact linkstamp : linkstamps) {
        PathFragment stampOutputPath = stampOutputDirectory.getRelative(
            FileSystemUtils.replaceExtension(linkstamp.getRootRelativePath(), ".o"));
        mapBuilder.put(linkstamp,
            ruleContext.getAnalysisEnvironment().getDerivedArtifact(
                stampOutputPath, outputBinary.getRoot()));
      }
      return mapBuilder.build();
    }

    protected ActionOwner getOwner() {
      return ruleContext.getActionOwner();
    }

    protected Artifact createArtifact(PathFragment rootRelativePath) {
      return analysisEnvironment.getDerivedArtifact(
          rootRelativePath, configuration.getBinDirectory());
    }

    protected Artifact getInterfaceSoBuilder() {
      return analysisEnvironment.getEmbeddedToolArtifact(CppRuleClasses.BUILD_INTERFACE_SO);
    }

    /**
     * Set the crosstool inputs required for the action.
     */
    public Builder setCrosstoolInputs(NestedSet<Artifact> inputs) {
      this.crosstoolInputs = inputs;
      return this;
    }

    /**
     * Sets the C++ runtime library inputs for the action.
     */
    public Builder setRuntimeInputs(Artifact middleman, NestedSet<Artifact> inputs) {
      Preconditions.checkArgument((middleman == null) == inputs.isEmpty());
      this.runtimeMiddleman = middleman;
      this.runtimeInputs = inputs;
      return this;
    }

    /**
     * Sets the interface output of the link.  A non-null argument can
     * only be provided if the link type is {@code DYNAMIC_LIBRARY}
     * and fake is false.
     */
    public Builder setInterfaceOutputPath(PathFragment path) {
      this.interfaceOutputPath = path;
      return this;
    }

    /**
     * Add additional inputs needed for the linkstamp compilation that is being done as part of the
     * link.
     */
    public Builder addCompilationInputs(Iterable<Artifact> inputs) {
      this.compilationInputs.addAll(inputs);
      return this;
    }

    public Builder addTransitiveCompilationInputs(NestedSet<Artifact> inputs) {
      this.compilationInputs.addTransitive(inputs);
      return this;
    }

    private void addNonLibraryInput(LinkerInput input) {
      String name = input.getArtifact().getFilename();
      Preconditions.checkArgument(
          !Link.ARCHIVE_LIBRARY_FILETYPES.matches(name)
          && !Link.SHARED_LIBRARY_FILETYPES.matches(name),
          "'%s' is a library file", input);
      this.nonLibraries.add(input);
    }
    /**
     * Adds a single artifact to the set of inputs (C++ source files, header files, etc). Artifacts
     * that are not of recognized types will be used for dependency checking but will not be passed
     * to the linker. The artifact must not be an archive or a shared library.
     */
    public Builder addNonLibraryInput(Artifact input) {
      addNonLibraryInput(LinkerInputs.simpleLinkerInput(input));
      return this;
    }

    /**
     * Adds multiple artifacts to the set of inputs (C++ source files, header files, etc).
     * Artifacts that are not of recognized types will be used for dependency checking but will
     * not be passed to the linker. The artifacts must not be archives or shared libraries.
     */
    public Builder addNonLibraryInputs(Iterable<Artifact> inputs) {
      for (Artifact input : inputs) {
        addNonLibraryInput(LinkerInputs.simpleLinkerInput(input));
      }
      return this;
    }

    public Builder addFakeNonLibraryInputs(Iterable<Artifact> inputs) {
      for (Artifact input : inputs) {
        addNonLibraryInput(LinkerInputs.fakeLinkerInput(input));
      }
      return this;
    }

    private void checkLibrary(LibraryToLink input) {
      String name = input.getArtifact().getFilename();
      Preconditions.checkArgument(
          Link.ARCHIVE_LIBRARY_FILETYPES.matches(name) ||
          Link.SHARED_LIBRARY_FILETYPES.matches(name),
          "'%s' is not a library file", input);
    }

    /**
     * Adds a single artifact to the set of inputs. The artifact must be an archive or a shared
     * library. Note that all directly added libraries are implicitly ordered before all nested
     * sets added with {@link #addLibraries}, even if added in the opposite order.
     */
    public Builder addLibrary(LibraryToLink input) {
      checkLibrary(input);
      libraries.add(input);
      return this;
    }

    /**
     * Adds multiple artifact to the set of inputs. The artifacts must be archives or shared
     * libraries.
     */
    public Builder addLibraries(NestedSet<LibraryToLink> inputs) {
      for (LibraryToLink input : inputs) {
        checkLibrary(input);
      }
      this.libraries.addTransitive(inputs);
      return this;
    }

    /**
     * Sets the type of ELF file to be created (.a, .so, .lo, executable). The
     * default is {@link LinkTargetType#STATIC_LIBRARY}.
     */
    public Builder setLinkType(LinkTargetType linkType) {
      this.linkType = linkType;
      return this;
    }

    /**
     * Sets the degree of "staticness" of the link: fully static (static binding
     * of all symbols), mostly static (use dynamic binding only for symbols from
     * glibc), dynamic (use dynamic binding wherever possible). The default is
     * {@link LinkStaticness#FULLY_STATIC}.
     */
    public Builder setLinkStaticness(LinkStaticness linkStaticness) {
      this.linkStaticness = linkStaticness;
      return this;
    }

    /**
     * Adds a C++ source file which will be compiled at link time. This is used
     * to embed various values from the build system into binaries to identify
     * their provenance.
     *
     * <p>Link stamps are also automatically added to the inputs.
     */
    public Builder addLinkstamps(Map<Artifact, ImmutableList<Artifact>> linkstamps) {
      this.linkstamps.addAll(linkstamps.keySet());
      // Add inputs for linkstamping.
      if (!linkstamps.isEmpty()) {
        // This will just be the compiler unless include scanning is disabled, in which case it will
        // include all header files. Since we insist that linkstamps declare all their headers, all
        // header files would be overkill, but that only happens when include scanning is disabled.
        addTransitiveCompilationInputs(toolchain.getCompile());
        for (Map.Entry<Artifact, ImmutableList<Artifact>> entry : linkstamps.entrySet()) {
          addCompilationInputs(entry.getValue());
        }
      }
      return this;
    }

    public Builder addLinkstampCompilerOptions(ImmutableList<String> linkstampOptions) {
      this.linkstampOptions = linkstampOptions;
      return this;
    }

    /**
     * Adds an additional linker option.
     */
    public Builder addLinkopt(String linkopt) {
      this.linkopts.add(linkopt);
      return this;
    }

    /**
     * Adds multiple linker options at once.
     *
     * @see #addLinkopt(String)
     */
    public Builder addLinkopts(Collection<String> linkopts) {
      this.linkopts.addAll(linkopts);
      return this;
    }

    /**
     * Merges the given link params into this builder by calling {@link #addLinkopts}, {@link
     * #addLibraries}, and {@link #addLinkstamps}.
     */
    public Builder addLinkParams(CcLinkParams linkParams, RuleErrorConsumer errorListener) {
      addLinkopts(linkParams.flattenedLinkopts());
      addLibraries(linkParams.getLibraries());
      addLinkstamps(CppHelper.resolveLinkstamps(errorListener, linkParams));
      return this;
    }

    /**
     * Sets whether this link action will be used for a cc_fake_binary; false by
     * default.
     */
    public Builder setFake(boolean fake) {
      this.fake = fake;
      return this;
    }

    /**
     * Sets whether this link action is used for a native dependency library.
     */
    public Builder setNativeDeps(boolean isNativeDeps) {
      this.isNativeDeps = isNativeDeps;
      return this;
    }

    /**
     * Setting this to true overrides the default whole-archive computation and force-enables
     * whole archives for every archive in the link. This is only necessary for linking executable
     * binaries that are supposed to export symbols.
     *
     * <p>Usually, the link action while use whole archives for dynamic libraries that are native
     * deps (or the legacy whole archive flag is enabled), and that are not dynamically linked.
     *
     * <p>(Note that it is possible to build dynamic libraries with cc_binary rules by specifying
     * linkshared = 1, and giving the rule a name that matches the pattern {@code
     * lib&lt;name&gt;.so}.)
     */
    public Builder setWholeArchive(boolean wholeArchive) {
      this.wholeArchive = wholeArchive;
      return this;
    }

    /**
     * Sets whether this link action should use test-specific flags (e.g. $EXEC_ORIGIN instead of
     * $ORIGIN for the solib search path or lazy binding);  false by default.
     */
    public Builder setUseTestOnlyFlags(boolean useTestOnlyFlags) {
      this.useTestOnlyFlags = useTestOnlyFlags;
      return this;
    }

    /**
     * Sets the name of the directory where the solib symlinks for the dynamic runtime libraries
     * live. This is usually automatically set from the cc_toolchain.
     */
    public Builder setRuntimeSolibDir(PathFragment runtimeSolibDir) {
      this.runtimeSolibDir = runtimeSolibDir;
      return this;
    }

    /**
     * Creates a builder without the need for a {@link RuleContext}.
     * This is to be used exclusively for testing purposes.
     *
     * <p>Link stamping is not supported if using this method.
     */
    @VisibleForTesting
    public static Builder createTestBuilder(
        final ActionOwner owner, final AnalysisEnvironment analysisEnvironment,
        final PathFragment outputPath, BuildConfiguration config) {
      return new Builder(null, outputPath, config, analysisEnvironment, null) {
        @Override
        protected Artifact createArtifact(PathFragment path) {
          return new Artifact(configuration.getBinDirectory().getPath().getRelative(path),
              configuration.getBinDirectory(), configuration.getBinFragment().getRelative(path),
              analysisEnvironment.getOwner());
        }
        @Override
        protected ActionOwner getOwner() {
          return owner;
        }
      };
    }
  }

  /**
   * Immutable ELF linker context, suitable for serialization.
   */
  @Immutable @ThreadSafe
  public static final class Context implements TransitiveInfoProvider {
    // Morally equivalent with {@link Builder}, except these are immutable.
    // Keep these in sync with {@link Builder}.
    private final ImmutableSet<LinkerInput> nonLibraries;
    private final NestedSet<LibraryToLink> libraries;
    private final NestedSet<Artifact> crosstoolInputs;
    private final Artifact runtimeMiddleman;
    private final NestedSet<Artifact> runtimeInputs;
    private final NestedSet<Artifact> compilationInputs;
    private final ImmutableSet<Artifact> linkstamps;
    private final ImmutableList<String> linkopts;
    private final LinkTargetType linkType;
    private final LinkStaticness linkStaticness;
    private final boolean fake;
    private final boolean isNativeDeps;
    private final boolean useTestOnlyFlags;

    /**
     * Given a {@link Builder}, creates a {@code Context} to pass to another target.
     * Note well: Keep the Builder->Context and Context->Builder transforms consistent!
     * @param builder a mutable {@link CppLinkAction.Builder} to clone from
     */
    public Context(Builder builder) {
      this.nonLibraries = ImmutableSet.copyOf(builder.nonLibraries);
      this.libraries = NestedSetBuilder.<LibraryToLink>linkOrder()
          .addTransitive(builder.libraries.build()).build();
      this.crosstoolInputs =
          NestedSetBuilder.<Artifact>stableOrder().addTransitive(builder.crosstoolInputs).build();
      this.runtimeMiddleman = builder.runtimeMiddleman;
      this.runtimeInputs =
          NestedSetBuilder.<Artifact>stableOrder().addTransitive(builder.runtimeInputs).build();
      this.compilationInputs = NestedSetBuilder.<Artifact>stableOrder()
          .addTransitive(builder.compilationInputs.build()).build();
      this.linkstamps = ImmutableSet.copyOf(builder.linkstamps);
      this.linkopts = ImmutableList.copyOf(builder.linkopts);
      this.linkType = builder.linkType;
      this.linkStaticness = builder.linkStaticness;
      this.fake = builder.fake;
      this.isNativeDeps = builder.isNativeDeps;
      this.useTestOnlyFlags = builder.useTestOnlyFlags;
    }
  }
}
