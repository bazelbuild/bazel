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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
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
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Action that represents a linking step.
 */
@ThreadCompatible
public final class CppLinkAction extends AbstractAction {
  /**
   * An abstraction for creating intermediate and output artifacts for C++ linking.
   *
   * <p>This is unfortunately necessary, because most of the time, these artifacts are well-behaved
   * ones sitting under a package directory, but nativedeps link actions can be shared. In order to
   * avoid creating every artifact here with {@code getShareableArtifact()}, we abstract the
   * artifact creation away.
   */
  public interface LinkArtifactFactory {
    /**
     * Create an artifact at the specified root-relative path in the bin directory.
     */
    Artifact create(RuleContext ruleContext, PathFragment rootRelativePath);
  }

  /**
   * An implementation of {@link LinkArtifactFactory} that can only create artifacts in the package
   * directory.
   */
  public static final LinkArtifactFactory DEFAULT_ARTIFACT_FACTORY = new LinkArtifactFactory() {
    @Override
    public Artifact create(RuleContext ruleContext, PathFragment rootRelativePath) {
      return ruleContext.getDerivedArtifact(rootRelativePath,
          ruleContext.getConfiguration().getBinDirectory());
    }
  };

  private static final String LINK_GUID = "58ec78bd-1176-4e36-8143-439f656b181d";
  private static final String FAKE_LINK_GUID = "da36f819-5a15-43a9-8a45-e01b60e10c8b";

  private final CppConfiguration cppConfiguration;
  private final LibraryToLink outputLibrary;
  private final LibraryToLink interfaceOutputLibrary;

  private final LinkCommandLine linkCommandLine;

  /** True for cc_fake_binary targets. */
  private final boolean fake;
  private final boolean isLTOIndexing;

  // This is set for both LTO indexing and LTO linking.
  @Nullable private final Iterable<LTOBackendArtifacts> allLTOBackendArtifacts;
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
  private CppLinkAction(
      ActionOwner owner,
      Iterable<Artifact> inputs,
      ImmutableList<Artifact> outputs,
      CppConfiguration cppConfiguration,
      LibraryToLink outputLibrary,
      LibraryToLink interfaceOutputLibrary,
      boolean fake,
      boolean isLTOIndexing,
      Iterable<LTOBackendArtifacts> allLTOBackendArtifacts,
      LinkCommandLine linkCommandLine) {
    super(owner, inputs, outputs);
    this.mandatoryInputs = inputs;
    this.cppConfiguration = cppConfiguration;
    this.outputLibrary = outputLibrary;
    this.interfaceOutputLibrary = interfaceOutputLibrary;
    this.fake = fake;
    this.isLTOIndexing = isLTOIndexing;
    this.allLTOBackendArtifacts = allLTOBackendArtifacts;
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

  public ImmutableMap<String, String> getEnvironment() {
    if (OS.getCurrent() == OS.WINDOWS) {
      // TODO(bazel-team): Both GCC and clang rely on their execution directories being on
      // PATH, otherwise they fail to find dependent DLLs (and they fail silently...). On
      // the other hand, Windows documentation says that the directory of the executable
      // is always searched for DLLs first. Not sure what to make of it.
      // Other options are to forward the system path (brittle), or to add a PATH field to
      // the crosstool file.
      //
      // @see com.google.devtools.build.lib.rules.cpp.CppCompileAction#getEnvironment.
      return ImmutableMap.of(
          "PATH",
          cppConfiguration.getToolPathFragment(CppConfiguration.Tool.GCC).getParentDirectory()
              .getPathString()
      );
    }
    return ImmutableMap.of();
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
  public final List<String> getCommandLine() {
    return linkCommandLine.getCommandLine();
  }

  Iterable<LTOBackendArtifacts> getAllLTOBackendArtifacts() {
    return allLTOBackendArtifacts;
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
        if ((CppFileTypes.OBJECT_FILE.matches(objectFile.getFilename())
                || CppFileTypes.PIC_OBJECT_FILE.matches(objectFile.getFilename()))
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
    f.addBoolean(isLTOIndexing);
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
    message.append(
        ShellEscaper.escapeString(getCppConfiguration().getLdExecutable().getPathString()));
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
  public String getMnemonic() {
    return (isLTOIndexing) ? "CppLTOIndexing" : "CppLink";
  }

  @Override
  protected String getRawProgressMessage() {
    return (isLTOIndexing ? "LTO indexing " : "Linking ")
        + outputLibrary.getArtifact().prettyPrint();
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
  public static boolean enableSymbolsCounts(
      CppConfiguration cppConfiguration, boolean fake, LinkTargetType linkType) {
    return cppConfiguration.getSymbolCounts()
        && cppConfiguration.supportsGoldLinker()
        && linkType == LinkTargetType.EXECUTABLE
        && !fake;
  }

  public static PathFragment symbolCountsFileName(PathFragment binaryName) {
    return binaryName.replaceName(binaryName.getBaseName() + ".sc");
  }

  /**
   * Builder class to construct {@link CppLinkAction}s.
   */
  public static class Builder {
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
    private LinkArtifactFactory linkArtifactFactory = DEFAULT_ARTIFACT_FACTORY;

    private boolean isLTOIndexing = false;
    private Iterable<LTOBackendArtifacts> allLTOArtifacts = null;

    /**
     * Creates a builder that builds {@link CppLinkAction} instances.
     *
     * @param ruleContext the rule that owns the action
     * @param output the output artifact
     */
    public Builder(RuleContext ruleContext, Artifact output) {
      this(ruleContext, output, ruleContext.getConfiguration(),
          ruleContext.getAnalysisEnvironment(), CppHelper.getToolchain(ruleContext));
    }

    /**
     * Creates a builder that builds {@link CppLinkAction} instances.
     *
     * @param ruleContext the rule that owns the action
     * @param output the output artifact
     */
    public Builder(RuleContext ruleContext, Artifact output,
        BuildConfiguration configuration, CcToolchainProvider toolchain) {
      this(ruleContext, output, configuration,
          ruleContext.getAnalysisEnvironment(), toolchain);
    }

    /**
     * Creates a builder that builds {@link CppLinkAction}s.
     *
     * @param ruleContext the rule that owns the action
     * @param output the output artifact
     * @param configuration the configuration used to determine the tool chain
     *        and the default link options
     */
    private Builder(@Nullable RuleContext ruleContext, Artifact output,
        BuildConfiguration configuration, AnalysisEnvironment analysisEnvironment,
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
     * Given a Context, creates a Builder that builds {@link CppLinkAction}s.
     * Note well: Keep the Builder->Context and Context->Builder transforms consistent!
     * @param ruleContext the rule that owns the action
     * @param output the output artifact
     * @param linkContext an immutable CppLinkAction.Context from the original builder
     */
    public Builder(RuleContext ruleContext, Artifact output, Context linkContext,
        BuildConfiguration configuration) {
      // These Builder-only fields get set in the constructor:
      //   ruleContext, analysisEnvironment, outputPath, configuration, runtimeSolibDir
      this(ruleContext, output, configuration, ruleContext.getAnalysisEnvironment(),
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

    public CppLinkAction.Builder setLinkArtifactFactory(LinkArtifactFactory linkArtifactFactory) {
      this.linkArtifactFactory = linkArtifactFactory;
      return this;
    }

    private Iterable<LTOBackendArtifacts> createLTOArtifacts(
        PathFragment ltoOutputRootPrefix, NestedSet<LibraryToLink> uniqueLibraries) {
      Set<Artifact> compiled = new LinkedHashSet<>();
      for (LibraryToLink lib : uniqueLibraries) {
        Iterables.addAll(compiled, lib.getLTOBitcodeFiles());
      }

      // This flattens the set of object files, so for M binaries and N .o files,
      // this is O(M*N). If we had a nested set of .o files, we could have O(M + N) instead.
      NestedSetBuilder<Artifact> bitcodeBuilder = NestedSetBuilder.stableOrder();
      for (LibraryToLink lib : uniqueLibraries) {
        if (!lib.containsObjectFiles()) {
          continue;
        }
        for (Artifact a : lib.getObjectFiles()) {
          if (compiled.contains(a)) {
            bitcodeBuilder.add(a);
          }
        }
      }
      for (LinkerInput input : nonLibraries) {
        // This relies on file naming conventions. It would be less fragile to have a dedicated
        // field for non-library .o files.
        if (CppFileTypes.OBJECT_FILE.matches(input.getArtifact().getExecPath())
            || CppFileTypes.PIC_OBJECT_FILE.matches(input.getArtifact().getExecPath())) {
          bitcodeBuilder.add(input.getArtifact());
        }
      }

      NestedSet<Artifact> allBitcode = bitcodeBuilder.build();

      ImmutableList.Builder<LTOBackendArtifacts> ltoOutputs = ImmutableList.builder();
      for (Artifact a : allBitcode) {
        LTOBackendArtifacts ltoArtifacts = new LTOBackendArtifacts(
            ltoOutputRootPrefix, a, allBitcode, ruleContext, linkArtifactFactory);
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

    /**
     * Builds the Action as configured and returns it.
     */
    public CppLinkAction build() {
      if (interfaceOutput != null && (fake || linkType != LinkTargetType.DYNAMIC_LIBRARY)) {
        throw new RuntimeException("Interface output can only be used "
                                   + "with non-fake DYNAMIC_LIBRARY targets");
      }

      final ImmutableList<Artifact> buildInfoHeaderArtifacts = !linkstamps.isEmpty()
          ? ruleContext.getBuildInfo(CppBuildInfo.KEY)
          : ImmutableList.<Artifact>of();

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
          LinkerInputs.newInputLibrary(output, filteredNonLibraryArtifacts, this.ltoBitcodeFiles);
      final LibraryToLink interfaceOutputLibrary =
          (interfaceOutput == null)
              ? null
              : LinkerInputs.newInputLibrary(
                  interfaceOutput, filteredNonLibraryArtifacts, this.ltoBitcodeFiles);

      final ImmutableMap<Artifact, Artifact> linkstampMap =
          mapLinkstampsToOutputs(linkstamps, ruleContext, output, linkArtifactFactory);

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
              ? linkArtifactFactory.create(ruleContext, paramRootPath)
              : null;

      LinkCommandLine.Builder linkCommandLineBuilder =
          new LinkCommandLine.Builder(configuration, getOwner(), ruleContext)
              .setLinkerInputs(linkerInputs)
              .setRuntimeInputs(
                  ImmutableList.copyOf(LinkerInputs.simpleLinkerInputs(runtimeInputs)))
              .setLinkTargetType(linkType)
              .setLinkStaticness(linkStaticness)
              .setFeatures(features)
              .setRuntimeSolibDir(linkType.isStaticLibraryLink() ? null : runtimeSolibDir)
              .setNativeDeps(isNativeDeps)
              .setUseTestOnlyFlags(useTestOnlyFlags)
              .setNeedWholeArchive(needWholeArchive)
              .setParamFile(paramFile)
              .setAllLTOArtifacts(isLTOIndexing ? null : allLTOArtifacts)
              .setToolchain(toolchain);

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
        opts.add("-flto");
        opts.add(
            "-Wl,-plugin-opt,thin-lto="
                + configuration.getBinDirectory().getExecPathString()
                + ":"
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
        Collection<Artifact> linkstamps, RuleContext ruleContext, Artifact outputBinary,
        LinkArtifactFactory linkArtifactFactory) {
      ImmutableMap.Builder<Artifact, Artifact> mapBuilder = ImmutableMap.builder();

      PathFragment outputBinaryPath = outputBinary.getRootRelativePath();
      PathFragment stampOutputDirectory = outputBinaryPath.getParentDirectory().
          getRelative("_objs").getRelative(outputBinaryPath.getBaseName());

      for (Artifact linkstamp : linkstamps) {
        PathFragment stampOutputPath = stampOutputDirectory.getRelative(
            FileSystemUtils.replaceExtension(linkstamp.getRootRelativePath(), ".o"));
        mapBuilder.put(linkstamp,
            // Note that link stamp actions can be shared between link actions that output shared
            // native dep libraries.
            linkArtifactFactory.create(ruleContext, stampOutputPath));
      }
      return mapBuilder.build();    }

    protected ActionOwner getOwner() {
      return ruleContext.getActionOwner();
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
     * This is the LTO indexing step, rather than the real link.
     *
     * <p>When using this, build() will store allLTOArtifacts as a side-effect so the next build()
     * call can emit the real link. Do not call addInput() between the two build() calls.
     *
     */
    public Builder setLTOIndexing(boolean ltoIndexing) {
      this.isLTOIndexing = ltoIndexing;
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
    public Builder setInterfaceOutput(Artifact interfaceOutput) {
      this.interfaceOutput = interfaceOutput;
      return this;
    }

    public Builder setSymbolCountsOutput(Artifact symbolCounts) {
      this.symbolCounts = symbolCounts;
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

    public Builder addLTOBitcodeFiles(Iterable<Artifact> files) {
      for (Artifact a : files) {
        ltoBitcodeFiles.add(a);
      }
      return this;
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
          Link.ARCHIVE_LIBRARY_FILETYPES.matches(name)
              || Link.SHARED_LIBRARY_FILETYPES.matches(name),
          "'%s' is not a library file",
          input);
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
      ExtraLinkTimeLibraries extraLinkTimeLibraries = linkParams.getExtraLinkTimeLibraries();
      if (extraLinkTimeLibraries != null) {
        for (ExtraLinkTimeLibrary extraLibrary : extraLinkTimeLibraries.getExtraLibraries()) {
          addLibraries(extraLibrary.buildLibraries(ruleContext));
        }
      }
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
        final Artifact output, BuildConfiguration config) {
      return new Builder(null, output, config, analysisEnvironment, null) {
        @Override
        protected ActionOwner getOwner() {
          return owner;
        }
      };
    }
  }

  /**
   * TransitiveInfoProvider for ELF link actions.
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
