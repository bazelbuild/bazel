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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.actions.extra.CppLinkInfo;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.skylark.Args;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Action that represents a linking step. */
@ThreadCompatible
@AutoCodec
public final class CppLinkAction extends AbstractAction implements CommandAction {

  /**
   * An abstraction for creating intermediate and output artifacts for C++ linking.
   *
   * <p>This is unfortunately necessary, because most of the time, these artifacts are well-behaved
   * ones sitting under a package directory, but nativedeps link actions can be shared. In order to
   * avoid creating every artifact here with {@code getShareableArtifact()}, we abstract the
   * artifact creation away.
   */
  public interface LinkArtifactFactory {
    /** Create an artifact at the specified root-relative path in the bin directory. */
    Artifact create(
        ActionConstructionContext actionConstructionContext,
        RepositoryName repositoryName,
        BuildConfiguration configuration,
        PathFragment rootRelativePath);
  }

  /**
   * An implementation of {@link LinkArtifactFactory} that can only create artifacts in the package
   * directory.
   */
  public static final LinkArtifactFactory DEFAULT_ARTIFACT_FACTORY =
      new LinkArtifactFactory() {
        @Override
        public Artifact create(
            ActionConstructionContext actionConstructionContext,
            RepositoryName repositoryName,
            BuildConfiguration configuration,
            PathFragment rootRelativePath) {
          return actionConstructionContext.getDerivedArtifact(
              rootRelativePath, configuration.getBinDirectory(repositoryName));
        }
      };

  private static final String LINK_GUID = "58ec78bd-1176-4e36-8143-439f656b181d";
  private static final String FAKE_LINK_GUID = "da36f819-5a15-43a9-8a45-e01b60e10c8b";
  
  @Nullable private final String mnemonic;
  private final LibraryToLink outputLibrary;
  private final Artifact linkOutput;
  private final LibraryToLink interfaceOutputLibrary;
  private final ImmutableMap<String, String> toolchainEnv;
  private final ImmutableMap<String, String> executionRequirements;
  private final ImmutableList<Artifact> linkstampObjects;

  private final LinkCommandLine linkCommandLine;

  /** True for cc_fake_binary targets. */
  private final boolean fake;

  private final Iterable<Artifact> fakeLinkerInputArtifacts;
  private final boolean isLtoIndexing;

  private final PathFragment ldExecutable;
  private final String hostSystemName;
  private final String targetCpu;

  // Linking uses a lot of memory; estimate 1 MB per input file, min 1.5 Gib. It is vital to not
  // underestimate too much here, because running too many concurrent links can thrash the machine
  // to the point where it stops responding to keystrokes or mouse clicks. This is primarily a
  // problem with memory consumption, not CPU or I/O usage.
  public static final ResourceSet LINK_RESOURCES_PER_INPUT =
      ResourceSet.createWithRamCpu(1, 0);

  // This defines the minimum of each resource that will be reserved.
  public static final ResourceSet MIN_STATIC_LINK_RESOURCES =
      ResourceSet.createWithRamCpu(1536, 1);

  // Dynamic linking should be cheaper than static linking.
  public static final ResourceSet MIN_DYNAMIC_LINK_RESOURCES =
      ResourceSet.createWithRamCpu(1024, 1);

  /**
   * Use {@link CppLinkActionBuilder} to create instances of this class. Also see there for the
   * documentation of all parameters.
   *
   * <p>This constructor is intentionally private and is only to be called from {@link
   * CppLinkActionBuilder#build()}.
   */
  CppLinkAction(
      ActionOwner owner,
      String mnemonic,
      NestedSet<Artifact> inputs,
      ImmutableSet<Artifact> outputs,
      LibraryToLink outputLibrary,
      Artifact linkOutput,
      LibraryToLink interfaceOutputLibrary,
      boolean fake,
      Iterable<Artifact> fakeLinkerInputArtifacts,
      boolean isLtoIndexing,
      ImmutableList<Artifact> linkstampObjects,
      LinkCommandLine linkCommandLine,
      ActionEnvironment env,
      ImmutableMap<String, String> toolchainEnv,
      ImmutableMap<String, String> executionRequirements,
      PathFragment ldExecutable,
      String hostSystemName,
      String targetCpu) {
    super(owner, inputs, outputs, env);
    this.mnemonic = getMnemonic(mnemonic, isLtoIndexing);
    this.outputLibrary = outputLibrary;
    this.linkOutput = linkOutput;
    this.interfaceOutputLibrary = interfaceOutputLibrary;
    this.fake = fake;
    this.fakeLinkerInputArtifacts = CollectionUtils.makeImmutable(fakeLinkerInputArtifacts);
    this.isLtoIndexing = isLtoIndexing;
    this.linkstampObjects = linkstampObjects;
    this.linkCommandLine = linkCommandLine;
    this.toolchainEnv = toolchainEnv;
    this.executionRequirements = executionRequirements;
    this.ldExecutable = ldExecutable;
    this.hostSystemName = hostSystemName;
    this.targetCpu = targetCpu;
  }

  @VisibleForTesting
  public String getTargetCpu() {
    return targetCpu;
  }

  public String getHostSystemName() {
    return hostSystemName;
  }

  @Override
  @VisibleForTesting
  public NestedSet<Artifact> getPossibleInputsForTesting() {
    return getInputs();
  }

  @Override
  @VisibleForTesting
  public ImmutableMap<String, String> getIncompleteEnvironmentForTesting() {
    return getEnvironment(ImmutableMap.of());
  }

  public ImmutableMap<String, String> getEnvironment(Map<String, String> clientEnv) {
    LinkedHashMap<String, String> result = Maps.newLinkedHashMapWithExpectedSize(env.size());
    env.resolve(result, clientEnv);

    result.putAll(toolchainEnv);

    if (!executionRequirements.containsKey(ExecutionRequirements.REQUIRES_DARWIN)) {
      // This prevents gcc from writing the unpredictable (and often irrelevant)
      // value of getcwd() into the debug info.
      result.put("PWD", "/proc/self/cwd");
    }
    return ImmutableMap.copyOf(result);
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

  /**
   * Returns the output of this action as a {@link LibraryToLink} or null if it is an executable.
   */
  @Nullable
  public LibraryToLink getOutputLibrary() {
    return outputLibrary;
  }

  public LibraryToLink getInterfaceOutputLibrary() {
    return interfaceOutputLibrary;
  }

  /** Returns the path to the output artifact produced by the linker. */
  private Path getOutputFile(ActionExecutionContext actionExecutionContext) {
    return actionExecutionContext.getInputPath(linkOutput);
  }

  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    return executionRequirements;
  }

  @Override
  public Sequence<CommandLineArgsApi> getStarlarkArgs() throws EvalException {
    ImmutableSet<Artifact> directoryInputs =
        getInputs().toList().stream()
            .filter(artifact -> artifact.isDirectory())
            .collect(ImmutableSet.toImmutableSet());

    CommandLine commandLine = linkCommandLine.getCommandLineForStarlark();

    CommandLineAndParamFileInfo commandLineAndParamFileInfo =
        new CommandLineAndParamFileInfo(commandLine, /* paramFileInfo= */ null);

    Args args = Args.forRegisteredAction(commandLineAndParamFileInfo, directoryInputs);

    return StarlarkList.immutableCopyOf(ImmutableList.of(args));
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException {
    return linkCommandLine.arguments();
  }

  /**
   * Returns the command line specification for this link, included any required linkstamp
   * compilation steps. The command line may refer to a .params file.
   *
   * @param expander ArtifactExpander for expanding TreeArtifacts.
   * @return a finalized command line suitable for execution
   */
  public final List<String> getCommandLine(@Nullable ArtifactExpander expander)
      throws CommandLineExpansionException {
    return linkCommandLine.getCommandLine(expander);
  }

  /**
   * Returns a (possibly empty) list of linkstamp object files.
   *
   * <p>This is used to embed various values from the build system into binaries to identify their
   * provenance.
   */
  public ImmutableList<Artifact> getLinkstampObjects() {
    return linkstampObjects;
  }

  @Override
  @ThreadCompatible
  public ActionContinuationOrResult beginExecution(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    if (fake) {
      executeFake(actionExecutionContext);
      return ActionContinuationOrResult.of(ActionResult.EMPTY);
    }
    Spawn spawn = createSpawn(actionExecutionContext);
    SpawnContinuation spawnContinuation =
        actionExecutionContext
            .getContext(SpawnStrategy.class)
            .beginExecution(spawn, actionExecutionContext);
    return new CppLinkActionContinuation(actionExecutionContext, spawnContinuation);
  }

  private Spawn createSpawn(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    try {
      return new SimpleSpawn(
          this,
          ImmutableList.copyOf(getCommandLine(actionExecutionContext.getArtifactExpander())),
          getEnvironment(actionExecutionContext.getClientEnv()),
          getExecutionInfo(),
          getInputs(),
          getOutputs(),
          estimateResourceConsumptionLocal());
    } catch (CommandLineExpansionException e) {
      throw new ActionExecutionException(
          "failed to generate link command for rule '"
              + getOwner().getLabel()
              + ": "
              + e.getMessage(),
          this,
          /* catastrophe= */ false);
    }
  }

  // Don't forget to update FAKE_LINK_GUID if you modify this method.
  @ThreadCompatible
  private void executeFake(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    try {
      // Prefix all fake output files in the command line with $TEST_TMPDIR/.
      final String outputPrefix = "$TEST_TMPDIR/";
      List<String> escapedLinkArgv =
          escapeLinkArgv(
              linkCommandLine.getRawLinkArgv(actionExecutionContext.getArtifactExpander()),
              outputPrefix);
      // Write the commands needed to build the real target to the fake target
      // file.
      StringBuilder s = new StringBuilder();
      Joiner.on('\n')
          .appendTo(
              s,
              "# This is a fake target file, automatically generated.",
              "# Do not edit by hand!",
              "echo $0 is a fake target file and not meant to be executed.",
              "exit 0",
              "EOS",
              "",
              "makefile_dir=.",
              "");

      // Concatenate all the (fake) .o files into the result.
      for (Artifact objectFile : fakeLinkerInputArtifacts) {
        if (CppFileTypes.OBJECT_FILE.matches(objectFile.getFilename())
            || CppFileTypes.PIC_OBJECT_FILE.matches(objectFile.getFilename())) {
          s.append(
              FileSystemUtils.readContentAsLatin1(
                  actionExecutionContext.getInputPath(objectFile))); // (IOException)
        }
      }
      Path outputFile = getOutputFile(actionExecutionContext);

      s.append(outputFile.getBaseName()).append(": ");
      Joiner.on(' ').appendTo(s, escapedLinkArgv);
      s.append('\n');
      if (outputFile.exists()) {
        outputFile.setWritable(true); // (IOException)
      }
      FileSystemUtils.writeContent(outputFile, ISO_8859_1, s.toString());
      outputFile.setExecutable(true); // (IOException)
      for (Artifact output : getOutputs()) {
        // Make ThinLTO link actions (that also have ThinLTO-specific outputs) kind of work; It does
        // not actually work because this makes cc_fake_binary see the indexing action and not the
        // actual linking action, but it's good enough for now.
        FileSystemUtils.touchFile(actionExecutionContext.getInputPath(output));
      }
    } catch (IOException | CommandLineExpansionException e) {
      throw new ActionExecutionException("failed to create fake link command for rule '"
                                         + getOwner().getLabel() + ": " + e.getMessage(),
                                         this, false);
    }
  }

  /**
   * Shell-escapes the raw link command line.
   *
   * @param rawLinkArgv raw link command line
   * @param outputPrefix to be prepended to any outputs
   * @return escaped link command line
   */
  private List<String> escapeLinkArgv(List<String> rawLinkArgv, String outputPrefix) {
    ImmutableList.Builder<String> escapedArgs = ImmutableList.builder();
    for (String rawArg : rawLinkArgv) {
      String escapedArg;
      if (rawArg.equals(getPrimaryOutput().getExecPathString())) {
        escapedArg = outputPrefix + ShellEscaper.escapeString(rawArg);
      } else if (rawArg.startsWith(Link.FAKE_OBJECT_PREFIX)) {
        escapedArg =
            outputPrefix
                + ShellEscaper.escapeString(rawArg.substring(Link.FAKE_OBJECT_PREFIX.length()));
      } else {
        escapedArg = ShellEscaper.escapeString(rawArg);
      }
      escapedArgs.add(escapedArg);
    }
    return escapedArgs.build();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException {
    // The uses of getLinkConfiguration in this method may not be consistent with the computed key.
    // I.e., this may be incrementally incorrect.
    CppLinkInfo.Builder info = CppLinkInfo.newBuilder();
    info.addAllInputFile(
        Artifact.toExecPaths(getLinkCommandLine().getLinkerInputArtifacts().toList()));
    info.setOutputFile(getPrimaryOutput().getExecPathString());
    if (interfaceOutputLibrary != null) {
      info.setInterfaceOutputFile(interfaceOutputLibrary.getArtifact().getExecPathString());
    }
    info.setLinkTargetType(getLinkCommandLine().getLinkTargetType().name());
    info.setLinkStaticness(getLinkCommandLine().getLinkingMode().name());
    info.addAllLinkStamp(Artifact.toExecPaths(getLinkstampObjects()));
    info.addAllBuildInfoHeaderArtifact(Artifact.toExecPaths(getBuildInfoHeaderArtifacts()));
    info.addAllLinkOpt(getLinkCommandLine().getRawLinkArgv(null));

    try {
      return super.getExtraActionInfo(actionKeyContext)
          .setExtension(CppLinkInfo.cppLinkInfo, info.build());
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("CppLinkAction command line expansion cannot fail.");
    }
  }

  /** Returns the (ordered, immutable) list of header files that contain build info. */
  public Iterable<Artifact> getBuildInfoHeaderArtifacts() {
    return linkCommandLine.getBuildInfoHeaderArtifacts();
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp)
      throws CommandLineExpansionException {
    fp.addString(fake ? FAKE_LINK_GUID : LINK_GUID);
    fp.addString(ldExecutable.getPathString());
    fp.addStrings(linkCommandLine.arguments());
    fp.addStringMap(toolchainEnv);
    fp.addStrings(getExecutionInfo().keySet());

    // TODO(bazel-team): For correctness, we need to ensure the invariant that all values accessed
    // during the execution phase are also covered by the key. Above, we add the argv to the key,
    // which covers most cases. Unfortunately, the extra action and fake support methods above also
    // sometimes directly access settings from the link configuration that may or may not affect the
    // key. We either need to change the code to cover them in the key computation, or change the
    // LinkConfiguration to disallow the combinations where the value of a setting does not affect
    // the argv.
    fp.addBoolean(linkCommandLine.isNativeDeps());
    fp.addBoolean(linkCommandLine.useTestOnlyFlags());
    if (linkCommandLine.getToolchainLibrariesSolibDir() != null) {
      fp.addPath(linkCommandLine.getToolchainLibrariesSolibDir());
    }
    fp.addBoolean(isLtoIndexing);
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
    message.append(ShellEscaper.escapeString(linkCommandLine.getLinkerPathString()));
    message.append('\n');
    // Outputting one argument per line makes it easier to diff the results.
    try {
      List<String> arguments = linkCommandLine.arguments();
      for (String argument : ShellEscaper.escapeAll(arguments)) {
        message.append("  Argument: ");
        message.append(argument);
        message.append('\n');
      }
    } catch (CommandLineExpansionException e) {
      message.append("  Could not expand command line: ");
      message.append(e);
      message.append('\n');
    }
    return message.toString();
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  static String getMnemonic(String mnemonic, boolean isLtoIndexing) {
    if (mnemonic == null) {
      return isLtoIndexing ? "CppLTOIndexing" : "CppLink";
    }
    return mnemonic;
  }

  @Override
  protected String getRawProgressMessage() {
    return (isLtoIndexing ? "LTO indexing " : "Linking ") + linkOutput.prettyPrint();
  }

  /**
   * Estimate the resources consumed when this action is run locally.
   */
  public ResourceSet estimateResourceConsumptionLocal() {
    // It's ok if this behaves differently even if the key is identical.
    ResourceSet minLinkResources =
        getLinkCommandLine().getLinkingMode() == LinkingMode.DYNAMIC
            ? MIN_DYNAMIC_LINK_RESOURCES
            : MIN_STATIC_LINK_RESOURCES;

    int inputSize = getLinkCommandLine().getLinkerInputArtifacts().memoizedFlattenAndGetSize();
    return ResourceSet.createWithRamCpu(
        Math.max(
            inputSize * LINK_RESOURCES_PER_INPUT.getMemoryMb(), minLinkResources.getMemoryMb()),
        Math.max(inputSize * LINK_RESOURCES_PER_INPUT.getCpuUsage(), minLinkResources.getCpuUsage())
    );
  }

  private final class CppLinkActionContinuation extends ActionContinuationOrResult {
    private final ActionExecutionContext actionExecutionContext;
    private final SpawnContinuation spawnContinuation;

    public CppLinkActionContinuation(
        ActionExecutionContext actionExecutionContext, SpawnContinuation spawnContinuation) {
      this.actionExecutionContext = actionExecutionContext;
      this.spawnContinuation = spawnContinuation;
    }

    @Override
    public ListenableFuture<?> getFuture() {
      return spawnContinuation.getFuture();
    }

    @Override
    public ActionContinuationOrResult execute()
        throws ActionExecutionException, InterruptedException {
      try {
        SpawnContinuation nextContinuation = spawnContinuation.execute();
        if (!nextContinuation.isDone()) {
          return new CppLinkActionContinuation(actionExecutionContext, nextContinuation);
        }
        return ActionContinuationOrResult.of(ActionResult.create(nextContinuation.get()));
      } catch (ExecException e) {
        throw e.toActionExecutionException(
            "Linking of rule '" + getOwner().getLabel() + "'",
            actionExecutionContext.getVerboseFailures(),
            CppLinkAction.this);
      }
    }
  }

  @Override
  public Sequence<String> getSkylarkArgv() throws EvalException {
    try {
      return StarlarkList.immutableCopyOf(getArguments());
    } catch (CommandLineExpansionException exception) {
      throw new EvalException(Location.BUILTIN, exception);
    }
  }
}
