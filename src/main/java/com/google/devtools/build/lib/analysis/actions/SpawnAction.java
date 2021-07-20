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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.devtools.build.lib.packages.ExecGroup.DEFAULT_EXEC_GROUP_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.CommandLines.ExpandedCommandLines;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ResourceSetOrBuilder;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.extra.EnvironmentVariable;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.SpawnStrategyResolver;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.starlarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CompileTimeConstant;
import com.google.errorprone.annotations.DoNotCall;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** An Action representing an arbitrary subprocess to be forked and exec'd. */
public class SpawnAction extends AbstractAction implements CommandAction {

  /** Sets extensions on {@link ExtraActionInfo}. */
  public interface ExtraActionInfoSupplier {
    void extend(ExtraActionInfo.Builder builder);
  }

  private static final String GUID = "ebd6fce3-093e-45ee-adb6-bf513b602f0d";

  private final CommandLines commandLines;
  private final CommandLineLimits commandLineLimits;

  private final boolean executeUnconditionally;
  private final boolean isShellCommand;
  private final CharSequence progressMessage;
  private final String mnemonic;

  private final ResourceSetOrBuilder resourceSetOrBuilder;
  private final ImmutableMap<String, String> executionInfo;

  private final ExtraActionInfoSupplier extraActionInfoSupplier;
  private final Artifact primaryOutput;
  private final Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> resultConsumer;

  /**
   * Constructs a SpawnAction using direct initialization arguments.
   *
   * <p>All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action.
   * @param tools the set of files comprising the tool that does the work (e.g. compiler).
   * @param inputs the set of all files potentially read by this action; must not be subsequently
   *     modified.
   * @param outputs the set of all files written by this action; must not be subsequently modified.
   * @param primaryOutput the primary output of this action
   * @param resourceSetOrBuilder the resources consumed by executing this Action.
   * @param env the action environment
   * @param commandLines the command lines to execute. This includes the main argv vector and any
   *     param file-backed command lines.
   * @param commandLineLimits the command line limits, from the build configuration
   * @param isShellCommand Whether the command line represents a shell command with the given shell
   *     executable. This is used to give better error messages.
   * @param progressMessage the message printed during the progression of the build.
   * @param mnemonic the mnemonic that is reported in the master log.
   */
  public SpawnAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      Iterable<Artifact> outputs,
      Artifact primaryOutput,
      ResourceSetOrBuilder resourceSetOrBuilder,
      CommandLines commandLines,
      CommandLineLimits commandLineLimits,
      boolean isShellCommand,
      ActionEnvironment env,
      CharSequence progressMessage,
      String mnemonic) {
    this(
        owner,
        tools,
        inputs,
        outputs,
        primaryOutput,
        resourceSetOrBuilder,
        commandLines,
        commandLineLimits,
        isShellCommand,
        env,
        ImmutableMap.<String, String>of(),
        progressMessage,
        EmptyRunfilesSupplier.INSTANCE,
        mnemonic,
        false,
        null,
        null);
  }

  /**
   * Constructs a SpawnAction using direct initialization arguments.
   *
   * <p>All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action
   * @param tools the set of files comprising the tool that does the work (e.g. compiler). This is a
   *     subset of "inputs" and is only used by the WorkerSpawnStrategy
   * @param inputs the set of all files potentially read by this action; must not be subsequently
   *     modified
   * @param outputs the set of all files written by this action; must not be subsequently modified.
   * @param primaryOutput the primary output of this action
   * @param resourceSetOrBuilder the resources consumed by executing this Action.
   * @param env the action's environment
   * @param executionInfo out-of-band information for scheduling the spawn
   * @param commandLines the command lines to execute. This includes the main argv vector and any
   *     param file-backed command lines.
   * @param commandLineLimits the command line limits, from the build configuration
   * @param isShellCommand Whether the command line represents a shell command with the given shell
   *     executable. This is used to give better error messages.
   * @param progressMessage the message printed during the progression of the build
   * @param runfilesSupplier {@link RunfilesSupplier}s describing the runfiles for the action
   * @param mnemonic the mnemonic that is reported in the master log
   */
  public SpawnAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      Iterable<? extends Artifact> outputs,
      Artifact primaryOutput,
      ResourceSetOrBuilder resourceSetOrBuilder,
      CommandLines commandLines,
      CommandLineLimits commandLineLimits,
      boolean isShellCommand,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier,
      String mnemonic,
      boolean executeUnconditionally,
      ExtraActionInfoSupplier extraActionInfoSupplier,
      Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> resultConsumer) {
    super(owner, tools, inputs, runfilesSupplier, outputs, env);
    this.primaryOutput = primaryOutput;
    this.resourceSetOrBuilder = resourceSetOrBuilder;
    this.executionInfo = executionInfo;
    this.commandLines = commandLines;
    this.commandLineLimits = commandLineLimits;
    this.isShellCommand = isShellCommand;
    this.progressMessage = progressMessage;
    this.mnemonic = mnemonic;
    this.executeUnconditionally = executeUnconditionally;
    this.extraActionInfoSupplier = extraActionInfoSupplier;
    this.resultConsumer = resultConsumer;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return primaryOutput;
  }

  @VisibleForTesting
  public CommandLines getCommandLines() {
    return commandLines;
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException, InterruptedException {
    return ImmutableList.copyOf(commandLines.allArguments());
  }

  @Override
  public Sequence<CommandLineArgsApi> getStarlarkArgs() throws EvalException {
    ImmutableList.Builder<CommandLineArgsApi> result = ImmutableList.builder();
    ImmutableSet<Artifact> directoryInputs =
        getInputs().toList().stream()
            .filter(artifact -> artifact.isDirectory())
            .collect(ImmutableSet.toImmutableSet());

    for (CommandLineAndParamFileInfo commandLine : commandLines.getCommandLines()) {
      result.add(Args.forRegisteredAction(commandLine, directoryInputs));
    }
    return StarlarkList.immutableCopyOf(result.build());
  }

  @Override
  public Sequence<String> getStarlarkArgv() throws EvalException, InterruptedException {
    try {
      return StarlarkList.immutableCopyOf(getArguments());
    } catch (CommandLineExpansionException ex) {
      throw new EvalException(ex);
    }
  }

  @Override
  @VisibleForTesting
  public NestedSet<Artifact> getPossibleInputsForTesting() {
    return getInputs();
  }

  /** Returns command argument, argv[0]. */
  @VisibleForTesting
  public String getCommandFilename() throws CommandLineExpansionException, InterruptedException {
    return Iterables.getFirst(getArguments(), null);
  }

  /** Returns the (immutable) list of arguments, excluding the command name, argv[0]. */
  @VisibleForTesting
  public List<String> getRemainingArguments()
      throws CommandLineExpansionException, InterruptedException {
    return ImmutableList.copyOf(Iterables.skip(getArguments(), 1));
  }

  @VisibleForTesting
  public boolean isShellCommand() {
    return isShellCommand;
  }

  @Override
  public boolean isVolatile() {
    return executeUnconditionally;
  }

  @Override
  public boolean executeUnconditionally() {
    return executeUnconditionally;
  }

  /** Hook for subclasses to perform work before the spawn is executed. */
  protected void beforeExecute(ActionExecutionContext actionExecutionContext)
      throws ExecException {}

  /**
   * Hook for subclasses to perform work after the spawn is executed. This method is only executed
   * if the subprocess execution returns normally, not in case of errors (non-zero exit,
   * setup/network failures, etc.).
   */
  protected void afterExecute(
      ActionExecutionContext actionExecutionContext, List<SpawnResult> spawnResults)
      throws ExecException {}

  @Override
  public final ActionContinuationOrResult beginExecution(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Spawn spawn;
    try {
      beforeExecute(actionExecutionContext);
      spawn = getSpawn(actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException(this);
    } catch (CommandLineExpansionException e) {
      throw createCommandLineException(e);
    }
    SpawnContinuation spawnContinuation =
        actionExecutionContext
            .getContext(SpawnStrategyResolver.class)
            .beginExecution(spawn, actionExecutionContext);
    return new SpawnActionContinuation(actionExecutionContext, spawnContinuation);
  }

  private ActionExecutionException createCommandLineException(CommandLineExpansionException e) {
    DetailedExitCode detailedExitCode =
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(Strings.nullToEmpty(e.getMessage()))
                .setSpawn(
                    FailureDetails.Spawn.newBuilder().setCode(Code.COMMAND_LINE_EXPANSION_FAILURE))
                .build());
    return new ActionExecutionException(e, this, /*catastrophe=*/ false, detailedExitCode);
  }

  /**
   * Returns a Spawn that is representative of the command that this Action will execute. This
   * function must not modify any state.
   *
   * <p>This method is final, as it is merely a shorthand use of the generic way to obtain a spawn,
   * which also depends on the client environment. Subclasses that wish to override the way to get a
   * spawn should override the other getSpawn() methods instead.
   */
  @VisibleForTesting
  public final Spawn getSpawn() throws CommandLineExpansionException, InterruptedException {
    return getSpawn(getInputs());
  }

  final Spawn getSpawn(NestedSet<Artifact> inputs)
      throws CommandLineExpansionException, InterruptedException {
    return new ActionSpawn(
        commandLines.allArguments(),
        /*env=*/ ImmutableMap.of(),
        /*envResolved=*/ false,
        inputs,
        /*additionalInputs=*/ ImmutableList.of(),
        /*filesetMappings=*/ ImmutableMap.of());
  }

  /**
   * Return a spawn that is representative of the command that this Action will execute in the given
   * client environment.
   */
  public Spawn getSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException, InterruptedException {
    return getSpawn(
        actionExecutionContext.getArtifactExpander(),
        actionExecutionContext.getClientEnv(),
        /*envResolved=*/ false,
        actionExecutionContext.getTopLevelFilesets());
  }

  /**
   * Return a spawn that is representative of the command that this Action will execute in the given
   * environment.
   *
   * @param envResolved If set to true, the passed environment variables will be used as the Spawn
   *     effective environment. Otherwise they will be used as client environment to resolve the
   *     action env.
   */
  Spawn getSpawn(
      ArtifactExpander artifactExpander,
      Map<String, String> env,
      boolean envResolved,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings)
      throws CommandLineExpansionException, InterruptedException {
    ExpandedCommandLines expandedCommandLines =
        commandLines.expand(artifactExpander, getPrimaryOutput().getExecPath(), commandLineLimits);
    return new ActionSpawn(
        ImmutableList.copyOf(expandedCommandLines.arguments()),
        env,
        envResolved,
        getInputs(),
        expandedCommandLines.getParamFiles(),
        filesetMappings);
  }

  Spawn getSpawnForExtraAction() throws CommandLineExpansionException, InterruptedException {
    return getSpawn(getInputs());
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addString(GUID);
    commandLines.addToFingerprint(actionKeyContext, artifactExpander, fp);
    fp.addString(getMnemonic());
    // We don't need the toolManifests here, because they are a subset of the inputManifests by
    // definition and the output of an action shouldn't change whether something is considered a
    // tool or not.
    fp.addPaths(getRunfilesSupplier().getRunfilesDirs());
    ImmutableList<Artifact> runfilesManifests = getRunfilesSupplier().getManifests();
    fp.addInt(runfilesManifests.size());
    for (Artifact runfilesManifest : runfilesManifests) {
      fp.addPath(runfilesManifest.getExecPath());
    }
    env.addTo(fp);
    fp.addStringMap(getExecutionInfo());
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    message.append(getProgressMessage());
    message.append('\n');
    for (Map.Entry<String, String> entry : env.getFixedEnv().toMap().entrySet()) {
      message.append("  Environment variable: ");
      message.append(ShellEscaper.escapeString(entry.getKey()));
      message.append('=');
      message.append(ShellEscaper.escapeString(entry.getValue()));
      message.append('\n');
    }
    for (String var : getClientEnvironmentVariables()) {
      message.append("  Environment variables taken from the client environment: ");
      message.append(ShellEscaper.escapeString(var));
      message.append('\n');
    }
    try {
      for (String argument : ShellEscaper.escapeAll(getArguments())) {
        message.append("  Argument: ");
        message.append(argument);
        message.append('\n');
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      message.append("Interrupted while expanding command line\n");
    } catch (CommandLineExpansionException e) {
      message.append("Could not expand command line: ");
      message.append(e);
      message.append('\n');
    }
    return message.toString();
  }

  @Override
  public final String getMnemonic() {
    return mnemonic;
  }

  @Override
  protected String getRawProgressMessage() {
    if (progressMessage != null) {
      if (progressMessage instanceof String) {
        String progressMessageStr = (String) progressMessage;
        if (progressMessageStr.contains("%{label}") && getOwner().getLabel() != null) {
          progressMessageStr =
              progressMessageStr.replace("%{label}", getOwner().getLabel().toString());
        }
        if (progressMessageStr.contains("%{output}") && getPrimaryOutput() != null) {
          progressMessageStr =
              progressMessageStr.replace("%{output}", getPrimaryOutput().getExecPathString());
        }
        if (progressMessageStr.contains("%{input}") && getPrimaryInput() != null) {
          progressMessageStr =
              progressMessageStr.replace("%{input}", getPrimaryInput().getExecPathString());
        }
        return progressMessageStr;
      } else {
        // handles OnDemandString
        return progressMessage.toString();
      }
    }
    return super.getRawProgressMessage();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException, InterruptedException {
    ExtraActionInfo.Builder builder = super.getExtraActionInfo(actionKeyContext);
    if (extraActionInfoSupplier == null) {
      SpawnInfo spawnInfo = getExtraActionSpawnInfo();
      return builder
          .setExtension(SpawnInfo.spawnInfo, spawnInfo);
    } else {
      extraActionInfoSupplier.extend(builder);
      return builder;
    }
  }

  /**
   * Returns information about this spawn action for use by the extra action mechanism.
   *
   * <p>Subclasses of SpawnAction may override this in order to provide action-specific behaviour.
   * This can be necessary, for example, when the action discovers inputs.
   */
  protected SpawnInfo getExtraActionSpawnInfo()
      throws CommandLineExpansionException, InterruptedException {
    SpawnInfo.Builder info = SpawnInfo.newBuilder();
    Spawn spawn = getSpawnForExtraAction();
    info.addAllArgument(spawn.getArguments());
    for (Map.Entry<String, String> variable : spawn.getEnvironment().entrySet()) {
      info.addVariable(
          EnvironmentVariable.newBuilder()
              .setName(variable.getKey())
              .setValue(variable.getValue())
              .build());
    }
    for (ActionInput input : spawn.getInputFiles().toList()) {
      // Explicitly ignore middleman artifacts here.
      if (!(input instanceof Artifact) || !((Artifact) input).isMiddlemanArtifact()) {
        info.addInputFile(input.getExecPathString());
      }
    }
    info.addAllOutputFile(ActionInputHelper.toExecPaths(spawn.getOutputFiles()));
    return info.build();
  }

  @Override
  @VisibleForTesting
  public final ImmutableMap<String, String> getIncompleteEnvironmentForTesting() {
    // TODO(ulfjack): AbstractAction should declare getEnvironment with a return value of type
    // ActionEnvironment to avoid developers misunderstanding the purpose of this method. That
    // requires first updating all subclasses and callers to actually handle environments correctly,
    // so it's not a small change.
    return env.getFixedEnv().toMap();
  }

  /**
   * Returns the out-of-band execution data for this action.
   */
  @Override
  public Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  /** A spawn instance that is tied to a specific SpawnAction. */
  private class ActionSpawn extends BaseSpawn {
    private final NestedSet<ActionInput> inputs;
    private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings;
    private final ImmutableMap<String, String> effectiveEnvironment;

    /**
     * Creates an ActionSpawn with the given environment variables.
     *
     * <p>Subclasses of ActionSpawn may subclass in order to provide action-specific values for
     * environment variables or action inputs.
     */
    private ActionSpawn(
        ImmutableList<String> arguments,
        Map<String, String> env,
        boolean envResolved,
        NestedSet<Artifact> inputs,
        Iterable<? extends ActionInput> additionalInputs,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings)
        throws CommandLineExpansionException {
      super(
          arguments,
          ImmutableMap.<String, String>of(),
          executionInfo,
          SpawnAction.this.getRunfilesSupplier(),
          SpawnAction.this,
          SpawnAction.this.resourceSetOrBuilder.buildResourceSet(inputs));

      NestedSetBuilder<ActionInput> inputsBuilder = NestedSetBuilder.stableOrder();
      ImmutableList<Artifact> manifests = getRunfilesSupplier().getManifests();
      for (Artifact input : inputs.toList()) {
        if (!input.isFileset() && !manifests.contains(input)) {
          inputsBuilder.add(input);
        }
      }
      inputsBuilder.addAll(additionalInputs);
      this.inputs = inputsBuilder.build();
      this.filesetMappings = filesetMappings;

      /**
       * If the action environment is already resolved using the client environment, the given
       * environment variables are used as they are. Otherwise, they are used as clientEnv to
       * resolve the action environment variables
       */
      if (envResolved) {
        effectiveEnvironment = ImmutableMap.copyOf(env);
      } else {
        effectiveEnvironment = SpawnAction.this.getEffectiveEnvironment(env);
      }
    }

    @Override
    public ImmutableMap<String, String> getEnvironment() {
      return effectiveEnvironment;
    }

    @Override
    public ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> getFilesetMappings() {
      return ImmutableMap.copyOf(filesetMappings);
    }

    @Override
    public NestedSet<? extends ActionInput> getInputFiles() {
      return inputs;
    }
  }

  /**
   * Builder class to construct {@link SpawnAction} instances.
   */
  public static class Builder {

    private final NestedSetBuilder<Artifact> toolsBuilder = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    private final List<Artifact> outputs = new ArrayList<>();
    private final List<RunfilesSupplier> inputRunfilesSuppliers = new ArrayList<>();
    private ResourceSetOrBuilder resourceSetOrBuilder = AbstractAction.DEFAULT_RESOURCE_SET;
    private ActionEnvironment actionEnvironment = null;
    private ImmutableMap<String, String> environment = ImmutableMap.of();
    private ImmutableSet<String> inheritedEnvironment = ImmutableSet.of();
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private boolean isShellCommand = false;
    private boolean useDefaultShellEnvironment = false;
    protected boolean executeUnconditionally;
    private CustomCommandLine.Builder executableArgs;
    private List<CommandLineAndParamFileInfo> commandLines = new ArrayList<>();

    private CharSequence progressMessage;
    private String mnemonic = "Unknown";
    protected ExtraActionInfoSupplier extraActionInfoSupplier = null;
    private boolean disableSandboxing = false;
    private String execGroup = DEFAULT_EXEC_GROUP_NAME;

    private Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> resultConsumer = null;

    /**
     * Creates a SpawnAction builder.
     */
    public Builder() {}

    /**
     * Creates a builder that is a copy of another builder.
     */
    public Builder(Builder other) {
      this.toolsBuilder.addTransitive(other.toolsBuilder.build());
      this.inputsBuilder.addTransitive(other.inputsBuilder.build());
      this.outputs.addAll(other.outputs);
      this.inputRunfilesSuppliers.addAll(other.inputRunfilesSuppliers);
      this.resourceSetOrBuilder = other.resourceSetOrBuilder;
      this.actionEnvironment = other.actionEnvironment;
      this.environment = other.environment;
      this.executionInfo = other.executionInfo;
      this.isShellCommand = other.isShellCommand;
      this.useDefaultShellEnvironment = other.useDefaultShellEnvironment;
      this.executableArgs = other.executableArgs;
      this.commandLines = new ArrayList<>(other.commandLines);
      this.progressMessage = other.progressMessage;
      this.mnemonic = other.mnemonic;
    }

    /**
     * Builds the SpawnAction using the passed-in action configuration.
     *
     * <p>This method makes a copy of all the collections, so it is safe to reuse the builder after
     * this method returns.
     *
     * <p>This is annotated with @CheckReturnValue, which causes a compiler error when you call this
     * method and ignore its return value. This is because some time ago, calling .build() had the
     * side-effect of registering it with the RuleContext that was passed in to the constructor.
     * This logic was removed, but if people don't notice and still rely on the side-effect, things
     * may break.
     *
     * @return the SpawnAction.
     */
    @CheckReturnValue
    public SpawnAction build(ActionConstructionContext context) {
      return build(context.getActionOwner(execGroup), context.getConfiguration());
    }

    @VisibleForTesting
    @CheckReturnValue
    public SpawnAction build(ActionOwner owner, BuildConfiguration configuration) {
      CommandLines.Builder result = CommandLines.builder();
      result.addCommandLine(executableArgs.build());
      for (CommandLineAndParamFileInfo pair : this.commandLines) {
        result.addCommandLine(pair);
      }
      CommandLines commandLines = result.build();
      ActionEnvironment env =
          actionEnvironment != null
              ? actionEnvironment
              : useDefaultShellEnvironment
                  ? configuration.getActionEnvironment()
                  : ActionEnvironment.create(environment, inheritedEnvironment);
      return buildSpawnAction(
          owner, commandLines, configuration.getCommandLineLimits(), configuration, env);
    }

    @CheckReturnValue
    SpawnAction buildForActionTemplate(ActionOwner owner) {
      CommandLines.Builder result = CommandLines.builder();
      result.addCommandLine(executableArgs.build());
      for (CommandLineAndParamFileInfo pair : commandLines) {
        result.addCommandLine(pair.commandLine);
      }
      return buildSpawnAction(
          owner,
          result.build(),
          CommandLineLimits.UNLIMITED,
          null,
          ActionEnvironment.create(environment, inheritedEnvironment));
    }

    /**
     * Builds the SpawnAction using the passed-in action configuration.
     *
     * <p>This method makes a copy of all the collections, so it is safe to reuse the builder after
     * this method returns.
     */
    private SpawnAction buildSpawnAction(
        ActionOwner owner,
        CommandLines commandLines,
        CommandLineLimits commandLineLimits,
        @Nullable BuildConfiguration configuration,
        ActionEnvironment env) {
      NestedSet<Artifact> tools = toolsBuilder.build();

      // Tools are by definition a subset of the inputs, so make sure they're present there, too.
      NestedSet<Artifact> inputsAndTools =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(inputsBuilder.build())
              .addTransitive(tools)
              .build();

      if (disableSandboxing) {
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        builder.putAll(executionInfo);
        builder.put("nosandbox", "1");
        executionInfo = builder.build();
      }

      return createSpawnAction(
          owner,
          tools,
          inputsAndTools,
          ImmutableList.copyOf(outputs),
          outputs.get(0),
          resourceSetOrBuilder,
          commandLines,
          commandLineLimits,
          isShellCommand,
          env,
          configuration,
          configuration == null
              ? executionInfo
              : configuration.modifiedExecutionInfo(executionInfo, mnemonic),
          progressMessage,
          CompositeRunfilesSupplier.fromSuppliers(this.inputRunfilesSuppliers),
          mnemonic);
    }

    /** Creates a SpawnAction. */
    protected SpawnAction createSpawnAction(
        ActionOwner owner,
        NestedSet<Artifact> tools,
        NestedSet<Artifact> inputsAndTools,
        ImmutableList<Artifact> outputs,
        Artifact primaryOutput,
        ResourceSetOrBuilder resourceSetOrBuilder,
        CommandLines commandLines,
        CommandLineLimits commandLineLimits,
        boolean isShellCommand,
        ActionEnvironment env,
        @Nullable BuildConfiguration configuration,
        ImmutableMap<String, String> executionInfo,
        CharSequence progressMessage,
        RunfilesSupplier runfilesSupplier,
        String mnemonic) {
      return new SpawnAction(
          owner,
          tools,
          inputsAndTools,
          outputs,
          primaryOutput,
          resourceSetOrBuilder,
          commandLines,
          commandLineLimits,
          isShellCommand,
          env,
          executionInfo,
          progressMessage,
          runfilesSupplier,
          mnemonic,
          executeUnconditionally,
          extraActionInfoSupplier,
          resultConsumer);
    }

    /**
     * Adds an artifact that is necessary for executing the spawn itself (e.g. a compiler), in
     * contrast to an artifact that is necessary for the spawn to do its work (e.g. source code).
     *
     * <p>The artifact is implicitly added to the inputs of the action as well.
     */
    public Builder addTool(Artifact tool) {
      toolsBuilder.add(tool);
      return this;
    }

    /**
     * Adds an executable and its runfiles, which is necessary for executing the spawn itself (e.g.
     * a compiler), in contrast to artifacts that are necessary for the spawn to do its work (e.g.
     * source code).
     */
    public Builder addTool(FilesToRunProvider tool) {
      addTransitiveTools(tool.getFilesToRun());
      addRunfilesSupplier(tool.getRunfilesSupplier());
      return this;
    }

    /** Adds an input to this action. */
    public Builder addInput(Artifact artifact) {
      inputsBuilder.add(artifact);
      return this;
    }

    /**
     * Adds tools to this action.
     */
    public Builder addTools(Iterable<Artifact> artifacts) {
      toolsBuilder.addAll(artifacts);
      return this;
    }

    /** Adds tools to this action. */
    public Builder addTransitiveTools(NestedSet<Artifact> artifacts) {
      toolsBuilder.addTransitive(artifacts);
      return this;
    }

    /**
     * Adds inputs to this action.
     */
    public Builder addInputs(Iterable<Artifact> artifacts) {
      inputsBuilder.addAll(artifacts);
      return this;
    }

    /** @deprecated Use {@link #addTransitiveInputs} to avoid excessive memory use. */
    @Deprecated
    @DoNotCall
    public final Builder addInputs(NestedSet<Artifact> artifacts) {
      // Do not delete this method, or else addInputs(Iterable) calls with a NestedSet argument
      // will not be flagged.
      inputsBuilder.addAll(artifacts.toList());
      return this;
    }

    /**
     * Adds transitive inputs to this action.
     */
    public Builder addTransitiveInputs(NestedSet<Artifact> artifacts) {
      inputsBuilder.addTransitive(artifacts);
      return this;
    }

    public Builder addRunfilesSupplier(RunfilesSupplier supplier) {
      inputRunfilesSuppliers.add(supplier);
      return this;
    }

    public Builder addOutput(Artifact artifact) {
      outputs.add(artifact);
      return this;
    }

    public Builder addOutputs(Iterable<Artifact> artifacts) {
      Iterables.addAll(outputs, artifacts);
      return this;
    }

    /**
     * Checks whether the action produces any outputs
     */
    public boolean hasOutputs() {
      return !outputs.isEmpty();
    }

    /**
     * Sets RecourceSet for builder. If ResourceSetBuilder set, then ResourceSetBuilder will
     * override setResources.
     */
    public Builder setResources(ResourceSetOrBuilder resourceSetOrBuilder) {
      this.resourceSetOrBuilder = resourceSetOrBuilder;
      return this;
    }

    /** Sets the action environment. */
    public Builder setEnvironment(ActionEnvironment actionEnvironment) {
      this.actionEnvironment = actionEnvironment;
      return this;
    }

    /**
     * Sets the map of environment variables. Do not use! This makes the builder ignore the
     * 'default shell environment', which is computed from the --action_env command line option.
     */
    public Builder setEnvironment(Map<String, String> environment) {
      this.environment = ImmutableMap.copyOf(environment);
      this.useDefaultShellEnvironment = false;
      return this;
    }

    /**
     * Sets the set of inherited environment variables. Do not use! This makes the builder ignore
     * the 'default shell environment', which is computed from the --action_env command line option.
     */
    public Builder setInheritedEnvironment(Iterable<String> inheritedEnvironment) {
      this.inheritedEnvironment = ImmutableSet.copyOf(inheritedEnvironment);
      this.useDefaultShellEnvironment = false;
      return this;
    }

    /**
     * Sets the map of execution info.
     */
    public Builder setExecutionInfo(Map<String, String> info) {
      this.executionInfo = ImmutableMap.copyOf(info);
      return this;
    }

    /**
     * Sets the environment to the configuration's default shell environment.
     *
     * <p><b>All actions should set this if possible and avoid using {@link #setEnvironment}.</b>
     *
     * <p>When this property is set, the action will use a minimal, standardized environment map.
     *
     * <p>The list of envvars available to the action (the keys in this map) comes from two places:
     * from the rule class provider and from the command line or rc-files via {@code --action_env}
     * flags.
     *
     * <p>The values for these variables may come from one of three places: from the configuration
     * fragment, or from the {@code --action_env} flag (when the flag specifies a name-value pair,
     * e.g. {@code --action_env=FOO=bar}), or from the client environment (when the flag only
     * specifies a name, e.g. {@code --action_env=HOME}).
     *
     * <p>The client environment is specified by the {@code --client_env} flags. The Bazel client
     * passes these flags to the Bazel server upon each build (e.g. {@code
     * --client_env=HOME=/home/johndoe}), so the server can keep track of environmental changes
     * between builds, and always use the up-to-date environment (as opposed to calling {@code
     * System.getenv}, which it should never do, though as of 2017-08-02 it still does in a few
     * places).
     *
     * <p>The {@code --action_env} has priority over configuration-fragment-dictated envvar values,
     * i.e. if the configuration fragment tries to add FOO=bar to the environment, and there's also
     * {@link --action_env=FOO=baz} or {@link --action_env=FOO}, then FOO will be available to the
     * action and its value will be "baz", or whatever the corresponding {@code --client_env} flag
     * specified, respectively.
     *
     * @see {@link BuildConfiguration#getLocalShellEnvironment}
     */
    public Builder useDefaultShellEnvironment() {
      this.environment = null;
      this.inheritedEnvironment = null;
      this.useDefaultShellEnvironment = true;
      return this;
    }

    /**
     * Makes the action always execute, even if none of its inputs have changed.
     *
     * <p>Only use this when absolutely necessary, since this is a performance hit and we'd like to
     * get rid of this mechanism eventually. You'll eventually be able to declare a Skyframe
     * dependency on the build ID, which would accomplish the same thing.
     */
    public Builder executeUnconditionally() {
      // This should really be implemented by declaring a Skyframe dependency on the build ID
      // instead, however, we can't just do that yet from within actions, so we need to go through
      // Action.executeUnconditionally() which in turn is called by ActionCacheChecker.
      this.executeUnconditionally = true;
      return this;
    }

    /**
     * Sets the executable path; the path is interpreted relative to the execution root, unless it's
     * a bare file name.
     *
     * <p><b>Caution</b>: if the executable is a bare file name ("foo"), it will be interpreted
     * relative to PATH. See https://github.com/bazelbuild/bazel/issues/13189 for details. To avoid
     * that, use {@link #setExecutable(Artifact)} instead.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setExecutable(PathFragment executable) {
      this.executableArgs = CustomCommandLine.builder().addPath(executable);
      this.isShellCommand = false;
      return this;
    }

    /**
     * Sets the executable as an artifact.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setExecutable(Artifact executable) {
      addTool(executable);
      this.executableArgs = CustomCommandLine.builder().addCallablePath(executable.getExecPath());
      this.isShellCommand = false;
      return this;
    }

    /**
     * Sets the executable as a configured target. Automatically adds the files to run to the tools
     * and inputs and uses the executable of the target as the executable.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setExecutable(TransitiveInfoCollection executable) {
      FilesToRunProvider provider = executable.getProvider(FilesToRunProvider.class);
      Preconditions.checkArgument(provider != null);
      return setExecutable(provider);
    }

    /**
     * Sets the executable as a configured target. Automatically adds the files to run to the tools
     * and inputs and uses the executable of the target as the executable.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setExecutable(FilesToRunProvider executableProvider) {
      Preconditions.checkArgument(executableProvider.getExecutable() != null,
          "The target does not have an executable");
      this.executableArgs =
          CustomCommandLine.builder()
              .addCallablePath(executableProvider.getExecutable().getExecPath());
      this.isShellCommand = false;
      return addTool(executableProvider);
    }

    private Builder setJavaExecutable(PathFragment javaExecutable, Artifact deployJar,
        List<String> jvmArgs, String... launchArgs) {
      this.executableArgs =
          CustomCommandLine.builder()
              .addPath(javaExecutable)
              .addAll(ImmutableList.copyOf(jvmArgs))
              .addAll(ImmutableList.copyOf(launchArgs));
      toolsBuilder.add(deployJar);
      this.isShellCommand = false;
      return this;
    }

    /**
     * Sets the executable to be a java class executed from the given deploy jar. The deploy jar is
     * automatically added to the action inputs.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setJavaExecutable(
        PathFragment javaExecutable,
        Artifact deployJar,
        String javaMainClass,
        List<String> jvmArgs) {
      return setJavaExecutable(javaExecutable, deployJar, jvmArgs, "-cp",
          deployJar.getExecPathString(), javaMainClass);
    }

    /**
     * Sets the executable to be a jar executed from the given deploy jar. The deploy jar is
     * automatically added to the action inputs.
     *
     * <p>This method is similar to {@link #setJavaExecutable} but it assumes that the Jar artifact
     * declares a main class.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setJarExecutable(
        PathFragment javaExecutable, Artifact deployJar, List<String> jvmArgs) {
      return setJavaExecutable(javaExecutable, deployJar, jvmArgs, "-jar",
          deployJar.getExecPathString());
    }

    /**
     * Sets the executable to be the shell and adds the given command as the command to be executed.
     *
     * <p>Note that this will not clear the arguments, so any arguments will be passed in addition
     * to the command given here.
     *
     * <p>Calling this method overrides any previous values set via calls to {@link #setExecutable},
     * {@link #setJavaExecutable}, or {@link #setShellCommand}.
     */
    public Builder setShellCommand(PathFragment shExecutable, String command) {
      // 0=shell command switch, 1=command
      this.executableArgs =
          CustomCommandLine.builder().addPath(shExecutable).add("-c").addDynamicString(command);
      this.isShellCommand = true;
      return this;
    }

    /**
     * Sets the executable to be the shell and adds the given interned commands as the
     * commands to be executed.
     */
    public Builder setShellCommand(Iterable<String> command) {
      this.executableArgs = CustomCommandLine.builder().addAll(ImmutableList.copyOf(command));
      this.isShellCommand = true;
      return this;
    }

    /** Returns a {@link CustomCommandLine.Builder} for executable arguments. */
    public CustomCommandLine.Builder executableArguments() {
      if (executableArgs == null) {
        executableArgs = CustomCommandLine.builder();
      }
      return this.executableArgs;
    }

    /** Appends the arguments to the list of executable arguments. */
    public Builder addExecutableArguments(String... arguments) {
      Preconditions.checkState(executableArgs != null);
      this.executableArgs.addAll(ImmutableList.copyOf(arguments));
      return this;
    }

    /**
     * Add multiple arguments in the order they are returned by the collection to the list of
     * executable arguments.
     */
    public Builder addExecutableArguments(Iterable<String> arguments) {
      Preconditions.checkState(executableArgs != null);
      this.executableArgs.addAll(ImmutableList.copyOf(arguments));
      return this;
    }

    /**
     * Adds a delegate to compute the command line at a later time.
     *
     * <p>The arguments are added after the executable arguments. If you add multiple command lines,
     * they are expanded in the corresponding order.
     *
     * <p>The main intention of this method is to save memory by allowing client-controlled sharing
     * between actions and configured targets. Objects passed to this method MUST be immutable.
     *
     * <p>See also {@link CustomCommandLine}.
     */
    public Builder addCommandLine(CommandLine commandLine) {
      this.commandLines.add(new CommandLineAndParamFileInfo(commandLine, null));
      return this;
    }

    /**
     * Adds a delegate to compute the command line at a later time, optionally spilled to a params
     * file.
     *
     * <p>The arguments are added after the executable arguments. If you add multiple command lines,
     * they are expanded in the corresponding order. If the command line is spilled to a params
     * file, it is replaced with an argument pointing to the param file.
     *
     * <p>The main intention of this method is to save memory by allowing client-controlled sharing
     * between actions and configured targets. Objects passed to this method MUST be immutable.
     *
     * <p>See also {@link CustomCommandLine}.
     */
    public Builder addCommandLine(CommandLine commandLine, @Nullable ParamFileInfo paramFileInfo) {
      this.commandLines.add(new CommandLineAndParamFileInfo(commandLine, paramFileInfo));
      return this;
    }

    /**
     * Sets the progress message.
     *
     * <p>The message may contain <code>%{label}</code>, <code>%{input}</code> or <code>%{output}
     * </code> patterns, which are substituted with label string, first input or output's path,
     * respectively.
     */
    public Builder setProgressMessage(@CompileTimeConstant String progressMessage) {
      this.progressMessage = progressMessage;
      return this;
    }

    /**
     * Sets the progress message. The string is lazily evaluated.
     *
     * @param progressMessage The message to display
     * @param subject Passed to {@link String#format}
     * @deprecated Use {@link #setProgressMessage(String)} with provided patterns.
     */
    @FormatMethod
    @Deprecated
    public Builder setProgressMessage(@FormatString String progressMessage, Object subject) {
      return setProgressMessage(
          new OnDemandString() {
            @Override
            public String toString() {
              return String.format(progressMessage, subject);
            }
          });
    }

    /**
     * Sets the progress message. The string is lazily evaluated.
     *
     * @param progressMessage The message to display
     * @param subject0 Passed to {@link String#format}
     * @param subject1 Passed to {@link String#format}
     * @deprecated Use {@link #setProgressMessage(String)} with provided patterns.
     */
    @FormatMethod
    @Deprecated
    public Builder setProgressMessage(
        @FormatString String progressMessage, Object subject0, Object subject1) {
      return setProgressMessage(
          new OnDemandString() {
            @Override
            public String toString() {
              return String.format(progressMessage, subject0, subject1);
            }
          });
    }

    /**
     * Sets the progress message. The string is lazily evaluated.
     *
     * @param progressMessage The message to display
     * @param subject0 Passed to {@link String#format}
     * @param subject1 Passed to {@link String#format}
     * @param subject2 Passed to {@link String#format}
     * @deprecated Use {@link #setProgressMessage(String)} with provided patterns.
     */
    @FormatMethod
    @Deprecated
    public Builder setProgressMessage(
        @FormatString String progressMessage, Object subject0, Object subject1, Object subject2) {
      return setProgressMessage(
          new OnDemandString() {
            @Override
            public String toString() {
              return String.format(progressMessage, subject0, subject1, subject2);
            }
          });
    }

    /**
     * Sets the progress message. The string is lazily evaluated.
     *
     * @param progressMessage The message to display
     * @param subject0 Passed to {@link String#format}
     * @param subject1 Passed to {@link String#format}
     * @param subject2 Passed to {@link String#format}
     * @param subject3 Passed to {@link String#format}
     * @deprecated Use {@link #setProgressMessage(String)} with provided patterns.
     */
    @FormatMethod
    @Deprecated
    public Builder setProgressMessage(
        @FormatString String progressMessage,
        Object subject0,
        Object subject1,
        Object subject2,
        Object subject3) {
      return setProgressMessage(
          new OnDemandString() {
            @Override
            public String toString() {
              return String.format(progressMessage, subject0, subject1, subject2, subject3);
            }
          });
    }

    /**
     * Sets a lazily computed progress message.
     *
     * <p>When possible, prefer use of one of the overloads that use {@link String#format}. If you
     * do use this overload, take care not to capture anything expensive.
     */
    private Builder setProgressMessage(OnDemandString progressMessage) {
      this.progressMessage = progressMessage;
      return this;
    }

    /**
     * Sets the progress message.
     *
     * <p>Same as {@link #setProgressMessage(String)}, except that it may be used with non compile
     * time constants (needed for Starlark literals).
     */
    public Builder setProgressMessageFromStarlark(String progressMessage) {
      this.progressMessage = progressMessage;
      return this;
    }

    /** @throws IllegalArgumentException if the mnemonic is invalid. */
    public Builder setMnemonic(String mnemonic) {
      Preconditions.checkArgument(
          !mnemonic.isEmpty() && CharMatcher.javaLetterOrDigit().matchesAllOf(mnemonic),
          "mnemonic must only contain letters and/or digits, and have non-zero length, was: \"%s\"",
          mnemonic);
      this.mnemonic = mnemonic;
      return this;
    }

    public <T> Builder setExtraActionInfo(ExtraActionInfoSupplier extraActionInfoSupplier) {
      this.extraActionInfoSupplier = extraActionInfoSupplier;
      return this;
    }

    public Builder disableSandboxing() {
      this.disableSandboxing = true;
      return this;
    }

    /**
     * Sets the exec group for this action by name. This does not check that {@code execGroup} is
     * being set to a valid exec group (i.e. one that actually exists). This method expects callers
     * to do that work.
     */
    public Builder setExecGroup(String execGroup) {
      this.execGroup = execGroup;
      return this;
    }

    public Builder addResultConsumer(
        Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> resultConsumer) {
      this.resultConsumer = resultConsumer;
      return this;
    }
  }

  private final class SpawnActionContinuation extends ActionContinuationOrResult {
    private final ActionExecutionContext actionExecutionContext;
    private final SpawnContinuation spawnContinuation;

    public SpawnActionContinuation(
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
        if (nextContinuation.isDone()) {
          if (resultConsumer != null) {
            resultConsumer.accept(Pair.of(actionExecutionContext, nextContinuation.get()));
          }
          List<SpawnResult> spawnResults = nextContinuation.get();
          afterExecute(actionExecutionContext, spawnResults);
          return ActionContinuationOrResult.of(ActionResult.create(nextContinuation.get()));
        }
        return new SpawnActionContinuation(actionExecutionContext, nextContinuation);
      } catch (ExecException e) {
        throw e.toActionExecutionException(SpawnAction.this);
      }
    }
  }
}
