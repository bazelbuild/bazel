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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.CheckReturnValue;

/**
 * An Action representing an arbitrary subprocess to be forked and exec'd.
 */
public class SpawnAction extends AbstractAction {
  private static class ExtraActionInfoSupplier<T> {
    private final GeneratedExtension<ExtraActionInfo, T> extension;
    private final T value;

    private ExtraActionInfoSupplier(
        GeneratedExtension<ExtraActionInfo, T> extension,
        T value) {
      this.extension = extension;
      this.value = value;
    }

    void extend(ExtraActionInfo.Builder builder) {
      builder.setExtension(extension, value);
    }
  }

  private static final String GUID = "ebd6fce3-093e-45ee-adb6-bf513b602f0d";

  private static final String VERBOSE_FAILURES_KEY = "BAZEL_VERBOSE_FAILURES";
  private static final String SUBCOMMANDS_KEY = "BAZEL_SUBCOMMANDS";

  private final CommandLine argv;

  private final boolean executeUnconditionally;
  private final String progressMessage;
  private final String mnemonic;
  // entries are (directory for remote execution, Artifact)
  private final ImmutableMap<PathFragment, Artifact> inputManifests;

  private final ResourceSet resourceSet;
  private ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;

  private final ExtraActionInfoSupplier<?> extraActionInfoSupplier;

  private final boolean verboseFailuresAndSubcommandsInEnv;

  /**
   * Constructs a SpawnAction using direct initialization arguments.
   * <p>
   * All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action.
   * @param tools the set of files comprising the tool that does the work (e.g. compiler).
   * @param inputs the set of all files potentially read by this action; must
   *        not be subsequently modified.
   * @param outputs the set of all files written by this action; must not be
   *        subsequently modified.
   * @param resourceSet the resources consumed by executing this Action
   * @param environment the map of environment variables.
   * @param argv the command line to execute. This is merely a list of options
   *        to the executable, and is uninterpreted by the build tool for the
   *        purposes of dependency checking; typically it may include the names
   *        of input and output files, but this is not necessary.
   * @param progressMessage the message printed during the progression of the build
   * @param mnemonic the mnemonic that is reported in the master log.
   */
  public SpawnAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      ResourceSet resourceSet,
      CommandLine argv,
      Map<String, String> environment,
      String progressMessage,
      String mnemonic) {
    this(
        owner,
        tools,
        inputs,
        outputs,
        resourceSet,
        argv,
        ImmutableMap.copyOf(environment),
        ImmutableMap.<String, String>of(),
        progressMessage,
        ImmutableMap.<PathFragment, Artifact>of(),
        mnemonic,
        false,
        null,
        false);
  }

  /**
   * Constructs a SpawnAction using direct initialization arguments.
   *
   * <p>All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action.
   * @param tools the set of files comprising the tool that does the work (e.g. compiler). This is
   *        a subset of "inputs" and is only used by the WorkerSpawnStrategy.
   * @param inputs the set of all files potentially read by this action; must not be subsequently
   *        modified.
   * @param outputs the set of all files written by this action; must not be subsequently modified.
   * @param resourceSet the resources consumed by executing this Action
   * @param environment the map of environment variables.
   * @param executionInfo out-of-band information for scheduling the spawn.
   * @param argv the argv array (including argv[0]) of arguments to pass. This is merely a list of
   *        options to the executable, and is uninterpreted by the build tool for the purposes of
   *        dependency checking; typically it may include the names of input and output files, but
   *        this is not necessary.
   * @param progressMessage the message printed during the progression of the build
   * @param inputManifests entries in inputs that are symlink manifest files.
   *        These are passed to remote execution in the environment rather than as inputs.
   * @param mnemonic the mnemonic that is reported in the master log.
   * @param verboseFailuresAndSubcommandsInEnv if the presense of "--verbose_failures" and/or
   *        "--subcommands" in the execution should be propogated to the environment of the
   *        action.
   */
  public SpawnAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      ResourceSet resourceSet,
      CommandLine argv,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      String progressMessage,
      ImmutableMap<PathFragment, Artifact> inputManifests,
      String mnemonic,
      boolean executeUnconditionally,
      ExtraActionInfoSupplier<?> extraActionInfoSupplier,
      boolean verboseFailuresAndSubcommandsInEnv) {
    super(owner, tools, inputs, outputs);
    this.resourceSet = resourceSet;
    this.executionInfo = executionInfo;
    this.environment = environment;
    this.argv = argv;
    this.progressMessage = progressMessage;
    this.inputManifests = inputManifests;
    this.mnemonic = mnemonic;
    this.executeUnconditionally = executeUnconditionally;
    this.extraActionInfoSupplier = extraActionInfoSupplier;
    this.verboseFailuresAndSubcommandsInEnv = verboseFailuresAndSubcommandsInEnv;
  }

  /**
   * Returns the (immutable) list of all arguments, including the command name, argv[0].
   */
  @VisibleForTesting
  public List<String> getArguments() {
    return ImmutableList.copyOf(argv.arguments());
  }

  /**
   * Returns command argument, argv[0].
   */
  @VisibleForTesting
  public String getCommandFilename() {
    return Iterables.getFirst(argv.arguments(), null);
  }

  /**
   * Returns the (immutable) list of arguments, excluding the command name,
   * argv[0].
   */
  @VisibleForTesting
  public List<String> getRemainingArguments() {
    return ImmutableList.copyOf(Iterables.skip(argv.arguments(), 1));
  }

  @VisibleForTesting
  public boolean isShellCommand() {
    return argv.isShellCommand();
  }

  @Override
  public boolean isVolatile() {
    return executeUnconditionally;
  }

  @Override
  public boolean executeUnconditionally() {
    return executeUnconditionally;
  }

  /**
   * Executes the action without handling ExecException errors.
   *
   * <p>Called by {@link #execute}.
   */
  protected void internalExecute(
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException {
    getContext(actionExecutionContext.getExecutor()).exec(getSpawn(), actionExecutionContext);
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();

    // Reconstruct environment to communicate if verbose_failures and/or subcommands is set.
    if (verboseFailuresAndSubcommandsInEnv) {
      ImmutableMap.Builder<String, String> environmentBuilder =
          new ImmutableMap.Builder<String, String>().putAll(environment);

      if (executor.getVerboseFailures()) {
        environmentBuilder.put(VERBOSE_FAILURES_KEY, "true");
      }
      if (executor.reportsSubcommands()) {
        environmentBuilder.put(SUBCOMMANDS_KEY, "true");
      }

      this.environment = environmentBuilder.build();
    }

    try {
      internalExecute(actionExecutionContext);
    } catch (ExecException e) {
      String failMessage = progressMessage;
      if (isShellCommand()) {
        // The possible reasons it could fail are: shell executable not found, shell
        // exited non-zero, or shell died from signal.  The first is impossible
        // and the second two aren't very interesting, so in the interests of
        // keeping the noise-level down, we don't print a reason why, just the
        // command that failed.
        //
        // 0=shell executable, 1=shell command switch, 2=command
        failMessage = "error executing shell command: " + "'"
            + truncate(Iterables.get(argv.arguments(), 2), 200) + "'";
      }
      throw e.toActionExecutionException(failMessage, executor.getVerboseFailures(), this);
    }
  }

  public ImmutableMap<PathFragment, Artifact> getInputManifests() {
    return inputManifests;
  }

  /**
   * Returns s, truncated to no more than maxLen characters, appending an
   * ellipsis if truncation occurred.
   */
  private static String truncate(String s, int maxLen) {
    return s.length() > maxLen
        ? s.substring(0, maxLen - "...".length()) + "..."
        : s;
  }

  /**
   * Returns a Spawn that is representative of the command that this Action
   * will execute. This function must not modify any state.
   */
  public Spawn getSpawn() {
    return new ActionSpawn();
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStrings(argv.arguments());
    f.addString(getMnemonic());
    // We don't need the toolManifests here, because they are a subset of the inputManifests by
    // definition and the output of an action shouldn't change whether something is considered a
    // tool or not.
    f.addInt(inputManifests.size());
    for (Map.Entry<PathFragment, Artifact> input : inputManifests.entrySet()) {
      f.addString(input.getKey().getPathString() + "/");
      f.addPath(input.getValue().getExecPath());
    }
    f.addStringMap(getEnvironment());
    f.addStringMap(getExecutionInfo());
    return f.hexDigestAndReset();
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    message.append(getProgressMessage());
    message.append('\n');
    for (Map.Entry<String, String> entry : getEnvironment().entrySet()) {
      message.append("  Environment variable: ");
      message.append(ShellEscaper.escapeString(entry.getKey()));
      message.append('=');
      message.append(ShellEscaper.escapeString(entry.getValue()));
      message.append('\n');
    }
    for (String argument : ShellEscaper.escapeAll(argv.arguments())) {
      message.append("  Argument: ");
      message.append(argument);
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
      return progressMessage;
    }
    return super.getRawProgressMessage();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    ExtraActionInfo.Builder builder = super.getExtraActionInfo();
    if (extraActionInfoSupplier == null) {
      Spawn spawn = getSpawn();
      SpawnInfo spawnInfo = spawn.getExtraActionInfo();

      return builder
          .setExtension(SpawnInfo.spawnInfo, spawnInfo);
    } else {
      extraActionInfoSupplier.extend(builder);
      return builder;
    }
  }

  /**
   * Returns the environment in which to run this action.
   *
   * <p>Note that if setVerboseFailuresAndSubcommandsInEnv() is called on the builder, and either
   * verbose_failures or subcommands is specified in the execution context, corresponding variables
   * will be added to the environment.  These variables will not be known until execution time,
   * however, and so are not returned by getEnvironment().
   */
  public Map<String, String> getEnvironment() {
    return environment;
  }

  /**
   * Returns the out-of-band execution data for this action.
   */
  public Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public String describeStrategy(Executor executor) {
    return getContext(executor).strategyLocality(getMnemonic(), isRemotable());
  }

  protected SpawnActionContext getContext(Executor executor) {
    return executor.getSpawnActionContext(getMnemonic());
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    SpawnActionContext context = getContext(executor);
    if (context.isRemotable(getMnemonic(), isRemotable())) {
      return ResourceSet.ZERO;
    }
    return resourceSet;
  }

  /**
   * Returns true if this can be run remotely.
   */
  public final boolean isRemotable() {
    // TODO(bazel-team): get rid of this method.
    return !executionInfo.containsKey("local");
  }

  /**
   * The Spawn which this SpawnAction will execute.
   */
  private class ActionSpawn extends BaseSpawn {

    private final List<Artifact> filesets = new ArrayList<>();

    public ActionSpawn() {
      super(ImmutableList.copyOf(argv.arguments()),
          ImmutableMap.<String, String>of(),
          executionInfo,
          inputManifests,
          SpawnAction.this,
          resourceSet);
      for (Artifact input : getInputs()) {
        if (input.isFileset()) {
          filesets.add(input);
        }
      }
    }

    @Override
    public ImmutableMap<String, String> getEnvironment() {
      return ImmutableMap.copyOf(SpawnAction.this.getEnvironment());
    }

    @Override
    public ImmutableList<Artifact> getFilesetManifests() {
      return ImmutableList.copyOf(filesets);
    }

    @Override
    public Iterable<? extends ActionInput> getInputFiles() {
      // Remove Fileset directories in inputs list. Instead, these are
      // included as manifests in getEnvironment().
      List<Artifact> inputs = Lists.newArrayList(getInputs());
      inputs.removeAll(filesets);
      inputs.removeAll(inputManifests.values());
      return inputs;
    }
  }

  /**
   * Builder class to construct {@link SpawnAction} instances.
   */
  public static class Builder {

    private final NestedSetBuilder<Artifact> toolsBuilder = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.stableOrder();
    private final List<Artifact> outputs = new ArrayList<>();
    private final Map<PathFragment, Artifact> toolManifests = new LinkedHashMap<>();
    private final Map<PathFragment, Artifact> inputManifests = new LinkedHashMap<>();
    private ResourceSet resourceSet = AbstractAction.DEFAULT_RESOURCE_SET;
    private ImmutableMap<String, String> environment = ImmutableMap.of();
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private boolean isShellCommand = false;
    private boolean useDefaultShellEnvironment = false;
    private boolean executeUnconditionally;
    private PathFragment executable;
    // executableArgs does not include the executable itself.
    private List<String> executableArgs;
    private final IterablesChain.Builder<String> argumentsBuilder = IterablesChain.builder();
    private CommandLine commandLine;

    private String progressMessage;
    private ParamFileInfo paramFileInfo = null;
    private String mnemonic = "Unknown";
    private ExtraActionInfoSupplier<?> extraActionInfoSupplier = null;

    private boolean verboseFailuresAndSubcommandsInEnv = false;

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
      this.toolManifests.putAll(other.toolManifests);
      this.inputManifests.putAll(other.inputManifests);
      this.resourceSet = other.resourceSet;
      this.environment = other.environment;
      this.executionInfo = other.executionInfo;
      this.isShellCommand = other.isShellCommand;
      this.useDefaultShellEnvironment = other.useDefaultShellEnvironment;
      this.executable = other.executable;
      this.executableArgs = (other.executableArgs != null)
          ? Lists.newArrayList(other.executableArgs)
          : null;
      this.argumentsBuilder.add(other.argumentsBuilder.build());
      this.commandLine = other.commandLine;
      this.progressMessage = other.progressMessage;
      this.paramFileInfo = other.paramFileInfo;
      this.mnemonic = other.mnemonic;
      this.verboseFailuresAndSubcommandsInEnv = other.verboseFailuresAndSubcommandsInEnv;
    }

    /**
     * Builds the SpawnAction using the passed in action configuration and returns it and all
     * dependent actions. The first item of the returned array is always the SpawnAction itself.
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
     * @return the SpawnAction and any actions required by it, with the first item always being the
     *      SpawnAction itself.
     */
    @CheckReturnValue
    public Action[] build(ActionConstructionContext context) {
      return build(context.getActionOwner(), context.getAnalysisEnvironment(),
          context.getConfiguration());
    }

    @VisibleForTesting @CheckReturnValue
    public Action[] build(ActionOwner owner, AnalysisEnvironment analysisEnvironment,
        BuildConfiguration configuration) {
      if (isShellCommand && executable == null) {
        executable = configuration.getShExecutable();
      }
      Preconditions.checkNotNull(executable);
      Preconditions.checkNotNull(executableArgs);

      if (useDefaultShellEnvironment) {
        this.environment = configuration.getDefaultShellEnvironment();
      }

      ImmutableList<String> argv = ImmutableList.<String>builder()
          .add(executable.getPathString())
          .addAll(executableArgs)
          .build();

      Iterable<String> arguments = argumentsBuilder.build();

      Artifact paramsFile = ParamFileHelper.getParamsFileMaybe(argv, arguments, commandLine,
          paramFileInfo, configuration, analysisEnvironment, outputs);

      List<Action> actions = new ArrayList<>();
      CommandLine actualCommandLine;
      if (paramsFile != null) {
        actualCommandLine = ParamFileHelper.createWithParamsFile(argv, arguments, commandLine,
            isShellCommand, owner, actions, paramFileInfo, paramsFile);
        inputsBuilder.add(paramsFile);
      } else {
        actualCommandLine = ParamFileHelper.createWithoutParamsFile(argv, arguments, commandLine,
            isShellCommand);
      }

      NestedSet<Artifact> tools = toolsBuilder.build();

      // Tools are by definition a subset of the inputs, so make sure they're present there, too.
      NestedSet<Artifact> inputsAndTools =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(inputsBuilder.build())
              .addTransitive(tools)
              .build();

      LinkedHashMap<PathFragment, Artifact> inputAndToolManifests =
          new LinkedHashMap<>(inputManifests);
      inputAndToolManifests.putAll(toolManifests);

      actions.add(
          0,
          new SpawnAction(
              owner,
              tools,
              inputsAndTools,
              ImmutableList.copyOf(outputs),
              resourceSet,
              actualCommandLine,
              environment,
              executionInfo,
              progressMessage,
              ImmutableMap.copyOf(inputAndToolManifests),
              mnemonic,
              executeUnconditionally,
              extraActionInfoSupplier,
              verboseFailuresAndSubcommandsInEnv));
      return actions.toArray(new Action[actions.size()]);
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
     * Adds an input to this action.
     */
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

    /**
     * Adds inputs to this action.
     */
    public Builder addInputs(Iterable<Artifact> artifacts) {
      inputsBuilder.addAll(artifacts);
      return this;
    }

    /**
     * Adds transitive inputs to this action.
     */
    public Builder addTransitiveInputs(NestedSet<Artifact> artifacts) {
      inputsBuilder.addTransitive(artifacts);
      return this;
    }

    private Builder addToolManifest(Artifact artifact, PathFragment remote) {
      toolManifests.put(remote, artifact);
      return this;
    }

    public Builder addInputManifest(Artifact artifact, PathFragment remote) {
      inputManifests.put(remote, artifact);
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

    public Builder setResources(ResourceSet resourceSet) {
      this.resourceSet = resourceSet;
      return this;
    }

    /**
     * Sets the map of environment variables.
     */
    public Builder setEnvironment(Map<String, String> environment) {
      this.environment = ImmutableMap.copyOf(environment);
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
     * Sets the environment to the configurations default shell environment,
     * see {@link BuildConfiguration#getDefaultShellEnvironment}.
     */
    public Builder useDefaultShellEnvironment() {
      this.environment = null;
      this.useDefaultShellEnvironment  = true;
      return this;
    }

    /**
     * Sets the environment variable "BAZEL_VERBOSE_FAILURES" to "true" if --verbose_failures is
     * set in the execution context. Sets the environment variable "BAZEL_SUBCOMMANDS" to "true"
     * if --subcommands is set in the execution context.
     *
     */
    public Builder setVerboseFailuresAndSubcommandsInEnv() {
      this.verboseFailuresAndSubcommandsInEnv = true;
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
     * Sets the executable path; the path is interpreted relative to the
     * execution root.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable(Artifact)}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
     */
    public Builder setExecutable(PathFragment executable) {
      this.executable = executable;
      this.executableArgs = Lists.newArrayList();
      this.isShellCommand = false;
      return this;
    }

    /**
     * Sets the executable as an artifact.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable(Artifact)}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
     */
    public Builder setExecutable(Artifact executable) {
      addTool(executable);
      return setExecutable(executable.getExecPath());
    }

    /**
     * Sets the executable as a configured target. Automatically adds the files to run to the tools
     * and inputs and uses the executable of the target as the executable.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable(Artifact)}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
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
     * {@link #setJavaExecutable}, or {@link #setShellCommand(String)}.
     */
    public Builder setExecutable(FilesToRunProvider executableProvider) {
      Preconditions.checkArgument(executableProvider.getExecutable() != null,
          "The target does not have an executable");
      setExecutable(executableProvider.getExecutable().getExecPath());
      return addTool(executableProvider);
    }

    private Builder setJavaExecutable(PathFragment javaExecutable, Artifact deployJar,
        List<String> jvmArgs, String... launchArgs) {
      this.executable = javaExecutable;
      this.executableArgs = Lists.newArrayList();
      executableArgs.add("-Xverify:none");
      executableArgs.addAll(jvmArgs);
      Collections.addAll(executableArgs, launchArgs);
      toolsBuilder.add(deployJar);
      this.isShellCommand = false;
      return this;
    }

    /**
     * Sets the executable to be a java class executed from the given deploy
     * jar. The deploy jar is automatically added to the action inputs.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
     */
    public Builder setJavaExecutable(PathFragment javaExecutable,
        Artifact deployJar, String javaMainClass, List<String> jvmArgs) {
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
     * {@link #setJavaExecutable}, or {@link #setShellCommand(String)}.
     */
    public Builder setJarExecutable(PathFragment javaExecutable,
        Artifact deployJar, List<String> jvmArgs) {
      return setJavaExecutable(javaExecutable, deployJar, jvmArgs, "-jar",
          deployJar.getExecPathString());
    }

    /**
     * Sets the executable to be the shell and adds the given command as the
     * command to be executed.
     *
     * <p>Note that this will not clear the arguments, so any arguments will
     * be passed in addition to the command given here.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable(Artifact)}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
     */
    public Builder setShellCommand(String command) {
      this.executable = null;
      // 0=shell command switch, 1=command
      this.executableArgs = Lists.newArrayList("-c", command);
      this.isShellCommand = true;
      return this;
    }

    /**
     * Sets the executable to be the shell and adds the given interned commands as the
     * commands to be executed.
     */
    public Builder setShellCommand(Iterable<String> command) {
      this.executable = new PathFragment(Iterables.getFirst(command, null));
      // The first item of the commands is the shell executable that should be used.
      this.executableArgs = ImmutableList.copyOf(Iterables.skip(command, 1));
      this.isShellCommand = true;
      return this;
    }

    /**
     * Adds an executable and its runfiles, which is necessary for executing the spawn itself (e.g.
     * a compiler), in contrast to artifacts that are necessary for the spawn to do its work (e.g.
     * source code).
     */
    public Builder addTool(FilesToRunProvider tool) {
      addTools(tool.getFilesToRun());
      if (tool.getRunfilesManifest() != null) {
        addToolManifest(
            tool.getRunfilesManifest(),
            BaseSpawn.runfilesForFragment(tool.getExecutable().getExecPath()));
      }
      return this;
    }

    /**
     * Appends the arguments to the list of executable arguments.
     */
    public Builder addExecutableArguments(String... arguments) {
      Preconditions.checkState(executableArgs != null);
      Collections.addAll(executableArgs, arguments);
      return this;
    }

    /**
     * Add multiple arguments in the order they are returned by the collection
     * to the list of executable arguments.
     */
    public Builder addExecutableArguments(Iterable<String> arguments) {
      Preconditions.checkState(executableArgs != null);
      Iterables.addAll(executableArgs, arguments);
      return this;
    }

    /**
     * Appends the argument to the list of command-line arguments.
     */
    public Builder addArgument(String argument) {
      Preconditions.checkState(commandLine == null);
      argumentsBuilder.addElement(argument);
      return this;
    }

    /**
     * Appends the arguments to the list of command-line arguments.
     */
    public Builder addArguments(String... arguments) {
      Preconditions.checkState(commandLine == null);
      argumentsBuilder.add(ImmutableList.copyOf(arguments));
      return this;
    }

    /**
     * Add multiple arguments in the order they are returned by the collection.
     */
    public Builder addArguments(Iterable<String> arguments) {
      Preconditions.checkState(commandLine == null);
      argumentsBuilder.add(CollectionUtils.makeImmutable(arguments));
      return this;
    }

    /**
     * Appends the argument both to the inputs and to the list of command-line
     * arguments.
     */
    public Builder addInputArgument(Artifact argument) {
      Preconditions.checkState(commandLine == null);
      addInput(argument);
      addArgument(argument.getExecPathString());
      return this;
    }

    /**
     * Appends the arguments both to the inputs and to the list of command-line
     * arguments.
     */
    public Builder addInputArguments(Iterable<Artifact> arguments) {
      for (Artifact argument : arguments) {
        addInputArgument(argument);
      }
      return this;
    }

    /**
     * Appends the argument both to the outputs and to the list of command-line
     * arguments.
     */
    public Builder addOutputArgument(Artifact argument) {
      Preconditions.checkState(commandLine == null);
      outputs.add(argument);
      argumentsBuilder.addElement(argument.getExecPathString());
      return this;
    }

    /**
     * Sets a delegate to compute the command line at a later time. This method
     * cannot be used in conjunction with the {@link #addArgument} or {@link
     * #addArguments} methods.
     *
     * <p>The main intention of this method is to save memory by allowing
     * client-controlled sharing between actions and configured targets.
     * Objects passed to this method MUST be immutable.
     */
    public Builder setCommandLine(CommandLine commandLine) {
      Preconditions.checkState(argumentsBuilder.isEmpty());
      this.commandLine = commandLine;
      return this;
    }

    public Builder setProgressMessage(String progressMessage) {
      this.progressMessage = progressMessage;
      return this;
    }

    public Builder setMnemonic(String mnemonic) {
      Preconditions.checkArgument(
          !mnemonic.isEmpty() && CharMatcher.JAVA_LETTER_OR_DIGIT.matchesAllOf(mnemonic),
          "mnemonic must only contain letters and/or digits, and have non-zero length, was: \"%s\"",
          mnemonic);
      this.mnemonic = mnemonic;
      return this;
    }

    public <T> Builder setExtraActionInfo(
        GeneratedExtension<ExtraActionInfo, T> extension, T value) {
      this.extraActionInfoSupplier = new ExtraActionInfoSupplier<T>(extension, value);
      return this;
    }

    /**
     * Enable use of a parameter file and set the encoding to ISO-8859-1 (latin1).
     *
     * <p>In order to use parameter files, at least one output artifact must be specified.
     */
    public Builder useParameterFile(ParameterFileType parameterFileType) {
      return useParameterFile(parameterFileType, ISO_8859_1, "@");
    }

    /**
     * Enable or disable the use of a parameter file, set the encoding to the given value, and
     * specify the argument prefix to use in passing the parameter file name to the tool.
     *
     * <p>The default argument prefix is "@". In order to use parameter files, at least one output
     * artifact must be specified.
     */
    public Builder useParameterFile(
        ParameterFileType parameterFileType, Charset charset, String flagPrefix) {
      paramFileInfo = new ParamFileInfo(parameterFileType, charset, flagPrefix);
      return this;
    }
  }
}
