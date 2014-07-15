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

package com.google.devtools.build.lib.view.actions;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.AbstractAction;
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
import com.google.devtools.build.lib.actions.extra.ActionType;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.AnalysisEnvironment;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * An Action representing an arbitrary subprocess to be forked and exec'd.
 */
public class SpawnAction extends ConfigurationAction {
  private static class ExtraActionInfoSupplier<T> {
    private final ActionType type;
    private final GeneratedExtension<ExtraActionInfo, T> extension;
    private final T value;

    private ExtraActionInfoSupplier(
        ActionType type,
        GeneratedExtension<ExtraActionInfo, T> extension,
        T value) {
      this.type = type;
      this.extension = extension;
      this.value = value;
    }

    void extend(ExtraActionInfo.Builder builder) {
      builder.setType(type);
      builder.setExtension(extension, value);
    }
  }

  private static final String GUID = "ebd6fce3-093e-45ee-adb6-bf513b602f0d";

  private final CommandLine argv;

  private final String progressMessage;
  private final String mnemonic;
  // entries are (directory for remote execution, Artifact)
  private final Map<PathFragment, Artifact> inputManifests;

  private final ResourceSet resourceSet;
  private final ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;

  private final ExtraActionInfoSupplier<?> extraActionInfoSupplier;
  /**
   * Constructs a SpawnAction using direct initialization arguments.
   * <p>
   * All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action.
   * @param inputs the set of all files potentially read by this action; must
   *        not be subsequently modified.
   * @param outputs the set of all files written by this action; must not be
   *        subsequently modified.
   * @param configuration the BuildConfiguration used to setup this action
   * @param resourceSet the resources consumed by executing this Action
   * @param environment the map of environment variables.
   * @param argv the command line to execute. This is merely a list of options
   *        to the executable, and is uninterpreted by the build tool for the
   *        purposes of dependency checking; typically it may include the names
   *        of input and output files, but this is not necessary.
   * @param progressMessage the message printed during the progression of the build
   * @param mnemonic the mnemonic that is reported in the master log.
   */
  public SpawnAction(ActionOwner owner,
      Iterable<Artifact> inputs, Iterable<Artifact> outputs,
      BuildConfiguration configuration,
      ResourceSet resourceSet,
      CommandLine argv,
      Map<String, String> environment,
      String progressMessage,
      String mnemonic) {
    this(owner, inputs, outputs, configuration,
        resourceSet, argv, ImmutableMap.copyOf(environment),
        ImmutableMap.<String, String>of(), progressMessage,
        ImmutableMap.<PathFragment, Artifact>of(), mnemonic, null);
  }

  /**
   * Constructs a SpawnAction using direct initialization arguments.
   *
   * <p>All collections provided must not be subsequently modified.
   *
   * @param owner the owner of the Action.
   * @param inputs the set of all files potentially read by this action; must
   *        not be subsequently modified.
   * @param outputs the set of all files written by this action; must not be
   *        subsequently modified.
   * @param configuration the BuildConfiguration used to setup this action
   * @param resourceSet the resources consumed by executing this Action
   * @param environment the map of environment variables.
   * @param executionInfo out-of-band information for scheduling the spawn.
   * @param argv the argv array (including argv[0]) of arguments to pass. This
   *        is merely a list of options to the executable, and is uninterpreted
   *        by the build tool for the purposes of dependency checking; typically
   *        it may include the names of input and output files, but this is not
   *        necessary.
   * @param progressMessage the message printed during the progression of the build
   * @param inputManifests entries in inputs that are symlink manifest files.
   *        These are passed to remote execution in the environment rather than as inputs.
   * @param mnemonic the mnemonic that is reported in the master log.
   */
  public SpawnAction(ActionOwner owner,
      Iterable<Artifact> inputs, Iterable<Artifact> outputs,
      BuildConfiguration configuration,
      ResourceSet resourceSet,
      CommandLine argv,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      String progressMessage,
      ImmutableMap<PathFragment, Artifact> inputManifests,
      String mnemonic,
      ExtraActionInfoSupplier extraActionInfoSupplier) {
    super(owner, inputs, outputs, configuration);
    this.resourceSet = resourceSet;
    this.executionInfo = executionInfo;
    this.environment = environment;
    this.argv = argv;
    this.progressMessage = progressMessage;
    this.inputManifests = inputManifests;
    this.mnemonic = mnemonic;
    this.extraActionInfoSupplier = extraActionInfoSupplier;
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
      throw newActionExecutionException(failMessage, e, executor.getVerboseFailures());
    }
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
    f.addInt(inputManifests.size());
    for (Map.Entry<PathFragment, Artifact> input : inputManifests.entrySet()) {
      f.addString(input.getKey().getPathString() + "/");
      f.addPath(input.getValue().getExecPath());
    }
    f.addStringMap(getEnvironment());
    f.addStringMap(getExecutionInfo());
    return f.hexDigest();
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
    return progressMessage;
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    Spawn spawn = getSpawn();
    SpawnInfo spawnInfo = spawn.getExtraActionInfo();

    return super.getExtraActionInfo()
        .setType(ActionType.SPAWN)
        .setExtension(SpawnInfo.spawnInfo, spawnInfo);
  }

  /**
   * Returns the environment in which to run this action.
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
      // Also expand middleman artifacts.
    }
  }

  /**
   * Builder class to construct {@link SpawnAction} instances.
   */
  public static class Builder {

    private final ActionOwner owner;
    private final AnalysisEnvironment analysisEnvironment;
    private final BuildConfiguration configuration;
    private final NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.stableOrder();
    private final List<Artifact> outputs = new ArrayList<>();
    private final Map<PathFragment, Artifact> inputManifests = new LinkedHashMap<>();
    private ResourceSet resourceSet = AbstractAction.DEFAULT_RESOURCE_SET;
    private ImmutableMap<String, String> environment = ImmutableMap.of();
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private boolean isShellCommand = false;
    private List<String> executableArgs;
    private final IterablesChain.Builder<String> argumentsBuilder = IterablesChain.builder();
    private CommandLine commandLine;

    private String progressMessage;
    private ParamFileInfo paramFileInfo = null;
    private String mnemonic = "Unknown";
    private ExtraActionInfoSupplier<?> extraActionInfoSupplier = null;
    private boolean registerSpawnAction = true;

    /**
     * Creates a builder that builds {@link SpawnAction} instances.
     */
    @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
    public Builder(ActionOwner owner, AnalysisEnvironment analysisEnvironment,
        BuildConfiguration configuration) {
      this.owner = owner;
      this.analysisEnvironment = analysisEnvironment;
      this.configuration = configuration;
    }

    /**
     * Creates a builder that builds {@link SpawnAction} instances, using the configuration of the
     * rule as the action configuration.
     */
    public Builder(ActionConstructionContext context) {
      this(context.getActionOwner(), context.getAnalysisEnvironment(), context.getConfiguration());
    }

    /**
     * Creates a builder that is a copy of another builder.
     */
    public Builder(Builder other) {
      this(other.owner, other.analysisEnvironment, other.configuration);
      this.inputsBuilder.addTransitive(other.inputsBuilder.build());
      this.outputs.addAll(other.outputs);
      this.inputManifests.putAll(other.inputManifests);
      this.resourceSet = other.resourceSet;
      this.environment = other.environment;
      this.isShellCommand = other.isShellCommand;
      this.executableArgs = (other.executableArgs != null)
          ? Lists.newArrayList(other.executableArgs)
          : null;
      this.argumentsBuilder.add(other.argumentsBuilder.build());
      this.commandLine = other.commandLine;
      this.progressMessage = other.progressMessage;
      this.paramFileInfo = other.paramFileInfo;
      this.mnemonic = other.mnemonic;
      this.registerSpawnAction = other.registerSpawnAction;
    }

    /**
     * Builds the Action as configured and returns it. This method makes a copy
     * of all the collections, so it is safe to reuse the builder after this
     * method returns.
     */
    public SpawnAction build() {
      Preconditions.checkState(executableArgs != null);

      Iterable<String> arguments = argumentsBuilder.build();

      Artifact paramsFile = ParamFileHelper.getParamsFile(executableArgs, arguments,
          commandLine, paramFileInfo, configuration, analysisEnvironment, outputs);

      CommandLine actualCommandLine;
      if (paramsFile != null) {
        actualCommandLine = ParamFileHelper.createWithParamsFile(executableArgs, arguments,
            commandLine, isShellCommand, owner, analysisEnvironment, paramFileInfo, paramsFile);
      } else {
        actualCommandLine = ParamFileHelper.createWithoutParamsFile(
            executableArgs, arguments, commandLine, isShellCommand);
      }

      Iterable<Artifact> actualInputs = collectActualInputs(paramsFile);

      SpawnAction action = new SpawnAction(owner, actualInputs, ImmutableList.copyOf(outputs),
          configuration, resourceSet, actualCommandLine,
          environment, executionInfo,
          progressMessage,
          ImmutableMap.copyOf(inputManifests),
          mnemonic,
          extraActionInfoSupplier);

      if (registerSpawnAction) {
        analysisEnvironment.registerAction(action);
      }
      return action;
    }

    private Iterable<Artifact> collectActualInputs(Artifact parameterFile) {
      if (parameterFile != null) {
        inputsBuilder.add(parameterFile);
      }
      return inputsBuilder.build();
    }

    /**
     * Set whether the builder registers the action just built with the analysis environment.
     */
    public Builder setRegisterSpawnAction(boolean registerSpawnAction) {
      // TODO(bazel-team): Remove this by calling registerAction() at every call site of build()
      this.registerSpawnAction = registerSpawnAction;
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
      return this;
    }

    /**
     * Sets the map of environment variables.
     */
    public Builder setExecutionInfo(Map<String, String> info) {
      this.executionInfo = ImmutableMap.copyOf(info);
      return this;
    }

    /**
     * Sets the map of environment variables from a collection of map entries.
     */
    public Builder setEnvironment(Iterable<Map.Entry<String, String>> environment) {
      // TODO(bazel-team): we use a list of Map.Entry-s because of type safety checks in Skylark.
      // Maybe we could come up with a better way.
      ImmutableMap.Builder<String, String> builder = ImmutableMap.<String, String>builder();
      for (Map.Entry<String, String> entry : environment) {
        builder.put(entry.getKey(), entry.getValue());
      }
      this.environment = builder.build();
      return this;
    }

    /**
     * Sets the environment to the configurations default shell environment,
     * see {@link BuildConfiguration#getDefaultShellEnvironment}.
     */
    public Builder useDefaultShellEnvironment() {
      this.environment = configuration.getDefaultShellEnvironment();
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
      Preconditions.checkArgument(!executable.isAbsolute() ||
          !executable.startsWith(configuration.getExecRoot().asFragment()));
      // TODO(bazel-team): the use of the canonicalizer is probably no longer necessary.
      this.executableArgs = Lists.newArrayList(
          StringCanonicalizer.intern(executable.getPathString()));
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
      return setExecutable(executable.getExecPath());
    }

    /**
     * Sets the executable as a configured target. Automatically adds the files
     * to run to the inputs and uses the executable of the target as the
     * executable.
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
     * Sets the executable as a configured target. Automatically adds the files
     * to run to the inputs and uses the executable of the target as the
     * executable.
     *
     * <p>Calling this method overrides any previous values set via calls to
     * {@link #setExecutable}, {@link #setJavaExecutable}, or
     * {@link #setShellCommand(String)}.
     */
    public Builder setExecutable(FilesToRunProvider executableProvider) {
      Preconditions.checkArgument(executableProvider.getExecutable() != null,
          "The target does not have an executable");
      setExecutable(executableProvider.getExecutable().getExecPath());
      return addTool(executableProvider);
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
      // Using an absolute path starting with the exec root can never work!
      Preconditions.checkArgument(
          !javaExecutable.startsWith(configuration.getExecRoot().asFragment()));
      this.executableArgs = Lists.newArrayList();
      executableArgs.add(StringCanonicalizer.intern(javaExecutable.getPathString()));
      executableArgs.add("-Xverify:none");
      executableArgs.addAll(jvmArgs);
      executableArgs.add("-cp");
      executableArgs.add(deployJar.getExecPathString());
      executableArgs.add(javaMainClass);
      inputsBuilder.add(deployJar);
      this.isShellCommand = false;
      return this;
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
      // 0=shell executable, 1=shell command switch, 2=command
      this.executableArgs = Lists.newArrayList(
          StringCanonicalizer.intern(configuration.getShExecutable().getPathString()),
          "-c", command);
      this.isShellCommand = true;
      return this;
    }

    /**
     * Sets the executable to be the shell and adds the given interned commands as the
     * commands to be executed.
     */
    public Builder setShellCommand(Iterable<String> command) {
      this.executableArgs = ImmutableList.copyOf(command);
      this.isShellCommand = true;
      return this;
    }

    /**
     * Adds an executable and its runfiles, so it can be called from a shell command.
     */
    public Builder addTool(FilesToRunProvider tool) {
      addInputs(tool.getFilesToRun());
      if (tool.getRunfilesManifest() != null) {
        addInputManifest(tool.getRunfilesManifest(),
            BaseSpawn.runfilesForFragment(tool.getExecutable().getExecPath()));
      }
      return this;
    }

    /**
     * Appends the arguments to the list of executable arguments.
     */
    public Builder addExecutableArguments(String... arguments) {
      Preconditions.checkState(executableArgs != null);
      executableArgs.addAll(Arrays.asList(arguments));
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
     * Appends the argument both to the ouputs and to the list of command-line
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
      this.mnemonic = mnemonic;
      return this;
    }

    public <T> Builder setExtraActionInfo(
        ActionType type, GeneratedExtension<ExtraActionInfo, T> extension, T value) {
      this.extraActionInfoSupplier = new ExtraActionInfoSupplier<T>(type, extension, value);
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
