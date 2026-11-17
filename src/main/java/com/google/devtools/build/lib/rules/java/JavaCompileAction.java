// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.actions.ActionAnalysisMetadata.mergeMaps;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnInputs;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.PathMappers;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.exec.SpawnStrategyResolver;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.JavaCompile;
import com.google.devtools.build.lib.server.FailureDetails.JavaCompile.Code;
import com.google.devtools.build.lib.starlarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** Action that represents a Java compilation. */
@ThreadCompatible
@Immutable
public final class JavaCompileAction extends AbstractAction implements CommandAction {
  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpu(/* memoryMb= */ 750, /* cpu= */ 1);
  private static final UUID GUID = UUID.fromString("e423747c-2827-49e6-b961-f6c08c10bb51");

  private static final ParamFileInfo PARAM_FILE_INFO =
      ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED).setUseAlways(true).build();

  private static final String SPAWN_OUTPUT_DEPS_PROTO_SUFFIX = ".unstripped";

  /**
   * Format of the .jdeps path written by the spawn if it {@linkplain
   * #hasSeparateSpawnOutputDepsProto writes a separate file}, applied to the exec path of the
   * declared output.
   */
  static final String SEPARATE_SPAWN_OUTPUT_DEPS_PROTO_FORMAT =
      "%s" + SPAWN_OUTPUT_DEPS_PROTO_SUFFIX;

  private final NestedSet<Artifact> tools;
  private final String mnemonic;
  private final boolean insertDependencies;
  private final ActionEnvironment env;
  private final ImmutableMap<String, String> executionInfo;
  private final CommandLine executableLine;
  private final CommandLine flagLine;
  private final ImmutableList<CommandLine> extraCommandLineArgs;
  private final BuildConfigurationValue configuration;
  private final OnDemandString progressMessage;

  private final NestedSet<Artifact> directJars;
  private final NestedSet<Artifact> mandatoryInputs;
  private final NestedSet<Artifact> transitiveInputs;
  private final NestedSet<Artifact> dependencyArtifacts;
  private final NestedSet<Artifact> additionalArtifactsForPathMapping;
  @Nullable private final Artifact outputDepsProto;
  private final JavaClasspathMode classpathMode;

  public JavaCompileAction(
      String mnemonic,
      boolean insertDependencies,
      ActionOwner owner,
      NestedSet<Artifact> tools,
      OnDemandString progressMessage,
      NestedSet<Artifact> mandatoryInputs,
      NestedSet<Artifact> transitiveInputs,
      NestedSet<Artifact> directJars,
      ImmutableSet<Artifact> outputs,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      CommandLine executableLine,
      CommandLine flagLine,
      ImmutableList<CommandLine> extraCommandLineArgs,
      BuildConfigurationValue configuration,
      NestedSet<Artifact> dependencyArtifacts,
      NestedSet<Artifact> additionalArtifactsForPathMapping,
      @Nullable Artifact outputDepsProto,
      JavaClasspathMode classpathMode) {
    super(owner, allInputs(mandatoryInputs, transitiveInputs, dependencyArtifacts), outputs);
    this.tools = tools;
    this.mnemonic = mnemonic;
    this.insertDependencies = insertDependencies;
    this.env = env;
    this.executionInfo = configuration.modifiedExecutionInfo(executionInfo, mnemonic);
    this.executableLine = executableLine;
    this.flagLine = flagLine;
    this.extraCommandLineArgs = extraCommandLineArgs;
    this.configuration = configuration;
    this.progressMessage = progressMessage;
    this.directJars = directJars;
    this.mandatoryInputs = mandatoryInputs;
    this.transitiveInputs = transitiveInputs;
    this.dependencyArtifacts = dependencyArtifacts;
    this.additionalArtifactsForPathMapping = additionalArtifactsForPathMapping;
    this.outputDepsProto = outputDepsProto;
    this.classpathMode = classpathMode;
    checkState(
        outputDepsProto != null
            || (classpathMode != JavaClasspathMode.BAZEL
                && classpathMode != JavaClasspathMode.BAZEL_NO_FALLBACK),
        "Cannot have null outputDepsProto with reduced class path mode BAZEL %s",
        describe());
  }

  /** Computes all of a {@link JavaCompileAction}'s inputs. */
  private static NestedSet<Artifact> allInputs(
      NestedSet<Artifact> mandatoryInputs,
      NestedSet<Artifact> transitiveInputs,
      NestedSet<Artifact> dependencyArtifacts) {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(mandatoryInputs)
        .addTransitive(transitiveInputs)
        .addTransitive(dependencyArtifacts)
        .build();
  }

  @Override
  public NestedSet<Artifact> getTools() {
    return tools;
  }

  @Override
  public ActionEnvironment getEnvironment() {
    return env;
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public NestedSet<Artifact> getAdditionalArtifactsForPathMapping() {
    return additionalArtifactsForPathMapping;
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addUUID(GUID);
    fp.addInt(classpathMode.ordinal());
    CoreOptions.OutputPathsMode outputPathsMode = PathMappers.getOutputPathsMode(configuration);
    CoreOptions.OutputPathsMode effectiveOutputPathsMode =
        PathMappers.getEffectiveOutputPathsMode(outputPathsMode, getMnemonic(), getExecutionInfo());
    executableLine.addToFingerprint(
        actionKeyContext, inputMetadataProvider, effectiveOutputPathsMode, fp);
    flagLine.addToFingerprint(
        actionKeyContext, inputMetadataProvider, effectiveOutputPathsMode, fp);
    for (CommandLine extraCommandLine : extraCommandLineArgs) {
      extraCommandLine.addToFingerprint(
          actionKeyContext, inputMetadataProvider, effectiveOutputPathsMode, fp);
    }
    // As the classpath is no longer part of commandLines implicitly, we need to explicitly add
    // the transitive inputs to the key here.
    actionKeyContext.addNestedSetToFingerprint(fp, transitiveInputs);
    getEnvironment().addTo(fp);
    fp.addStringMap(executionInfo);
    fp.addBoolean(
        outputDepsProto != null
            && configuration.getFragment(JavaConfiguration.class).inmemoryJdepsFiles());
    PathMappers.addToFingerprint(
        getMnemonic(),
        getExecutionInfo(),
        getAdditionalArtifactsForPathMapping(),
        actionKeyContext,
        outputPathsMode,
        fp);
  }

  /**
   * Compute a reduced classpath that is comprised of the header jars of all the direct dependencies
   * and the jars needed to build those (read from the produced .jdeps file). This duplicates the
   * logic from {@code
   * com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule#computeStrictClasspath}.
   */
  @VisibleForTesting
  NestedSet<Artifact> getReducedClasspath(
      ActionExecutionContext actionExecutionContext, JavaCompileActionContext context)
      throws IOException, InterruptedException {
    HashSet<String> direct = new HashSet<>();
    for (Artifact directJar : directJars.toList()) {
      direct.add(directJar.getExecPathString());
    }
    context.addDependencies(dependencyArtifacts.toList(), actionExecutionContext, direct);
    ImmutableList<Artifact> transitiveCollection = transitiveInputs.toList();
    ImmutableList<Artifact> reducedJars =
        ImmutableList.copyOf(
            Iterables.filter(
                transitiveCollection, input -> direct.contains(input.getExecPathString())));
    return NestedSetBuilder.wrap(Order.STABLE_ORDER, reducedJars);
  }

  private JavaSpawn getReducedSpawn(
      ActionExecutionContext actionExecutionContext,
      NestedSet<Artifact> reducedClasspath,
      boolean fallback)
      throws CommandLineExpansionException, InterruptedException {
    CustomCommandLine.Builder classpathLine = CustomCommandLine.builder();
    PathMapper pathMapper =
        PathMappers.create(
            this,
            PathMappers.getOutputPathsMode(configuration),
            /* isStarlarkAction= */ false,
            actionExecutionContext.getInputMetadataProvider());

    if (fallback) {
      classpathLine.addExecPaths("--classpath", transitiveInputs);
    } else {
      classpathLine.addExecPaths("--classpath", reducedClasspath);
    }

    if (classpathMode == JavaClasspathMode.BAZEL_NO_FALLBACK) {
      // No need of fallback logic, invoke SimpleJavaLibraryBuilder with a reduced --classpath
      classpathLine.add("--reduce_classpath_mode", "NONE");
    } else {
      // These flags instruct JavaBuilder that this is a compilation with a reduced classpath and
      // that it should report a special value back if a compilation error occurs that suggests
      // retrying with the full classpath.
      classpathLine.add("--reduce_classpath_mode", fallback ? "BAZEL_FALLBACK" : "BAZEL_REDUCED");
    }

    CommandLines.Builder commandLinesBuilder =
        CommandLines.builder()
            .addCommandLine(executableLine)
            .addCommandLine(flagLine, PARAM_FILE_INFO)
            .addCommandLine(classpathLine.build(), PARAM_FILE_INFO);
    for (CommandLine extraCommandLine : extraCommandLineArgs) {
      commandLinesBuilder.addCommandLine(extraCommandLine, PARAM_FILE_INFO);
    }
    CommandLines reducedCommandLine = commandLinesBuilder.build();
    CommandLines.ExpandedCommandLines expandedCommandLines =
        reducedCommandLine.expand(
            actionExecutionContext.getInputMetadataProvider(),
            getPrimaryOutput().getExecPath(),
            pathMapper,
            configuration.getCommandLineLimits());
    NestedSet<Artifact> inputs =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(mandatoryInputs)
            .addTransitive(fallback ? transitiveInputs : reducedClasspath)
            .build();
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext.getClientEnv(), pathMapper),
        getExecutionInfo(),
        inputs,
        /* onlyMandatoryOutput= */ fallback ? null : outputDepsProto,
        pathMapper);
  }

  private JavaSpawn getFullSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException, InterruptedException {
    PathMapper pathMapper =
        PathMappers.create(
            this,
            PathMappers.getOutputPathsMode(configuration),
            /* isStarlarkAction= */ false,
            actionExecutionContext.getInputMetadataProvider());
    CommandLines.ExpandedCommandLines expandedCommandLines =
        getCommandLines()
            .expand(
                actionExecutionContext.getInputMetadataProvider(),
                getPrimaryOutput().getExecPath(),
                pathMapper,
                configuration.getCommandLineLimits());
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext.getClientEnv(), pathMapper),
        getExecutionInfo(),
        getInputs(),
        /* onlyMandatoryOutput= */ null,
        pathMapper);
  }

  private ActionExecutionException wrapIOException(IOException e, String message) {
    return ActionExecutionException.fromExecException(
        new EnvironmentalExecException(
            e, createFailureDetail(message, Code.REDUCED_CLASSPATH_FALLBACK_CLEANUP_FAILURE)),
        JavaCompileAction.this);
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    ActionInput spawnOutputDepsProto = getSpawnOutputDepsProto();
    NestedSet<Artifact> reducedClasspath;
    Spawn spawn;
    try {
      if (classpathMode == JavaClasspathMode.BAZEL
          || classpathMode == JavaClasspathMode.BAZEL_NO_FALLBACK) {
        JavaCompileActionContext context =
            actionExecutionContext.getContext(JavaCompileActionContext.class);
        try {
          reducedClasspath = getReducedClasspath(actionExecutionContext, context);
        } catch (IOException e) {
          throw createActionExecutionException(e, Code.REDUCED_CLASSPATH_FAILURE);
        }
        spawn = getReducedSpawn(actionExecutionContext, reducedClasspath, /* fallback= */ false);
      } else {
        reducedClasspath = null;
        spawn = getFullSpawn(actionExecutionContext);
      }
    } catch (CommandLineExpansionException e) {
      throw createActionExecutionException(e, Code.COMMAND_LINE_EXPANSION_FAILURE);
    }

    ImmutableList<SpawnResult> primaryResults;
    try {
      primaryResults =
          actionExecutionContext
              .getContext(SpawnStrategyResolver.class)
              .exec(spawn, actionExecutionContext);
    } catch (ExecException e) {
      throw ActionExecutionException.fromExecException(e, this);
    }

    if (reducedClasspath == null) {
      if (outputDepsProto != null) {
        JavaCompileActionContext javaContext =
            actionExecutionContext.getContext(JavaCompileActionContext.class);
        boolean insert = insertDependencies && javaContext != null;
        if (spawnOutputDepsProto != outputDepsProto) {
          // The spawn wrote a separate .jdeps file, so the declared output has to be created now.
          Deps.Dependencies fullOutputDeps =
              readFullOutputDeps(
                  primaryResults,
                  spawnOutputDepsProto,
                  actionExecutionContext,
                  spawn.getPathMapper());
          if (insert) {
            javaContext.insertDependencies(outputDepsProto, fullOutputDeps);
          }
        } else if (insert) {
          try {
            javaContext.insertDependencies(
                outputDepsProto,
                createFullOutputDeps(
                    primaryResults.getFirst(),
                    outputDepsProto,
                    spawnOutputDepsProto,
                    getInputs(),
                    getAdditionalArtifactsForPathMapping(),
                    actionExecutionContext,
                    spawn.getPathMapper()));
          } catch (IOException e) {
            // Left empty. If we cannot read the .jdeps file now, we will read it later or throw an
            // appropriate error then.
          }
        }
      }
      return ActionResult.create(primaryResults);
    }

    Deps.Dependencies dependencies =
        readFullOutputDeps(
            primaryResults, spawnOutputDepsProto, actionExecutionContext, spawn.getPathMapper());

    if (insertDependencies) {
      actionExecutionContext
          .getContext(JavaCompileActionContext.class)
          .insertDependencies(outputDepsProto, dependencies);
    }
    if (!dependencies.getRequiresReducedClasspathFallback()) {
      return ActionResult.create(primaryResults);
    }

    // Fall back to running with the full classpath. This requires first deleting potential
    // artifacts generated by the reduced action and clearing the metadata caches.
    try {
      deleteOutputs(
          actionExecutionContext.getExecRoot(),
          actionExecutionContext.getPathResolver(),
          /* bulkDeleter= */ null,
          /* cleanupArchivedArtifacts= */ configuration
              .getOptions()
              .get(CoreOptions.class)
              .getArchivedArtifactsMnemonicsFilter()
              .test(getMnemonic()));
    } catch (IOException e) {
      throw wrapIOException(e, "Failed to delete reduced action outputs");
    }

    actionExecutionContext.getOutputMetadataStore().resetOutputs(getOutputs());

    try {
      actionExecutionContext.getFileOutErr().clearOut();
      actionExecutionContext.getFileOutErr().clearErr();
    } catch (IOException e) {
      throw wrapIOException(e, "Failed to clean reduced action stdout/stderr");
    }

    try {
      spawn = getReducedSpawn(actionExecutionContext, reducedClasspath, /* fallback= */ true);
    } catch (CommandLineExpansionException e) {
      Code detailedCode = Code.COMMAND_LINE_EXPANSION_FAILURE;
      throw createActionExecutionException(e, detailedCode);
    }

    ImmutableList<SpawnResult> fallbackResults;
    try {
      fallbackResults =
          actionExecutionContext
              .getContext(SpawnStrategyResolver.class)
              .exec(spawn, actionExecutionContext);
    } catch (ExecException e) {
      throw ActionExecutionException.fromExecException(e, this);
    }

    if (insertDependencies || spawnOutputDepsProto != outputDepsProto) {
      // If the spawn wrote a separate .jdeps file, readFullOutputDeps also writes the declared
      // output.
      Deps.Dependencies fallbackDependencies =
          readFullOutputDeps(
              fallbackResults, spawnOutputDepsProto, actionExecutionContext, spawn.getPathMapper());
      if (insertDependencies) {
        actionExecutionContext
            .getContext(JavaCompileActionContext.class)
            .insertDependencies(outputDepsProto, fallbackDependencies);
      }
    }
    return ActionResult.create(
        ImmutableList.copyOf(Iterables.concat(primaryResults, fallbackResults)));
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage.toString();
  }

  abstract static class ProgressMessage extends OnDemandString {

    private final Artifact output;
    private final int sourceFiles;
    private final int sourceJars;
    private final NestedSet<String> processorClasses;

    ProgressMessage(
        Artifact output,
        ImmutableSet<Artifact> sourceFiles,
        ImmutableList<Artifact> sourceJars,
        JavaPluginData plugins) {
      this.output = output;
      this.sourceFiles = sourceFiles.size();
      this.sourceJars = sourceJars.size();
      this.processorClasses = plugins.processorClasses();
    }

    abstract String prefix();

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder(prefix());
      sb.append(' ');
      sb.append(output.prettyPrint());
      sb.append(" (");
      boolean first = true;
      first = appendCount(sb, first, sourceFiles, "source file");
      appendCount(sb, first, sourceJars, "source jar");
      sb.append(")");
      appendProcessorNames(sb, processorClasses);
      return sb.toString();
    }

    private static void appendProcessorNames(StringBuilder sb, NestedSet<String> processorClasses) {
      if (processorClasses.isEmpty()) {
        return;
      }
      List<String> shortNames = new ArrayList<>();
      for (String name : processorClasses.toList()) {
        // Annotation processor names are qualified class names. Omit the package part for the
        // progress message, e.g. `com.google.Foo` -> `Foo`.
        int idx = name.lastIndexOf('.');
        String shortName = idx != -1 ? name.substring(idx + 1) : name;
        shortNames.add(shortName);
      }
      sb.append(" and running annotation processors (");
      Joiner.on(", ").appendTo(sb, shortNames);
      sb.append(")");
    }

    /**
     * Append an input count to the progress message, e.g. "2 source jars". If an input count has
     * already been appended, prefix with ", ".
     */
    private static boolean appendCount(StringBuilder sb, boolean first, int count, String name) {
      if (count > 0) {
        if (!first) {
          sb.append(", ");
        } else {
          first = false;
        }
        sb.append(count).append(' ').append(name);
        if (count > 1) {
          sb.append('s');
        }
      }
      return first;
    }
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException, InterruptedException {
    ExtraActionInfo.Builder builder = super.getExtraActionInfo(actionKeyContext);
    CommandLines.Builder commandLinesBuilder = CommandLines.builder().addCommandLine(flagLine);
    if (!transitiveInputs.isEmpty() || classpathMode != JavaClasspathMode.OFF) {
      commandLinesBuilder.addCommandLine(getFullClasspathLine());
    }
    for (CommandLine extraCommandLine : extraCommandLineArgs) {
      commandLinesBuilder.addCommandLine(extraCommandLine);
    }
    CommandLines commandLinesWithoutExecutable = commandLinesBuilder.build();
    ImmutableList<String> arguments = commandLinesWithoutExecutable.allArguments();
    if (mnemonic.equals("Javac")) {
      populateExtraActionInfo(builder, arguments);
    }
    return builder;
  }

  private void populateExtraActionInfo(
      ExtraActionInfo.Builder builder, ImmutableList<String> arguments) {
    JavaCompileInfo.Builder info =
        JavaCompileInfo.newBuilder()
            .setOutputjar(getPrimaryOutput().getExecPathString())
            .addAllArgument(arguments);
    PeekingIterator<String> it = Iterators.peekingIterator(arguments.iterator());
    while (it.hasNext()) {
      switch (it.next()) {
        case "--system" -> {
          if (it.hasNext()) {
            info.setSystem(it.next());
          }
        }
        case "--javacopts" -> {
          while (it.hasNext()) {
            String opt = it.next();
            if (opt.equals("--")) {
              break;
            }
            info.addJavacOpt(opt);
          }
        }
        case "--sources" -> collectFlagValues(it, info::addSourceFile);
        case "--classpath" -> collectFlagValues(it, info::addClasspath);
        case "--bootclasspath" -> collectFlagValues(it, info::addBootclasspath);
        case "--source_jars" -> collectFlagValues(it, info::addSourcepath);
        case "--processors" -> collectFlagValues(it, info::addProcessor);
        case "--processorpath" -> collectFlagValues(it, info::addProcessorpath);
        default -> {}
      }
    }
    builder.setExtension(JavaCompileInfo.javaCompileInfo, info.build());
  }

  private static void collectFlagValues(PeekingIterator<String> it, Consumer<String> consumer) {
    while (it.hasNext() && !it.peek().startsWith("--")) {
      consumer.accept(it.next());
    }
  }

  private final class JavaSpawn extends BaseSpawn {
    private final SpawnInputs inputs;
    private final Collection<? extends ActionInput> outputs;
    private final ActionInput onlyMandatoryOutput;
    private final PathMapper pathMapper;

    JavaSpawn(
        CommandLines.ExpandedCommandLines expandedCommandLines,
        Map<String, String> environment,
        ImmutableMap<String, String> executionInfo,
        NestedSet<Artifact> inputs,
        @Nullable Artifact onlyMandatoryOutput,
        PathMapper pathMapper) {
      super(
          expandedCommandLines.arguments(),
          environment,
          executionInfo,
          JavaCompileAction.this,
          LOCAL_RESOURCES);
      ActionInput spawnOutputDepsProto = getSpawnOutputDepsProto();
      this.onlyMandatoryOutput = onlyMandatoryOutput == null ? null : spawnOutputDepsProto;
      this.inputs = SpawnInputs.of(inputs, expandedCommandLines.getParamFiles());
      this.outputs = getSpawnOutputs(getOutputs(), outputDepsProto, spawnOutputDepsProto);
      this.pathMapper = pathMapper;
    }

    @Override
    public SpawnInputs getInputFiles() {
      return inputs;
    }

    @Override
    public Collection<? extends ActionInput> getOutputFiles() {
      return outputs;
    }

    @Override
    public Collection<Artifact> getOutputEdgesForExecutionGraph() {
      // Downstream spawns consume the declared .jdeps output, not the spawn's .jdeps.unstripped.
      return getOutputs();
    }

    @Override
    public boolean isMandatoryOutput(ActionInput output) {
      return onlyMandatoryOutput == null || onlyMandatoryOutput.equals(output);
    }

    @Override
    public PathMapper getPathMapper() {
      return pathMapper;
    }
  }

  @VisibleForTesting
  public CommandLines getCommandLines() {
    CommandLines.Builder builder =
        CommandLines.builder()
            .addCommandLine(executableLine)
            .addCommandLine(flagLine, PARAM_FILE_INFO);
    if (!transitiveInputs.isEmpty() || classpathMode != JavaClasspathMode.OFF) {
      builder.addCommandLine(getFullClasspathLine(), PARAM_FILE_INFO);
    }
    for (CommandLine extraCommandLine : extraCommandLineArgs) {
      builder.addCommandLine(extraCommandLine, PARAM_FILE_INFO);
    }
    return builder.build();
  }

  private CommandLine getFullClasspathLine() {
    CustomCommandLine.Builder classpathLine =
        CustomCommandLine.builder().addExecPaths("--classpath", transitiveInputs);
    if (classpathMode == JavaClasspathMode.JAVABUILDER) {
      classpathLine.add("--reduce_classpath_mode", "JAVABUILDER_REDUCED");
      if (!dependencyArtifacts.isEmpty()) {
        classpathLine.addExecPaths("--deps_artifacts", dependencyArtifacts);
      }
    }
    return classpathLine.build();
  }

  @Override
  public Sequence<String> getStarlarkArgv() throws EvalException, InterruptedException {
    try {
      return StarlarkList.immutableCopyOf(getArguments());
    } catch (CommandLineExpansionException ex) {
      throw new EvalException(ex);
    }
  }

  /** Returns the out-of-band execution data for this action. */
  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    if (outputDepsProto == null
        || !configuration.getFragment(JavaConfiguration.class).inmemoryJdepsFiles()) {
      return mergeMaps(super.getExecutionInfo(), executionInfo);
    }
    return mergeMaps(
        super.getExecutionInfo(),
        executionInfo,
        ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS,
        getSpawnOutputDepsProto().getExecPathString());
  }

  @Override
  public ImmutableList<String> getArguments()
      throws CommandLineExpansionException, InterruptedException {
    return ImmutableList.copyOf(getCommandLines().allArguments());
  }

  @Override
  public Sequence<CommandLineArgsApi> getStarlarkArgs() {
    ImmutableList.Builder<CommandLineArgsApi> result = ImmutableList.builder();
    ImmutableSet<Artifact> directoryInputs =
        getInputs().toList().stream().filter(Artifact::isDirectory).collect(toImmutableSet());
    for (CommandLineAndParamFileInfo commandLine : getCommandLines().unpack()) {
      result.add(Args.forRegisteredAction(commandLine, directoryInputs));
    }
    return StarlarkList.immutableCopyOf(result.build());
  }

  @Override
  @VisibleForTesting
  public ImmutableMap<String, String> getIncompleteEnvironmentForTesting() {
    return getEnvironment().getFixedEnv();
  }

  @Nullable
  @Override
  public NestedSet<Artifact> getPossibleInputsForTesting() {
    return null;
  }

  @Override
  public boolean mayModifySpawnOutputsAfterExecution() {
    // Falling back to the full classpath with --experimental_java_classpath=bazel overwrites the
    // outputs of the reduced classpath spawn.
    return true;
  }

  @Override
  protected ImmutableList<PathFragment> getAdditionalPathOutputsToDelete() {
    var spawnOutputDepsProto = getSpawnOutputDepsProto();
    return spawnOutputDepsProto == outputDepsProto
        ? ImmutableList.of()
        : ImmutableList.of(spawnOutputDepsProto.getExecPath());
  }

  @Nullable
  private ActionInput getSpawnOutputDepsProto() {
    return outputDepsProto == null
        ? null
        : getSpawnOutputDepsProto(
            outputDepsProto,
            hasSeparateSpawnOutputDepsProto(configuration, getMnemonic(), executionInfo));
  }

  /**
   * Returns whether the spawn of a Java compilation action writes its .jdeps output to a separate
   * file, from which the action writes the declared output with original paths restored.
   *
   * @param executionInfo the execution info after applying {@code --modify_execution_info}
   */
  static boolean hasSeparateSpawnOutputDepsProto(
      BuildConfigurationValue configuration, String mnemonic, Map<String, String> executionInfo) {
    // Header jars built with root-relative paths result in root-relative paths in the .jdeps files
    // of all actions that consume them.
    return configuration
            .getFragment(JavaConfiguration.class)
            .experimentalTurbineOutputRootRelativePath()
        || PathMappers.getEffectiveOutputPathsMode(
                PathMappers.getOutputPathsMode(configuration), mnemonic, executionInfo)
            != CoreOptions.OutputPathsMode.OFF;
  }

  /**
   * Returns the .jdeps output of the spawn of a Java compilation action, which is constructed on
   * demand to avoid retaining it on the action.
   */
  static ActionInput getSpawnOutputDepsProto(
      Artifact outputDepsProto, boolean separateSpawnOutputDepsProto) {
    return separateSpawnOutputDepsProto
        ? ActionInputHelper.fromPath(
            outputDepsProto
                .getExecPath()
                .replaceName(outputDepsProto.getFilename() + SPAWN_OUTPUT_DEPS_PROTO_SUFFIX))
        : outputDepsProto;
  }

  static Collection<? extends ActionInput> getSpawnOutputs(
      Collection<Artifact> outputs,
      @Nullable Artifact outputDepsProto,
      @Nullable ActionInput spawnOutputDepsProto) {
    if (spawnOutputDepsProto == outputDepsProto) {
      return outputs;
    }
    ImmutableList.Builder<ActionInput> spawnOutputs = ImmutableList.builder();
    for (Artifact output : outputs) {
      spawnOutputs.add(output.equals(outputDepsProto) ? spawnOutputDepsProto : output);
    }
    return spawnOutputs.build();
  }

  /**
   * Writes the action's .jdeps output from the spawn's .jdeps.unstripped output, restoring config
   * prefixes removed by path mapping.
   *
   * <p>For example: {@code bazel-out/bin/foo/foo.jar -> bazel-out/x86-fastbuild/bin/foo/foo.jar}.
   *
   * <p>The executor may strip config prefixes from actions (i.e. remove {@code /x86-fastbuild/} or
   * equivalent from all input and output paths, command lines, and input file contents). This
   * provides better caching for actions that don't vary by --cpu or --compilation_mode. The full
   * paths must be re-created in Bazel's output tree to keep correct builds. For example, if an
   * otherwise cacheable action's input file *contents* differ across CPUs (like a CPU-dependent
   * generated source file), Bazel needs to maintain distinct paths for each instance. These paths
   * are chosen in Bazel's analysis phase, before it's possible to input contents. So all actions in
   * the output tree must conservatively keep full paths.
   *
   * <p>So this method's ultimate purpose is to translate the executor-optimized version of a .jdeps
   * to the original Bazel-safe version.
   *
   * <p>Root-relative paths recorded by Turbine (see {@code --output_root_relative_path}) are also
   * restored to full exec paths, even if path mapping isn't enabled.
   *
   * <p>If the spawn wrote the action output directly, it doesn't contain paths that have to be
   * restored (see {@link #hasSeparateSpawnOutputDepsProto}) and is only read.
   *
   * <p>If config stripping is on, this method won't work with {@link
   * JavaConfiguration.JavaClasspathMode#JAVABUILDER}. That mode causes downstream Java compilations
   * to read this .jdeps on the executor. Since this method replaces config prefixes, the .jdeps
   * entries won't match the executor's stripped paths. This works best with {@link
   * JavaConfiguration.JavaClasspathMode#BAZEL}, where Bazel directly processes the .jdeps on the
   * local filesystem. Those paths match.
   *
   * @param spawnResult the executor action that created the possibly stripped .jdeps output
   * @param outputDepsProto path to the .jdeps output
   * @param spawnOutputDepsProto dependency output produced by the executor
   * @param actionInputs all inputs to the current action
   * @param additionalArtifactsForPathMapping any additional artifacts that may be referenced in the
   *     .jdeps file by path
   * @param actionExecutionContext the action execution context
   * @return the full deps proto (also written to disk to satisfy the action's declared output if
   *     the spawn wrote a separate file)
   */
  static Deps.Dependencies createFullOutputDeps(
      SpawnResult spawnResult,
      Artifact outputDepsProto,
      ActionInput spawnOutputDepsProto,
      NestedSet<Artifact> actionInputs,
      NestedSet<Artifact> additionalArtifactsForPathMapping,
      ActionExecutionContext actionExecutionContext,
      PathMapper pathMapper)
      throws IOException {

    ByteString executorJdeps =
        readExecutorJdeps(spawnResult, spawnOutputDepsProto, actionExecutionContext);
    Deps.Dependencies executorDeps =
        Deps.Dependencies.parseFrom(executorJdeps, ExtensionRegistry.getEmptyRegistry());
    if (spawnOutputDepsProto == outputDepsProto) {
      return executorDeps;
    }
    Deps.Dependencies fullOutputDeps =
        executorDeps.getDependencyCount() == 0
            ? executorDeps
            : restoreDependencyPaths(
                executorDeps, actionInputs, additionalArtifactsForPathMapping, pathMapper);
    try (var outputStream =
        actionExecutionContext.getInputPath(outputDepsProto).getOutputStream()) {
      (fullOutputDeps != executorDeps ? fullOutputDeps.toByteString() : executorJdeps)
          .writeTo(outputStream);
    }
    return fullOutputDeps;
  }

  private static Deps.Dependencies restoreDependencyPaths(
      Deps.Dependencies executorJdeps,
      NestedSet<Artifact> actionInputs,
      NestedSet<Artifact> additionalArtifactsForPathMapping,
      PathMapper pathMapper) {
    // For each of the action's generated inputs, revert its mapped or root-relative path back to
    // its original path.
    HashMap<String, PathFragment> mappedToOriginalPath = new HashMap<>();
    HashMap<PathFragment, PathFragment> rootRelativeToOriginalPath = new HashMap<>();
    for (Artifact actionInput :
        Iterables.concat(actionInputs.toList(), additionalArtifactsForPathMapping.toList())) {
      if (actionInput.isSourceArtifact()) {
        continue;
      }
      rootRelativeToOriginalPath.put(actionInput.getRootRelativePath(), actionInput.getExecPath());
      if (pathMapper.isNoop()) {
        continue;
      }
      String mappedPath = pathMapper.getMappedExecPathString(actionInput);
      PathFragment previousPath = mappedToOriginalPath.put(mappedPath, actionInput.getExecPath());
      if (previousPath != null && !previousPath.equals(actionInput.getExecPath())) {
        // Multiple inputs from different configs map to the same path. This is allowed when they
        // have identical content (checked by StrippingPathMapper.isPathStrippable). Pick any
        // original path for jdeps rewriting.
        mappedToOriginalPath.put(mappedPath, previousPath);
      }
    }

    // Rewrite the .jdeps proto with full paths.
    boolean modified = false;
    Deps.Dependencies.Builder fullDepsBuilder = Deps.Dependencies.newBuilder(executorJdeps);
    for (Deps.Dependency.Builder dep : fullDepsBuilder.getDependencyBuilderList()) {
      PathFragment pathOnExecutor = PathFragment.create(dep.getPath());
      PathFragment originalPath = rootRelativeToOriginalPath.get(pathOnExecutor);
      if (originalPath == null && !pathMapper.isNoop()) {
        originalPath = mappedToOriginalPath.get(pathOnExecutor.getPathString());
      }
      if (originalPath != null && !originalPath.equals(pathOnExecutor)) {
        dep.setPath(originalPath.getPathString());
        modified = true;
      }
    }
    if (!modified) {
      return executorJdeps;
    }
    return fullDepsBuilder.build();
  }

  private static ByteString readExecutorJdeps(
      SpawnResult spawnResult,
      ActionInput outputDepsProto,
      ActionExecutionContext actionExecutionContext)
      throws IOException {
    ByteString inMemoryOutput = spawnResult.getInMemoryOutput(outputDepsProto);
    if (inMemoryOutput != null) {
      return inMemoryOutput;
    }
    try (var inputStream = actionExecutionContext.getInputPath(outputDepsProto).getInputStream()) {
      return ByteString.readFrom(inputStream);
    }
  }

  /** Reads the full {@code .jdeps} output from the given spawn results. */
  private Deps.Dependencies readFullOutputDeps(
      List<SpawnResult> results,
      ActionInput spawnOutputDepsProto,
      ActionExecutionContext actionExecutionContext,
      PathMapper pathMapper)
      throws ActionExecutionException {
    // SpawnStrategy guarantees that the first list entry exists and corresponds to the executed
    // spawn (subsequent entries may come from tree artifact management or retries).
    SpawnResult result = results.getFirst();
    try {
      return createFullOutputDeps(
          result,
          outputDepsProto,
          spawnOutputDepsProto,
          getInputs(),
          getAdditionalArtifactsForPathMapping(),
          actionExecutionContext,
          pathMapper);
    } catch (IOException e) {
      throw ActionExecutionException.fromExecException(
          new EnvironmentalExecException(
              e, createFailureDetail(".jdeps read IOException", Code.JDEPS_READ_IO_EXCEPTION)),
          this);
    }
  }

  private ActionExecutionException createActionExecutionException(Exception e, Code detailedCode) {
    DetailedExitCode detailedExitCode =
        DetailedExitCode.of(createFailureDetail(Strings.nullToEmpty(e.getMessage()), detailedCode));
    return new ActionExecutionException(e, this, /* catastrophe= */ false, detailedExitCode);
  }

  static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setJavaCompile(JavaCompile.newBuilder().setCode(detailedCode))
        .build();
  }
}
