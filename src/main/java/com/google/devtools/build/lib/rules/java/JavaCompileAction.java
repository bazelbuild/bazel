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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionInfoSpecifier;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.view.proto.Deps;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/** Action that represents a Java compilation. */
@ThreadCompatible
@Immutable
public class JavaCompileAction extends AbstractAction
    implements ExecutionInfoSpecifier, CommandAction {
  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpu(/* memoryMb= */ 750, /* cpuUsage= */ 1);
  private static final UUID GUID = UUID.fromString("e423747c-2827-49e6-b961-f6c08c10bb51");

  private static final ParamFileInfo PARAM_FILE_INFO =
      ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
          .setCharset(ISO_8859_1)
          .setUseAlways(true)
          .build();

  enum CompilationType {
    JAVAC("Javac"),
    // TODO(cushon): rename the mnemonic to 'Turbine' after javac-turbine is turned down (and after
    // collecting data on the perform impact of the turndown)
    TURBINE("JavacTurbine");

    final String mnemonic;

    CompilationType(String mnemonic) {
      this.mnemonic = mnemonic;
    }
  }

  private final CompilationType compilationType;
  private final ImmutableMap<String, String> executionInfo;
  private final CommandLine executableLine;
  private final CommandLine flagLine;
  private final BuildConfiguration configuration;
  private final LazyString progressMessage;

  private final NestedSet<Artifact> directJars;
  private final NestedSet<Artifact> mandatoryInputs;
  private final NestedSet<Artifact> transitiveInputs;
  private final NestedSet<Artifact> dependencyArtifacts;
  private final Artifact outputDepsProto;
  private final JavaClasspathMode classpathMode;

  @Nullable private final ExtraActionInfoSupplier extraActionInfoSupplier;

  public JavaCompileAction(
      CompilationType compilationType,
      ActionOwner owner,
      ActionEnvironment env,
      NestedSet<Artifact> tools,
      RunfilesSupplier runfilesSupplier,
      LazyString progressMessage,
      NestedSet<Artifact> mandatoryInputs,
      NestedSet<Artifact> transitiveInputs,
      NestedSet<Artifact> directJars,
      ImmutableSet<Artifact> outputs,
      ImmutableMap<String, String> executionInfo,
      ExtraActionInfoSupplier extraActionInfoSupplier,
      CommandLine executableLine,
      CommandLine flagLine,
      BuildConfiguration configuration,
      NestedSet<Artifact> dependencyArtifacts,
      Artifact outputDepsProto,
      JavaClasspathMode classpathMode) {
    super(
        owner,
        tools,
        IterablesChain.concat(mandatoryInputs, transitiveInputs),
        runfilesSupplier,
        outputs,
        env);
    this.compilationType = compilationType;
    // TODO(djasper): The only thing that is conveyed through the executionInfo is whether worker
    // mode is enabled or not. Investigate whether we can store just that.
    this.executionInfo =
        configuration.modifiedExecutionInfo(executionInfo, compilationType.mnemonic);
    this.executableLine = executableLine;
    this.flagLine = flagLine;
    this.configuration = configuration;
    this.progressMessage = progressMessage;
    this.extraActionInfoSupplier = extraActionInfoSupplier;
    this.directJars = directJars;
    this.mandatoryInputs = mandatoryInputs;
    this.transitiveInputs = transitiveInputs;
    this.dependencyArtifacts = dependencyArtifacts;
    this.outputDepsProto = outputDepsProto;
    this.classpathMode = classpathMode;
  }

  @Override
  public String getMnemonic() {
    return compilationType.mnemonic;
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp)
      throws CommandLineExpansionException {
    fp.addUUID(GUID);
    fp.addInt(classpathMode.ordinal());
    executableLine.addToFingerprint(actionKeyContext, fp);
    flagLine.addToFingerprint(actionKeyContext, fp);
    // As the classpath is no longer part of commandLines implicitly, we need to explicitly add
    // the transitive inputs to the key here.
    actionKeyContext.addNestedSetToFingerprint(fp, transitiveInputs);
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
    fp.addStringMap(executionInfo);
  }

  /**
   * Compute a reduced classpath that is comprised of the header jars of all the direct dependencies
   * and the jars needed to build those (read from the produced .jdeps file). This duplicates the
   * logic from {@link
   * com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule#computeStrictClasspath}.
   */
  @VisibleForTesting
  ReducedClasspath getReducedClasspath(
      ActionExecutionContext actionExecutionContext, JavaCompileActionContext context)
      throws IOException {
    HashSet<String> direct = new HashSet<>();
    for (Artifact directJar : directJars) {
      direct.add(directJar.getExecPathString());
    }
    for (Artifact depArtifact : dependencyArtifacts) {
      for (Deps.Dependency dep :
          context.getDependencies(depArtifact, actionExecutionContext).getDependencyList()) {
        direct.add(dep.getPath());
      }
    }
    ImmutableList<Artifact> transitiveCollection = transitiveInputs.toList();
    ImmutableList<Artifact> reducedJars =
        ImmutableList.copyOf(
            Iterables.filter(
                transitiveCollection, input -> direct.contains(input.getExecPathString())));
    return new ReducedClasspath(reducedJars, transitiveCollection.size());
  }

  /**
   * Simpliar to {@link
   * com.google.devtools.build.lib.analysis.actions.SpawnAction.ExtraActionInfoSupplier} but
   * additionally includes the spawn arguments, which change between direct and fallback
   * invocations.
   */
  interface ExtraActionInfoSupplier {
    void extend(ExtraActionInfo.Builder builder, ImmutableList<String> arguments);
  }

  static class ReducedClasspath {
    final ImmutableList<Artifact> reducedJars;
    final int fullLength;

    ReducedClasspath(ImmutableList<Artifact> reducedJars, int fullLength) {
      this.reducedJars = reducedJars;
      this.fullLength = fullLength;
    }
  }

  @VisibleForTesting
  JavaSpawn getReducedSpawn(
      ActionExecutionContext actionExecutionContext,
      ReducedClasspath reducedClasspath,
      boolean fallback)
      throws CommandLineExpansionException {
    CustomCommandLine.Builder classpathLine = CustomCommandLine.builder();
    if (fallback) {
      classpathLine.addExecPaths("--classpath", transitiveInputs);
    } else {
      classpathLine.addExecPaths("--classpath", reducedClasspath.reducedJars);
    }
    // These flags instruct JavaBuilder that this is a compilation with a reduced classpath and
    // that it should report a special value back if a compilation error occurs that suggests
    // retrying with the full classpath.
    classpathLine.add("--reduce_classpath_mode", fallback ? "BAZEL_FALLBACK" : "BAZEL_REDUCED");
    classpathLine.add("--full_classpath_length", Integer.toString(reducedClasspath.fullLength));
    classpathLine.add(
        "--reduced_classpath_length", Integer.toString(reducedClasspath.reducedJars.size()));

    CommandLines reducedCommandLine =
        CommandLines.builder()
            .addCommandLine(executableLine)
            .addCommandLine(flagLine, PARAM_FILE_INFO)
            .addCommandLine(classpathLine.build(), PARAM_FILE_INFO)
            .build();
    CommandLines.ExpandedCommandLines expandedCommandLines =
        reducedCommandLine.expand(
            actionExecutionContext.getArtifactExpander(),
            getPrimaryOutput().getExecPath(),
            configuration.getCommandLineLimits());
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext),
        executionInfo,
        Iterables.concat(
            mandatoryInputs, fallback ? transitiveInputs : reducedClasspath.reducedJars));
  }

  private JavaSpawn getFullSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException {
    CommandLines.ExpandedCommandLines expandedCommandLines =
        getCommandLines()
            .expand(
                actionExecutionContext.getArtifactExpander(),
                getPrimaryOutput().getExecPath(),
                configuration.getCommandLineLimits());
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext),
        executionInfo,
        Iterables.concat(mandatoryInputs, transitiveInputs));
  }

  private ImmutableMap<String, String> getEffectiveEnvironment(
      ActionExecutionContext actionExecutionContext) {
    LinkedHashMap<String, String> effectiveEnvironment =
        Maps.newLinkedHashMapWithExpectedSize(env.size());
    env.resolve(effectiveEnvironment, actionExecutionContext.getClientEnv());
    return ImmutableMap.copyOf(effectiveEnvironment);
  }

  @Override
  public ActionContinuationOrResult beginExecution(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    ReducedClasspath reducedClasspath;
    Spawn spawn;
    try {
      if (classpathMode == JavaClasspathMode.BAZEL) {
        JavaCompileActionContext context =
            actionExecutionContext.getContext(JavaCompileActionContext.class);
        try {
          reducedClasspath = getReducedClasspath(actionExecutionContext, context);
        } catch (IOException e) {
          throw new ActionExecutionException(e, this, /*catastrophe=*/ false);
        }
        spawn = getReducedSpawn(actionExecutionContext, reducedClasspath, /* fallback= */ false);
      } else {
        reducedClasspath = null;
        spawn = getFullSpawn(actionExecutionContext);
      }
    } catch (CommandLineExpansionException e) {
      throw new ActionExecutionException(e, this, /*catastrophe=*/ false);
    }
    SpawnContinuation spawnContinuation =
        actionExecutionContext
            .getContext(SpawnActionContext.class)
            .beginExecution(spawn, actionExecutionContext);
    return new JavaActionContinuation(actionExecutionContext, reducedClasspath, spawnContinuation);
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage.toString();
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ProgressMessage extends LazyString {

    private final String prefix;
    private final Artifact output;
    private final ImmutableSet<Artifact> sourceFiles;
    private final ImmutableList<Artifact> sourceJars;
    private final JavaPluginInfo plugins;

    ProgressMessage(
        String prefix,
        Artifact output,
        ImmutableSet<Artifact> sourceFiles,
        ImmutableList<Artifact> sourceJars,
        JavaPluginInfo plugins) {
      this.prefix = prefix;
      this.output = output;
      this.sourceFiles = sourceFiles;
      this.sourceJars = sourceJars;
      this.plugins = plugins;
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder(prefix);
      sb.append(' ');
      sb.append(output.prettyPrint());
      sb.append(" (");
      boolean first = true;
      first = appendCount(sb, first, sourceFiles.size(), "source file");
      appendCount(sb, first, sourceJars.size(), "source jar");
      sb.append(")");
      appendProcessorNames(sb, plugins.processorClasses());
      return sb.toString();
    }

    private static void appendProcessorNames(StringBuilder sb, NestedSet<String> processorClasses) {
      if (processorClasses.isEmpty()) {
        return;
      }
      List<String> shortNames = new ArrayList<>();
      for (String name : processorClasses) {
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
      throws CommandLineExpansionException {
    ExtraActionInfo.Builder builder = super.getExtraActionInfo(actionKeyContext);
    CommandLines commandLinesWithoutExecutable =
        CommandLines.builder()
            .addCommandLine(flagLine)
            .addCommandLine(getFullClasspathLine())
            .build();
    if (extraActionInfoSupplier != null) {
      extraActionInfoSupplier.extend(builder, commandLinesWithoutExecutable.allArguments());
    }
    return builder;
  }

  private final class JavaSpawn extends BaseSpawn {
    final Iterable<ActionInput> inputs;

    public JavaSpawn(
        CommandLines.ExpandedCommandLines expandedCommandLines,
        Map<String, String> environment,
        Map<String, String> executionInfo,
        Iterable<Artifact> inputs) {
      super(
          ImmutableList.copyOf(expandedCommandLines.arguments()),
          environment,
          executionInfo,
          EmptyRunfilesSupplier.INSTANCE,
          JavaCompileAction.this,
          LOCAL_RESOURCES);
      this.inputs = Iterables.concat(inputs, expandedCommandLines.getParamFiles());
    }

    @Override
    public Iterable<? extends ActionInput> getInputFiles() {
      return inputs;
    }
  }

  @VisibleForTesting
  public CommandLines getCommandLines() {
    return CommandLines.builder()
        .addCommandLine(executableLine)
        .addCommandLine(flagLine, PARAM_FILE_INFO)
        .addCommandLine(getFullClasspathLine(), PARAM_FILE_INFO)
        .build();
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
  public Sequence<String> getSkylarkArgv() throws EvalException {
    try {
      return StarlarkList.immutableCopyOf(getArguments());
    } catch (CommandLineExpansionException exception) {
      throw new EvalException(Location.BUILTIN, exception);
    }
  }

  /** Returns the out-of-band execution data for this action. */
  @Override
  public Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException {
    return ImmutableList.copyOf(getCommandLines().allArguments());
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

  @Override
  public Iterable<Artifact> getPossibleInputsForTesting() {
    return null;
  }

  public Artifact getOutputDepsProto() {
    return outputDepsProto;
  }

  private ActionExecutionException toActionExecutionException(
      ExecException e, boolean verboseFailures) {
    String failMessage = getRawProgressMessage();
    return e.toActionExecutionException(failMessage, verboseFailures, this);
  }

  /** Reads the {@code .jdeps} output from the given spawn results. */
  private Deps.Dependencies readOutputDepsProto(
      List<SpawnResult> results, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    SpawnResult spawnResult = Iterables.getOnlyElement(results);
    InputStream inMemoryOutput = spawnResult.getInMemoryOutput(outputDepsProto);
    try (InputStream input =
        inMemoryOutput == null
            ? actionExecutionContext.getInputPath(outputDepsProto).getInputStream()
            : inMemoryOutput) {
      return Deps.Dependencies.parseFrom(input);
    } catch (IOException e) {
      throw toActionExecutionException(
          new EnvironmentalExecException(e), actionExecutionContext.getVerboseFailures());
    }
  }

  private final class JavaActionContinuation extends ActionContinuationOrResult {
    private final ActionExecutionContext actionExecutionContext;
    @Nullable private final ReducedClasspath reducedClasspath;
    private final SpawnContinuation spawnContinuation;

    public JavaActionContinuation(
        ActionExecutionContext actionExecutionContext,
        @Nullable ReducedClasspath reducedClasspath,
        SpawnContinuation spawnContinuation) {
      this.actionExecutionContext = actionExecutionContext;
      this.reducedClasspath = reducedClasspath;
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
          return new JavaActionContinuation(
              actionExecutionContext, reducedClasspath, nextContinuation);
        }

        List<SpawnResult> results = nextContinuation.get();
        if (reducedClasspath == null) {
          return ActionContinuationOrResult.of(ActionResult.create(results));
        }

        Deps.Dependencies dependencies = readOutputDepsProto(results, actionExecutionContext);
        if (compilationType == CompilationType.TURBINE) {
          actionExecutionContext
              .getContext(JavaCompileActionContext.class)
              .insertDependencies(outputDepsProto, dependencies);
        }
        if (!dependencies.getRequiresReducedClasspathFallback()) {
          return ActionContinuationOrResult.of(ActionResult.create(results));
        }

        // Fall back to running with the full classpath. This requires first deleting potential
        // artifacts generated by the reduced action and clearing the metadata caches.
        deleteOutputs(actionExecutionContext.getExecRoot());
        actionExecutionContext.getMetadataHandler().resetOutputs(getOutputs());
        Spawn spawn;
        try {
          spawn = getReducedSpawn(actionExecutionContext, reducedClasspath, /* fallback=*/ true);
        } catch (CommandLineExpansionException e) {
          throw new ActionExecutionException(e, JavaCompileAction.this, /*catastrophe=*/ false);
        }
        SpawnContinuation fallbackContinuation =
            actionExecutionContext
                .getContext(SpawnActionContext.class)
                .beginExecution(spawn, actionExecutionContext);
        return new JavaFallbackActionContinuation(
            actionExecutionContext, results, fallbackContinuation);
      } catch (IOException e) {
        throw toActionExecutionException(
            new EnvironmentalExecException(e), actionExecutionContext.getVerboseFailures());
      } catch (ExecException e) {
        throw toActionExecutionException(e, actionExecutionContext.getVerboseFailures());
      }
    }
  }

  private final class JavaFallbackActionContinuation extends ActionContinuationOrResult {
    private final ActionExecutionContext actionExecutionContext;
    private final List<SpawnResult> primaryResults;
    private final SpawnContinuation spawnContinuation;

    public JavaFallbackActionContinuation(
        ActionExecutionContext actionExecutionContext,
        List<SpawnResult> primaryResults,
        SpawnContinuation spawnContinuation) {
      this.actionExecutionContext = actionExecutionContext;
      this.primaryResults = primaryResults;
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
          return new JavaFallbackActionContinuation(
              actionExecutionContext, primaryResults, nextContinuation);
        }
        List<SpawnResult> fallbackResults = nextContinuation.get();
        if (compilationType == CompilationType.TURBINE) {
          actionExecutionContext
              .getContext(JavaCompileActionContext.class)
              .insertDependencies(
                  outputDepsProto, readOutputDepsProto(fallbackResults, actionExecutionContext));
        }
        return ActionContinuationOrResult.of(
            ActionResult.create(
                ImmutableList.copyOf(Iterables.concat(primaryResults, fallbackResults))));
      } catch (ExecException e) {
        throw toActionExecutionException(e, actionExecutionContext.getVerboseFailures());
      }
    }
  }
}
