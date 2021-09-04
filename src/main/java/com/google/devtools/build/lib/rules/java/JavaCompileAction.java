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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.concurrent.TimeUnit.SECONDS;
import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
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
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** Action that represents a Java compilation. */
@ThreadCompatible
@Immutable
public class JavaCompileAction extends AbstractAction implements CommandAction {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
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
  private final OnDemandString progressMessage;

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
      OnDemandString progressMessage,
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
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(mandatoryInputs)
            .addTransitive(transitiveInputs)
            .addTransitive(dependencyArtifacts)
            .build(),
        runfilesSupplier,
        outputs,
        env);
    if (outputs.stream().anyMatch(Artifact::isTreeArtifact)) {
      throw new IllegalArgumentException(
          String.format(
              "Unexpected tree artifact output(s): [%s] in JavaCompileAction: %s",
              outputs.stream()
                  .filter(Artifact::isTreeArtifact)
                  .map(Artifact::getExecPathString)
                  .collect(joining(",")),
              this));
    }
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
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addUUID(GUID);
    fp.addInt(classpathMode.ordinal());
    executableLine.addToFingerprint(actionKeyContext, artifactExpander, fp);
    flagLine.addToFingerprint(actionKeyContext, artifactExpander, fp);
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
    for (Artifact directJar : directJars.toList()) {
      direct.add(directJar.getExecPathString());
    }
    for (Artifact depArtifact : dependencyArtifacts.toList()) {
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
    final NestedSet<Artifact> reducedJars;
    final int reducedLength;
    final int fullLength;

    ReducedClasspath(ImmutableList<Artifact> reducedJars, int fullLength) {
      this.reducedJars = NestedSetBuilder.wrap(Order.STABLE_ORDER, reducedJars);
      this.reducedLength = reducedJars.size();
      this.fullLength = fullLength;
    }
  }

  @VisibleForTesting
  JavaSpawn getReducedSpawn(
      ActionExecutionContext actionExecutionContext,
      ReducedClasspath reducedClasspath,
      boolean fallback)
      throws CommandLineExpansionException, InterruptedException {
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
        "--reduced_classpath_length", Integer.toString(reducedClasspath.reducedLength));

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
    NestedSet<Artifact> inputs =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(mandatoryInputs)
            .addTransitive(fallback ? transitiveInputs : reducedClasspath.reducedJars)
            .build();
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext.getClientEnv()),
        executionInfo,
        inputs);
  }

  private JavaSpawn getFullSpawn(ActionExecutionContext actionExecutionContext)
      throws CommandLineExpansionException, InterruptedException {
    CommandLines.ExpandedCommandLines expandedCommandLines =
        getCommandLines()
            .expand(
                actionExecutionContext.getArtifactExpander(),
                getPrimaryOutput().getExecPath(),
                configuration.getCommandLineLimits());
    return new JavaSpawn(
        expandedCommandLines,
        getEffectiveEnvironment(actionExecutionContext.getClientEnv()),
        executionInfo,
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(mandatoryInputs)
            .addTransitive(transitiveInputs)
            // Full spawn mode means classPathMode != JavaClasspathMode.BAZEL, which means
            // JavaBuilder may read .jdeps files to perform classpath reduction on the executor. So
            // make sure these files are staged as inputs to the executor action.
            //
            // Contrast this with getReducedSpawn, which reduces the classpath in the Blaze process
            // *before* sending actions to the executor. In those cases we want to avoid staging
            // .jdeps files, which have config prefixes in output paths, which compromise caching
            // possible by stripping prefixes on the executor.
            .addTransitive(dependencyArtifacts)
            .build());
  }

  @Override
  public ImmutableMap<String, String> getEffectiveEnvironment(Map<String, String> clientEnv)
      throws CommandLineExpansionException {
    LinkedHashMap<String, String> effectiveEnvironment =
        Maps.newLinkedHashMapWithExpectedSize(env.size());
    env.resolve(effectiveEnvironment, clientEnv);
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
    SpawnContinuation spawnContinuation =
        actionExecutionContext
            .getContext(SpawnStrategyResolver.class)
            .beginExecution(spawn, actionExecutionContext);
    return new JavaActionContinuation(actionExecutionContext, reducedClasspath, spawnContinuation);
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage.toString();
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ProgressMessage extends OnDemandString {

    private final String prefix;
    private final Artifact output;
    private final ImmutableSet<Artifact> sourceFiles;
    private final ImmutableList<Artifact> sourceJars;
    private final JavaPluginData plugins;

    ProgressMessage(
        String prefix,
        Artifact output,
        ImmutableSet<Artifact> sourceFiles,
        ImmutableList<Artifact> sourceJars,
        JavaPluginData plugins) {
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
    final NestedSet<ActionInput> inputs;

    public JavaSpawn(
        CommandLines.ExpandedCommandLines expandedCommandLines,
        Map<String, String> environment,
        Map<String, String> executionInfo,
        NestedSet<Artifact> inputs) {
      super(
          ImmutableList.copyOf(expandedCommandLines.arguments()),
          environment,
          executionInfo,
          EmptyRunfilesSupplier.INSTANCE,
          JavaCompileAction.this,
          LOCAL_RESOURCES);
      this.inputs =
          NestedSetBuilder.<ActionInput>fromNestedSet(inputs)
              .addAll(expandedCommandLines.getParamFiles())
              .build();
    }

    @Override
    public NestedSet<? extends ActionInput> getInputFiles() {
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
  public Sequence<String> getStarlarkArgv() throws EvalException, InterruptedException {
    try {
      return StarlarkList.immutableCopyOf(getArguments());
    } catch (CommandLineExpansionException ex) {
      throw new EvalException(ex);
    }
  }

  /** Returns the out-of-band execution data for this action. */
  @Override
  public Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException, InterruptedException {
    return ImmutableList.copyOf(getCommandLines().allArguments());
  }

  @Override
  public Sequence<CommandLineArgsApi> getStarlarkArgs() throws EvalException {
    ImmutableList.Builder<CommandLineArgsApi> result = ImmutableList.builder();
    ImmutableSet<Artifact> directoryInputs =
        getInputs().toList().stream().filter(Artifact::isDirectory).collect(toImmutableSet());
    for (CommandLineAndParamFileInfo commandLine : getCommandLines().getCommandLines()) {
      result.add(Args.forRegisteredAction(commandLine, directoryInputs));
    }
    return StarlarkList.immutableCopyOf(result.build());
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
  public NestedSet<Artifact> getPossibleInputsForTesting() {
    return null;
  }

  /**
   * Locally rewrites a .jdeps file to replace missing config prefixes.
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
   * <p>If the executor doesn't strip config prefixes (i.e. config stripping isn't turned on as a
   * feature), this is a trivial copy.
   *
   * @param spawnResult the executor action that created the possibly stripped .jdeps output
   * @param outputDepsProto path to the .jdeps output
   * @param actionInputs all inputs to the current action
   * @param actionExecutionContext the action execution context
   * @return the full deps proto (also wdritten to disk to satisfy the action's declared output)
   */
  static Deps.Dependencies createFullOutputDeps(
      SpawnResult spawnResult,
      Artifact outputDepsProto,
      NestedSet<Artifact> actionInputs,
      ActionExecutionContext actionExecutionContext)
      throws IOException {

    // Get the executor-produce jdeps.
    InputStream inMemoryOutput = spawnResult.getInMemoryOutput(outputDepsProto);
    InputStream inputStream =
        inMemoryOutput == null
            ? actionExecutionContext.getInputPath(outputDepsProto).getInputStream()
            : inMemoryOutput;
    Deps.Dependencies executorJdeps =
        Deps.Dependencies.parseFrom(inputStream, ExtensionRegistry.getEmptyRegistry());
    inputStream.close();

    // For each of the action's generated inputs, map its stripped path to its full path.
    PathFragment outputRoot = outputDepsProto.getExecPath().subFragment(0, 1);
    Map<PathFragment, PathFragment> strippedToFullPaths = new HashMap<>();
    for (Artifact actionInput : actionInputs.toList()) {
      if (actionInput.isSourceArtifact()) {
        continue;
      }
      // Turns "bazel-out/x86-fastbuild/bin/foo" to "bazel-out/bin/foo".
      PathFragment strippedPath = outputRoot.getRelative(actionInput.getExecPath().subFragment(2));
      if (strippedToFullPaths.put(strippedPath, actionInput.getExecPath()) != null) {
        // If an entry already exists, that means different inputs reduce to the same stripped path.
        // That also means PathStripper would exempt this action from path stripping, so the
        // executor-produced .jdeps already includes full paths. No need to update it.
        return executorJdeps;
      }
    }

    // Rewrite the .jdeps proto with full paths.
    Deps.Dependencies.Builder fullDepsBuilder = Deps.Dependencies.newBuilder(executorJdeps);
    for (Deps.Dependency.Builder dep : fullDepsBuilder.getDependencyBuilderList()) {
      PathFragment pathOnExecutor = PathFragment.create(dep.getPath());
      PathFragment fullPath = strippedToFullPaths.get(pathOnExecutor);
      if (fullPath == null && pathOnExecutor.subFragment(0, 1).equals(outputRoot)) {
        // The stripped path -> full path map failed, which means the paths weren't stripped. Fast-
        // return the original jdeps to save unnecessary CPU time.
        return executorJdeps;
      }
      dep.setPath(fullPath == null ? pathOnExecutor.getPathString() : fullPath.getPathString());
    }
    Deps.Dependencies fullOutputDeps = fullDepsBuilder.build();

    // Write the updated proto back to the filesystem. If the executor produced in-memory-only
    // outputs (see getInMemoryOutput above), the filesystem version doesn't exist and we can skip
    // this. Note that in-memory and filesystem outputs aren't necessarily mutually exclusive.
    Path fsPath = actionExecutionContext.getInputPath(outputDepsProto);
    if (fsPath.exists()) {
      fsPath.setWritable(true);
      FileSystemUtils.writeContent(fsPath, fullOutputDeps.toByteArray());
    }

    return fullOutputDeps;
  }

  /** Reads the full {@code .jdeps} output from the given spawn results. */
  private Deps.Dependencies readFullOutputDeps(
      List<SpawnResult> results, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    try {
      return createFullOutputDeps(
          Iterables.getOnlyElement(results), outputDepsProto, getInputs(), actionExecutionContext);
    } catch (IOException e) {
      throw new EnvironmentalExecException(
              e, createFailureDetail(".jdeps read IOException", Code.JDEPS_READ_IO_EXCEPTION))
          .toActionExecutionException(this);
    }
  }

  private ActionExecutionException createActionExecutionException(Exception e, Code detailedCode) {
    DetailedExitCode detailedExitCode =
        DetailedExitCode.of(createFailureDetail(Strings.nullToEmpty(e.getMessage()), detailedCode));
    return new ActionExecutionException(e, this, /*catastrophe=*/ false, detailedExitCode);
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setJavaCompile(JavaCompile.newBuilder().setCode(detailedCode))
        .build();
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
        Deps.Dependencies dependencies = null;
        if (outputDepsProto != null) {
          dependencies = readFullOutputDeps(results, actionExecutionContext);
        }
        if (reducedClasspath == null) {
          return ActionContinuationOrResult.of(ActionResult.create(results));
        }

        if (compilationType == CompilationType.TURBINE) {
          actionExecutionContext
              .getContext(JavaCompileActionContext.class)
              .insertDependencies(outputDepsProto, dependencies);
        }
        if (!dependencies.getRequiresReducedClasspathFallback()) {
          return ActionContinuationOrResult.of(ActionResult.create(results));
        }

        logger.atInfo().atMostEvery(1, SECONDS).log(
            "Failed reduced classpath compilation for %s", JavaCompileAction.this.prettyPrint());
        // Fall back to running with the full classpath. This requires first deleting potential
        // artifacts generated by the reduced action and clearing the metadata caches.
        try {
          deleteOutputs(
              actionExecutionContext.getExecRoot(),
              actionExecutionContext.getPathResolver(),
              /*bulkDeleter=*/ null,
              // We don't create any tree artifacts anyway.
              /*outputPrefixForArchivedArtifactsCleanup=*/ null);
        } catch (IOException e) {
          throw new EnvironmentalExecException(
                  e,
                  createFailureDetail(
                      "Failed to delete reduced action outputs",
                      Code.REDUCED_CLASSPATH_FALLBACK_CLEANUP_FAILURE))
              .toActionExecutionException(JavaCompileAction.this);
        }
        actionExecutionContext.getMetadataHandler().resetOutputs(getOutputs());
        Spawn spawn;
        try {
          spawn = getReducedSpawn(actionExecutionContext, reducedClasspath, /* fallback=*/ true);
        } catch (CommandLineExpansionException e) {
          Code detailedCode = Code.COMMAND_LINE_EXPANSION_FAILURE;
          throw createActionExecutionException(e, detailedCode);
        }
        SpawnContinuation fallbackContinuation =
            actionExecutionContext
                .getContext(SpawnStrategyResolver.class)
                .beginExecution(spawn, actionExecutionContext);
        return new JavaFallbackActionContinuation(
            actionExecutionContext, results, fallbackContinuation);
      } catch (ExecException e) {
        throw e.toActionExecutionException(JavaCompileAction.this);
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
                  outputDepsProto, readFullOutputDeps(fallbackResults, actionExecutionContext));
        }
        return ActionContinuationOrResult.of(
            ActionResult.create(
                ImmutableList.copyOf(Iterables.concat(primaryResults, fallbackResults))));
      } catch (ExecException e) {
        throw e.toActionExecutionException(JavaCompileAction.this);
      }
    }
  }
}
