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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Stopwatch;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Table;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionGraphDumper;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactMTimeCache;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.Builder;
import com.google.devtools.build.lib.actions.DatabaseDependencyChecker;
import com.google.devtools.build.lib.actions.DependencyChecker;
import com.google.devtools.build.lib.actions.Dumper;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.IncrementalDependencyChecker;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.MakefileDumper;
import com.google.devtools.build.lib.actions.ParallelBuilder;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.ArtifactMetadataCache;
import com.google.devtools.build.lib.actions.cache.MetadataCache;
import com.google.devtools.build.lib.blaze.BlazeModule;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.buildtool.BuildRequest.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.CheckUpToDateFilter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.FileWriteStrategy;
import com.google.devtools.build.lib.exec.FilesetActionContextImpl;
import com.google.devtools.build.lib.exec.LegacyActionInputFileCache;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.exec.SymlinkTreeStrategy;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner.LoadingResult;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BuildView;
import com.google.devtools.build.lib.view.BuildView.AnalysisResult;
import com.google.devtools.build.lib.view.CompilationPrerequisitesProvider;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.FileProvider;
import com.google.devtools.build.lib.view.FilesToCompileProvider;
import com.google.devtools.build.lib.view.InputFileConfiguredTarget;
import com.google.devtools.build.lib.view.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.view.TempsProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.ViewCreationFailedException;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.view.fileset.FilesetActionContext;
import com.google.devtools.build.lib.view.test.TestActionContext;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * This class manages the execution phase. The entry point is {@link #executeBuild}.
 *
 * <p>This is only intended for use by {@link BuildTool}.
 *
 * <p>This class contains an ActionCache, and refers to the BlazeRuntime's BuildView and
 * PackageCache.
 *
 * @see BuildTool
 * @see BuildView
 */
public class ExecutionTool {
  private static class StrategyConverter {
    private Table<Class<? extends ActionContext>, String, ActionContext> classMap =
        HashBasedTable.create();
    private Map<Class<? extends ActionContext>, ActionContext> defaultClassMap =
        new HashMap<>();

    /**
     * Aggregates all {@link ActionContext}s that are in {@code contextProviders}.
     */
    @SuppressWarnings("unchecked")
    private StrategyConverter(Iterable<ActionContextProvider> contextProviders) {
      for (ActionContextProvider provider : contextProviders) {
        for (ActionContext strategy : provider.getActionContexts()) {
          ExecutionStrategy annotation =
              strategy.getClass().getAnnotation(ExecutionStrategy.class);
          if (annotation != null) {
            defaultClassMap.put(annotation.contextType(), strategy);

            for (String name : annotation.name()) {
              classMap.put(annotation.contextType(), name, strategy);
            }
          }
        }
      }
    }

    @SuppressWarnings("unchecked")
    private <T extends ActionContext> T getStrategy(Class<T> clazz, String name) {
      return (T) (name.isEmpty() ? defaultClassMap.get(clazz) : classMap.get(clazz, name));
    }

    private String getValidValues(Class<? extends ActionContext> context) {
      return Joiner.on(", ").join(Ordering.natural().sortedCopy(classMap.row(context).keySet()));
    }

    private String getUserFriendlyName(Class<? extends ActionContext> context) {
      ActionContextMarker marker = context.getAnnotation(ActionContextMarker.class);
      return marker != null
          ? marker.name()
          : context.getSimpleName();
    }
  }

  static final Logger LOG = Logger.getLogger(ExecutionTool.class.getName());

  private final BlazeRuntime runtime;
  private final BuildRequest request;
  private BlazeExecutor executor;
  private ActionInputFileCache fileCache;
  private List<ActionContextProvider> actionContextProviders;

  private Map<String, ActionContext> spawnStrategyMap = new HashMap<>();
  private List<ActionContext> strategies = new ArrayList<>();

  ExecutionTool(BlazeRuntime runtime, BuildRequest request) throws ExecutorInitException {
    this.runtime = runtime;
    this.request = request;

    List<ActionContextConsumer> actionContextConsumers = new ArrayList<>();
    actionContextProviders = new ArrayList<>();
    for (BlazeModule module : runtime.getBlazeModules()) {
      ActionContextProvider provider = module.getActionContextProvider();
      if (provider != null) {
        actionContextProviders.add(provider);
      }

      ActionContextConsumer consumer = module.getActionContextConsumer();
      if (consumer != null) {
        actionContextConsumers.add(consumer);
      }
    }

    actionContextProviders.add(new FilesetActionContextImpl.Provider(runtime.getReporter()));

    strategies.add(new FileWriteStrategy(request));
    strategies.add(new SymlinkTreeStrategy(runtime.getOutputService(), runtime.getBinTools()));

    StrategyConverter strategyConverter = new StrategyConverter(actionContextProviders);
    strategies.add(strategyConverter.getStrategy(FilesetActionContext.class, ""));

    for (ActionContextConsumer consumer : actionContextConsumers) {
      // There are many different SpawnActions, and we want to control the action context they use
      // independently from each other, for example, to run genrules locally and Java compile action
      // in prod. Thus, for SpawnActions, we decide the action context to use not only based on the
      // context class, but also the mnemonic of the action.
      for (Map.Entry<String, String> entry : consumer.getSpawnActionContexts().entrySet()) {
        SpawnActionContext context =
            strategyConverter.getStrategy(SpawnActionContext.class, entry.getValue());
        if (context == null) {
          throw new ExecutorInitException(String.format(
              "'%s' is an invalid value for spawn strategy. Valid values are: %s",
              entry.getValue(), strategyConverter.getValidValues(SpawnActionContext.class)));
        }

        spawnStrategyMap.put(entry.getKey(), context);
      }

      for (Map.Entry<Class<? extends ActionContext>, String> entry :
          consumer.getActionContexts().entrySet()) {
        ActionContext context = strategyConverter.getStrategy(entry.getKey(), entry.getValue());
        if (context != null) {
          strategies.add(context);
        } else if (!entry.getValue().isEmpty()) {
          // If the action context consumer requested the default value (by passing in the empty
          // string), we do not throw the exception, because we assume that whoever put together
          // the modules in this Blaze binary knew what they were doing.
          throw new ExecutorInitException(String.format(
              "'%s' is an invalid value for %s strategy. Valid values are: %s",
              entry.getValue(), strategyConverter.getUserFriendlyName(entry.getKey()),
              strategyConverter.getValidValues(entry.getKey())));
        }
      }
    }

    // If tests are to be run during build, too, we have to explicitly load the test action context.
    if (request.shouldRunTests()) {
      strategies.add(strategyConverter.getStrategy(TestActionContext.class,
          request.getOptions(ExecutionOptions.class).testStrategy));
    }
  }

  Executor getExecutor() throws ExecutorInitException {
    if (executor == null) {
      executor = createExecutor();
    }
    return executor;
  }

  /**
   * Creates an executor for the current set of blaze runtime, execution options, and request.
   */
  private BlazeExecutor createExecutor()
      throws ExecutorInitException {
    return new BlazeExecutor(
        runtime.getDirectories().getExecRoot(),
        runtime.getDirectories().getOutputPath(),
        getReporter(),
        getEventBus(),
        runtime.getClock(),
        request,
        request.getOptions(ExecutionOptions.class).verboseFailures,
        strategies,
        spawnStrategyMap,
        actionContextProviders);
  }

  void init() throws ExecutorInitException {
    createToolsSymlinks();
    getExecutor();
  }

  void shutdown() {
    for (ActionContextProvider actionContextProvider : actionContextProviders) {
      actionContextProvider.executionPhaseEnding();
    }
  }

  /**
   * Performs the execution phase (phase 3) of the build, in which the Builder
   * is applied to the action graph to bring the targets up to date. (This
   * function will return prior to execution-proper if --nobuild was specified.)
   *
   * @param loadingResult the loading phase output
   * @param analysisResult the analysis phase output
   * @param buildResult the mutable build result
   * @param skyframeExecutor the skyframe executor (if any)
   */
  void executeBuild(LoadingResult loadingResult, AnalysisResult analysisResult,
      BuildResult buildResult, @Nullable SkyframeExecutor skyframeExecutor,
      BuildConfigurationCollection configurations)
      throws BuildFailedException, InterruptedException, AbruptExitException, TestExecException,
      ViewCreationFailedException {
    Stopwatch timer = Stopwatch.createStarted();
    prepare(loadingResult, configurations);

    ActionGraph actionGraph = analysisResult.getActionGraph();
    executor.setActionGraph(actionGraph);

    // Get top-level artifacts.
    ImmutableSet<Artifact> artifactsToBuild = analysisResult.getArtifactsToBuild();
    Multimap<ConfiguredTarget, Artifact> targetCompletionMap =
        analysisResult.getTargetCompletionMap();

    // Optionally dump information derived from the action graph.
    // Ideally, we would do this after creating the symlinks, so that
    // we could display the short version from getPrettyPath().
    // However, the user might have specified --nobuild, in which case
    // we really shouldn't be writing anything to disk.  Note that
    // we'll take advantage of the symlinks, if they already exist, in
    // getPrettyPath.
    maybeDump(request, executor, actionGraph, artifactsToBuild);

    // If --nobuild is specified, this request completes successfully without
    // execution.  (We only got here because of --dump_action_graph or similar.)
    if (!request.getBuildOptions().performExecutionPhase) {
      return;
    }

    // Create symlinks only after we've verified that we're actually
    // supposed to build something.
    if (getWorkspace().getFileSystem().supportsSymbolicLinks()) {
      List<BuildConfiguration> targetConfigurations =
          getView().getConfigurationCollection().getTargetConfigurations();
      // TODO(bazel-team): This is not optimal - we retain backwards compatibility in the case where
      // there's only a single configuration, but we don't create any symlinks in the multi-config
      // case. Can we do better? [multi-config]
      if (targetConfigurations.size() == 1) {
        OutputDirectoryLinksUtils.createOutputDirectoryLinks(getWorkspace(), getExecRoot(),
            runtime.getOutputPath(), getReporter(), targetConfigurations.get(0),
            request.getSymlinkPrefix());
      }
    }

    OutputService outputService = runtime.getOutputService();
    if (outputService != null) {
      outputService.startBuild();
    } else {
      startLocalOutputBuild(); // TODO(bazel-team): this could be just another OutputService
    }

    Pair<ActionCache, MetadataCache> caches = getCaches();
    ArtifactMTimeCache artifactMTimeCache = getView().getArtifactMTimeCache();
    String workspace = (outputService != null) ? outputService.getWorkspace() : null;
    boolean buildIsIncremental = !useSkyframeFull(skyframeExecutor) &&
        artifactMTimeCache.isApplicableToBuild(
            request.getBuildOptions().useIncrementalDependencyChecker,
            artifactsToBuild, caches.second, workspace, analysisResult.hasStaleActionData());
    Builder builder = createBuilder(request, executor, caches.first, caches.second,
        artifactMTimeCache, buildIsIncremental, skyframeExecutor);

    //
    // Execution proper.  All statements below are logically nested in
    // begin/end pairs.  No early returns or exceptions please!
    //

    Collection<ConfiguredTarget> configuredTargets = buildResult.getActualTargets();
    getEventBus().post(new ExecutionStartingEvent(configuredTargets));

    getReporter().progress(null, "Building...");

    // Conditionally record dependency-checker log:
    ExplanationHandler explanationHandler =
        installExplanationHandler(request.getBuildOptions().explanationPath,
                                  request.getOptionsDescription());

    Set<Artifact> builtArtifacts = new HashSet<>();
    boolean interrupted = false;
    try {
      if (request.isRunningInEmacs()) {
        // The syntax of this message is tightly constrained by lisp/progmodes/compile.el in emacs
        request.getOutErr().printErrLn("blaze: Entering directory `" + getExecRoot() + "/'");
      }
      for (ActionContextProvider actionContextProvider : actionContextProviders) {
        actionContextProvider.executionPhaseStarting(
            fileCache,
            useSkyframeFull(skyframeExecutor) ? null : artifactMTimeCache,
            actionGraph,
            artifactsToBuild);
      }
      executor.executionPhaseStarting(request.getOutErr());
      if (useSkyframeFull(skyframeExecutor)) {
        skyframeExecutor.drainChangedFiles();
      }

      configureResourceManager(request);

      Profiler.instance().markPhase(ProfilePhase.EXECUTE);

      try {
        if (artifactMTimeCache != null) {
          artifactMTimeCache.beforeBuild();
        }
        // This probably does not work with Skyframe, because then the modified file set
        // returned by the runtime is not right (since it is updated in PackageCache). This also
        // fails to work when the order of commands is build-query-build (because changes during
        // between the first build command and the query command get lost).
        builder.buildArtifacts(artifactsToBuild,
            analysisResult.getExclusiveTestArtifacts(),
            analysisResult.getDependentActionGraph(),
            executor, runtime.getModifiedFileSetForSourceFiles(), builtArtifacts);
      } finally {
        if (artifactMTimeCache != null) {
          artifactMTimeCache.afterBuild();
        }
      }
      if (artifactMTimeCache != null) {
        artifactMTimeCache.markBuildSuccessful(artifactsToBuild);
      }

    } catch (InterruptedException e) {
      interrupted = true;
      throw e;
    } finally {
      if (request.isRunningInEmacs()) {
        request.getOutErr().printErrLn("blaze: Leaving directory `" + getExecRoot() + "/'");
      }
      getReporter().progress(null, "Building complete.");
      buildResult.setIncrementality(builder.getPercentageCached());

      // Transfer over source file "last save time" stats so the remote logger can find them.
      runtime.getEventBus().post(
          new ExecutionFinishedEvent(
              caches.second.getChangedFileSaveTimes(),
              caches.second.getLastFileSaveTime()));

      // Disable system load polling (noop if it was not enabled).
      ResourceManager.instance().setAutoSensing(false);
      executor.executionPhaseEnding(request.getOutErr());
      for (ActionContextProvider actionContextProvider : actionContextProviders) {
        actionContextProvider.executionPhaseEnding();
      }

      Profiler.instance().markPhase(ProfilePhase.FINISH);

      if (!interrupted) {
        saveCaches(caches.first, caches.second);
      }

      long startTime = Profiler.nanoTimeMaybe();
      determineSuccessfulTargets(request, buildResult, configuredTargets, builtArtifacts,
          targetCompletionMap, timer);
      showBuildResult(request, buildResult, configuredTargets);
      Preconditions.checkNotNull(buildResult.getSuccessfulTargets());
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.INFO, "Show results");
      if (explanationHandler != null) {
        uninstallExplanationHandler(explanationHandler);
      }
      // Finalize output service last, so that if we do throw an exception, we know all the other
      // code has already run.
      if (runtime.getOutputService() != null) {
        boolean isBuildSuccessful =
            buildResult.getSuccessfulTargets().size() == configuredTargets.size();
        runtime.getOutputService().finalizeBuild(isBuildSuccessful);
      }
    }
  }

  private void prepare(LoadingResult loadingResult, BuildConfigurationCollection configurations)
      throws ViewCreationFailedException {
    // Prepare for build.
    Profiler.instance().markPhase(ProfilePhase.PREPARE);

    // Create some tools symlinks / cleanup per-build state
    createActionLogDirectory();

    // Plant the symlink forest.
    plantSymlinkForest(loadingResult, configurations);
  }

  private void createToolsSymlinks() throws ExecutorInitException {
    try {
      runtime.getBinTools().setupBuildTools();
    } catch (ExecException e) {
      throw new ExecutorInitException("Tools symlink creation failed: "
          + e.getMessage() + "; build aborted", e);
    }
  }

  private void plantSymlinkForest(LoadingResult loadingResult,
      BuildConfigurationCollection configurations) throws ViewCreationFailedException {
    try {
      FileSystemUtils.deleteTreesBelowNotPrefixed(getExecRoot(),
          new String[] { ".", "_", "blaze-"});
      // Delete the build configuration's temporary directories
      for (BuildConfiguration configuration : configurations.getTargetConfigurations()) {
        configuration.prepareForExecutionPhase();
      }
      FileSystemUtils.plantLinkForest(loadingResult.getPackageRoots(), getExecRoot());
    } catch (IOException e) {
      throw new ViewCreationFailedException("Source forest creation failed: " + e.getMessage()
          + "; build aborted", e);
    }
  }

  private void createActionLogDirectory() throws ViewCreationFailedException {
    Path directory = runtime.getDirectories().getActionConsoleOutputDirectory();
    try {
      if (directory.exists()) {
        FileSystemUtils.deleteTree(directory);
      }
      directory.createDirectory();
    } catch (IOException ex) {
      throw new ViewCreationFailedException("couldn't delete action output directory: " +
          ex.getMessage());
    }
  }

  private void maybeDump(BuildRequest request, Executor executor, ActionGraph actionGraph,
      Collection<Artifact> artifactsToBuild) {
    BuildRequestOptions options = request.getBuildOptions();
    long startTime = Profiler.nanoTimeMaybe();

    List<Dumper> dumpers = new ArrayList<>();
    if (options.dumpMakefile) {
      dumpers.add(new MakefileDumper(executor, artifactsToBuild, actionGraph));
    }
    if (options.dumpActionGraph) {
      dumpers.add(new ActionGraphDumper(artifactsToBuild, actionGraph,
          options.dumpActionGraphForPackage, options.dumpActionGraphWithMiddlemen));
    }
    if (options.dumpTargets != null) {
      dumpers.add(new TargetsDumper(artifactsToBuild, actionGraph,
          runtime.getPackageManager(), options.dumpTargets, options.dumpHostDeps));
    }
    for (Dumper dumper : dumpers) {
      try {
        PrintStream out;
        if (options.dumpToStdout) {
          out = new PrintStream(getReporter().getOutErr().getOutputStream());
          getReporter().info(null, "Dumping " + dumper.getName() + " to stdout");
        } else {
          Path filename = getExecRoot().getRelative(dumper.getFileName());
          out = new PrintStream(filename.getOutputStream());
          getReporter().info(
              null,
              "Dumping " + dumper.getName() + " to "
                  + OutputDirectoryLinksUtils.getPrettyPath(
                      filename, getWorkspace(), request.getSymlinkPrefix()));
        }
        dumper.dump(out);
      } catch (IOException e) {
        getReporter().error(
            null,
            "I/O error while dumping " + dumper.getName() + "" + dumper.getName() + ": "
                + e.getMessage());
      } finally {
        Profiler.instance()
            .logSimpleTask(startTime, ProfilerTask.INFO, "Generating Makefile.blaze");
      }
    }
  }

  /**
   * Prepare for a local output build.
   */
  private void startLocalOutputBuild() throws BuildFailedException {
    long startTime = Profiler.nanoTimeMaybe();

    try {
      Path outputPath = runtime.getOutputPath();
      Path localOutputPath = runtime.getDirectories().getLocalOutputPath();

      if (outputPath.isSymbolicLink()) {
        // Remove the existing symlink first.
        outputPath.delete();
        if (localOutputPath.exists()) {
          // Pre-existing local output directory. Move to outputPath.
          localOutputPath.renameTo(outputPath);
        }
      }
    } catch (IOException e) {
      throw new BuildFailedException(e.getMessage());
    } finally {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.INFO,
          "Starting local output build");
    }
  }

  /**
   * If a path is supplied, creates and installs an ExplanationHandler. Returns
   * an instance on success. Reports an error and returns null otherwise.
   */
  private ExplanationHandler installExplanationHandler(PathFragment explanationPath,
                                                       String allOptions) {
    if (explanationPath == null) {
      return null;
    }
    ExplanationHandler handler;
    try {
      handler = new ExplanationHandler(
          getWorkspace().getRelative(explanationPath).getOutputStream(),
          allOptions);
    } catch (IOException e) {
      getReporter().warn(null, String.format(
          "Cannot write explanation of rebuilds to file '%s': %s",
          explanationPath, e.getMessage()));
      return null;
    }
    getReporter().info(null, "Writing explanation of rebuilds to '" + explanationPath + "'");
    getReporter().addHandler(handler);
    return handler;
  }

  /**
   * Uninstalls the specified ExplanationHandler (if any) and closes the log
   * file.
   */
  private void uninstallExplanationHandler(ExplanationHandler handler) {
    if (handler != null) {
      getReporter().removeHandler(handler);
      handler.log.close();
    }
  }

  /**
   * An EventHandler implementation that records DEPCHECKER events into a log
   * file, iff the --explain flag is specified during a build.
   */
  private static class ExplanationHandler implements EventHandler {

    private static final Set<EventKind> DEPCHECKER = EnumSet.of(EventKind.DEPCHECKER);

    private final PrintWriter log;

    private ExplanationHandler(OutputStream log, String optionsDescription) {
      this.log = new PrintWriter(log);
      this.log.println("Build options: " + optionsDescription);
    }

    /** Implements {@link EventHandler#getEventMask}. */
    @Override
    public Set<EventKind> getEventMask() {
      return DEPCHECKER;
    }

    /** Implements {@link EventHandler#handle}. */
    @Override
    public void handle(Event event) {
      log.println(event.getMessage());
    }
  }

  /**
   * Computes the result of the build. Sets the list of successful (up-to-date)
   * targets in the request object.
   *
   * @param request The build request, which specifies various options.
   *                This function sets the successfulTargets field of the request object.
   * @param configuredTargets The configured targets whose artifacts are to be
   *                          built.
   * @param builtArtifacts Set of successfully built artifacts.
   * @param timer A timer that was started when the execution phase started.
   */
  private void determineSuccessfulTargets(BuildRequest request, BuildResult result,
      Collection<ConfiguredTarget> configuredTargets, Set<Artifact> builtArtifacts,
      Multimap<ConfiguredTarget, Artifact> artifactsMap, Stopwatch timer) {
    Collection<ConfiguredTarget> successfulTargets = new LinkedHashSet<>();
    for (ConfiguredTarget target : configuredTargets) {
      if (builtArtifacts.containsAll(artifactsMap.get(target))) {
        successfulTargets.add(target);
      }
    }
    getEventBus().post(new ExecutionPhaseCompleteEvent(
        successfulTargets, request.getViewOptions().keepGoing,
        timer.stop().elapsed(TimeUnit.MILLISECONDS)));
    result.setSuccessfulTargets(successfulTargets);
  }

  /**
   * Shows the result of the build. Information includes the list of up-to-date
   * and failed targets and list of output artifacts for successful targets
   *
   * @param request The build request, which specifies various options.
   * @param configuredTargets The configured targets whose artifacts are to be
   *   built.
   *
   * TODO(bazel-team): (2010) refactor into using Reporter and info/progress events
   */
  private void showBuildResult(BuildRequest request, BuildResult result,
      Collection<ConfiguredTarget> configuredTargets) {
    // NOTE: be careful what you print!  We don't want to create a consistency
    // problem where the summary message and the exit code disagree.  The logic
    // here is already complex.

    // Filter the targets we care about into two buckets:
    Collection<ConfiguredTarget> succeeded = new ArrayList<>();
    Collection<ConfiguredTarget> failed = new ArrayList<>();
    for (ConfiguredTarget target : configuredTargets) {
      // TODO(bazel-team): this is quite ugly. Add a marker provider for this check.
      if (target instanceof InputFileConfiguredTarget) {
        // Suppress display of source files and filegroups (because we do no
        // work to build them).
        continue;
      } else if (target.getTarget() instanceof Rule) {
        if (((Rule) target.getTarget()).getRuleClass().equals("filegroup")) {
          continue;
        }
      }
      if (target.getTarget() instanceof Rule) {
        Rule rule = (Rule) target.getTarget();
        if (rule.getRuleClass().contains("$")) {
          // Suppress display of hidden rules
          continue;
        }
      }
      if (target instanceof OutputFileConfiguredTarget) {
        // Suppress display of generated files (because they appear underneath
        // their generating rule), EXCEPT those ones which are not part of the
        // filesToBuild of their generating rule (e.g. .par, _deploy.jar
        // files), OR when a user explicitly requests an output file but not
        // its rule.
        TransitiveInfoCollection generatingRule =
            getView().getGeneratingRule((OutputFileConfiguredTarget) target);
        if (CollectionUtils.containsAll(
            generatingRule.getProvider(FileProvider.class).getFilesToBuild(),
            target.getProvider(FileProvider.class).getFilesToBuild()) &&
            configuredTargets.contains(generatingRule)) {
          continue;
        }
      }

      Collection<ConfiguredTarget> successfulTargets = result.getSuccessfulTargets();
      (successfulTargets.contains(target) ? succeeded : failed).add(target);
    }

    // Suppress summary if --show_result value is exceeded:
    if (succeeded.size() + failed.size() > request.getBuildOptions().maxResultTargets) {
      return;
    }

    OutErr outErr = request.getOutErr();

    for (ConfiguredTarget target : succeeded) {
      Label label = target.getLabel();
      // For up-to-date targets report generated artifacts, but only
      // if they have associated action and not middleman artifacts.
      boolean headerFlag = true;
      for (Artifact artifact : getFilesToBuild(target, request)) {
        if (!artifact.isSourceArtifact()) {
          if (headerFlag) {
            outErr.printErr("Target " + label + " up-to-date:\n");
            headerFlag = false;
          }
          outErr.printErrLn("  " +
              OutputDirectoryLinksUtils.getPrettyPath(
                  artifact.getPath(), getWorkspace(), request.getSymlinkPrefix()));
        }
      }
      if (headerFlag) {
        outErr.printErr(
            "Target " + label + " up-to-date (nothing to build)\n");
      }
    }

    for (ConfiguredTarget target : failed) {
      outErr.printErr("Target " + target.getLabel() + " failed to build\n");

      // For failed compilation, it is still useful to examine temp artifacts,
      // (ie, preprocessed and assembler files).
      TempsProvider tempsProvider = target.getProvider(TempsProvider.class);
      if (tempsProvider != null) {
        for (Artifact temp : tempsProvider.getTemps()) {
          if (temp.getPath().exists()) {
            outErr.printErrLn("  See temp at " +
                OutputDirectoryLinksUtils.getPrettyPath(
                    temp.getPath(), getWorkspace(), request.getSymlinkPrefix()));
          }
        }
      }
    }
    if (!failed.isEmpty() && !request.getOptions(ExecutionOptions.class).verboseFailures) {
      outErr.printErr("Use --verbose_failures to see the command lines of failed build steps.\n");
    }
  }

  /**
   * Gets all the files to build for a given target and build request.
   * There may be artifacts that should be built which are not represented in the
   * configured target graph.  Currently, this only occurs when "--save_temps" is on.
   *
   * @param target configured target
   * @param request the build request
   * @return artifacts to build
   */
  private static Collection<Artifact> getFilesToBuild(ConfiguredTarget target,
      BuildRequest request) {
    ImmutableSet.Builder<Artifact> result = ImmutableSet.builder();
    if (request.getBuildOptions().compileOnly) {
      FilesToCompileProvider provider = target.getProvider(FilesToCompileProvider.class);
      if (provider != null) {
        result.addAll(provider.getFilesToCompile());
      }
    } else if (request.getBuildOptions().compilationPrerequisitesOnly) {
      CompilationPrerequisitesProvider provider =
          target.getProvider(CompilationPrerequisitesProvider.class);
      if (provider != null) {
        result.addAll(provider.getCompilationPrerequisites());
      }
    } else {
      FileProvider provider = target.getProvider(FileProvider.class);
      if (provider != null) {
        result.addAll(provider.getFilesToBuild());
      }
    }
    TempsProvider tempsProvider = target.getProvider(TempsProvider.class);
    if (tempsProvider != null) {
      result.addAll(tempsProvider.getTemps());
    }

    return result.build();
  }

  private Pair<ActionCache, MetadataCache> getCaches()
      throws LocalEnvironmentException {
    try {
      ActionCache actionCache = runtime.getPersistentActionCache();
      MetadataCache metadataCache = runtime.getPersistentMetadataCache();
      return Pair.of(actionCache, metadataCache);
    } catch (IOException e) {
      // TODO(bazel-team): (2010) Ideally we should just remove all cache data and reinitialize
      // caches.
      LoggingUtil.logToRemote(Level.WARNING, "Failed to initialize caches: " + e.getMessage(), e);
      throw new LocalEnvironmentException("couldn't create caches: " + e.getMessage() +
          ". If error persists, use 'blaze clean'");
    }
  }

  private static boolean useSkyframeFull(SkyframeExecutor skyframeExecutor) {
    return skyframeExecutor != null && skyframeExecutor.skyframeBuild();
  }

  private Builder createBuilder(BuildRequest request,
      Executor executor,
      ActionCache actionCache, MetadataCache metadataCache,
      @Nullable ArtifactMTimeCache artifactMTimeCache, boolean buildIsIncremental,
      @Nullable SkyframeExecutor skyframeExecutor) {
    boolean skyframeFull = useSkyframeFull(skyframeExecutor);
    BuildRequest.BuildRequestOptions options = request.getBuildOptions();
    boolean verboseExplanations = options.verboseExplanations;
    boolean keepGoing = request.getViewOptions().keepGoing;

    metadataCache.setInvocationStartTime(new Date().getTime());

    Path actionOutputRoot = runtime.getDirectories().getActionConsoleOutputDirectory();
    PackageUpToDateChecker packageUpToDateChecker = runtime.getPackageUpToDateChecker();
    Predicate<Action> executionFilter = CheckUpToDateFilter.fromOptions(
        request.getOptions(ExecutionOptions.class));

    // jobs should have been verified in BuildRequest#validateOptions().
    Preconditions.checkState(options.jobs >= -1);
    int actualJobs = options.jobs == 0 ? 1 : options.jobs;  // Treat 0 jobs as a single task.
    if (!skyframeFull) {
      BatchStat batchStat = null;
      if (runtime.getOutputService() != null) {
        batchStat = runtime.getOutputService().getBatchStatter();
      }

      Preconditions.checkState(!runtime.getSkyframeExecutor().skyframeBuild());
      ArtifactMetadataCache result = new ArtifactMetadataCache(
          metadataCache, runtime.getTimestampGranularityMonitor(), batchStat,
          packageUpToDateChecker);

      ArtifactMetadataCache artifactMetadataCache = buildIsIncremental
          ? artifactMTimeCache.getArtifactMetadataCache()
          : result;
      artifactMTimeCache.setArtifactMetadataCache(artifactMetadataCache);
      ActionInputFileCache buildSingleFileCache = createBuildSingleFileCache(
          executor.getExecRoot());
      fileCache = new LegacyActionInputFileCache(artifactMetadataCache,
          buildSingleFileCache, executor.getExecRoot().getPathFile());

      DependencyChecker dependencyChecker = buildIsIncremental
          ? new IncrementalDependencyChecker(
              actionCache, getView().getArtifactFactory(),
              artifactMetadataCache, artifactMTimeCache, packageUpToDateChecker,
              executionFilter, getEventBus(),
              verboseExplanations, enableIncrementalGraphPruning(),
              runtime.getAllowedMissingInputs())
          : new DatabaseDependencyChecker(
              actionCache, getView().getArtifactFactory(),
              artifactMetadataCache, packageUpToDateChecker,
              executionFilter, verboseExplanations, runtime.getAllowedMissingInputs());

      return new ParallelBuilder(getReporter(), getEventBus(), dependencyChecker, actualJobs,
          options.progressReportInterval, options.useBuilderStatistics, keepGoing,
          actionOutputRoot, fileCache, runtime.getAllowedMissingInputs());
    } else {
      // Unfortunately, the exec root cache is not shared with caches in the remote execution
      // client.
      fileCache = createBuildSingleFileCache(executor.getExecRoot());
      skyframeExecutor.setActionOutputRoot(actionOutputRoot);
      return new SkyframeBuilder(skyframeExecutor,
          new ActionCacheChecker(actionCache, getView().getArtifactFactory(),
              packageUpToDateChecker, executionFilter, verboseExplanations),
          keepGoing, actualJobs, fileCache, request.getBuildOptions().progressReportInterval);
    }
  }

  private void configureResourceManager(BuildRequest request) {
    ResourceManager resourceMgr = ResourceManager.instance();
    ExecutionOptions options = request.getOptions(ExecutionOptions.class);
    if (options.availableResources != null) {
      resourceMgr.setAvailableResources(options.availableResources);
      resourceMgr.setRamUtilizationPercentage(100);
    } else {
      resourceMgr.setAvailableResources(LocalHostCapacity.getLocalHostCapacity());
      resourceMgr.setRamUtilizationPercentage(options.ramUtilizationPercentage);
      if (options.useResourceAutoSense) {
        getReporter().warn(null, "Not using resource autosense due to known responsiveness issues");
      }
      ResourceManager.instance().setAutoSensing(/*autosense=*/false);
    }
  }

  /**
   * Writes the cache files to disk, reporting any errors that occurred during
   * writing.
   */
  private void saveCaches(ActionCache actionCache, MetadataCache metadataCache) {
    long actionCacheSizeInBytes = 0;
    long metadataCacheSizeInBytes = 0;
    long actionCacheSaveTime;
    long metadataCacheSaveTime;

    long startTime = BlazeClock.nanoTime();
    try {
      LOG.info("saving action cache...");
      actionCacheSizeInBytes = actionCache.save();
      LOG.info("action cache saved");
    } catch (IOException e) {
      getReporter().error(null, "I/O error while writing action log: " + e.getMessage());
    } finally {
      long stopTime = BlazeClock.nanoTime();
      actionCacheSaveTime =
          TimeUnit.MILLISECONDS.convert(stopTime - startTime, TimeUnit.NANOSECONDS);
      Profiler.instance().logSimpleTask(startTime, stopTime,
                                        ProfilerTask.INFO, "Saving action cache");
    }

    startTime = BlazeClock.nanoTime();
    try {
      LOG.info("saving metadata cache...");
      metadataCacheSizeInBytes = metadataCache.save();
      LOG.info("metadata cache saved");
    } catch (IOException e) {
      getReporter().error(null, "I/O error while writing metadata cache: " + e.getMessage());
    } finally {
      long stopTime = BlazeClock.nanoTime();
      metadataCacheSaveTime =
          TimeUnit.MILLISECONDS.convert(stopTime - startTime, TimeUnit.NANOSECONDS);
      Profiler.instance().logSimpleTask(startTime, stopTime,
                                        ProfilerTask.INFO, "Saving metadata cache");
    }

    runtime.getEventBus().post(new CachesSavedEvent(
        metadataCacheSaveTime, metadataCacheSizeInBytes,
        actionCacheSaveTime, actionCacheSizeInBytes));
  }

  private ActionInputFileCache createBuildSingleFileCache(Path execRoot) {
    String cwd = execRoot.getPathString();
    FileSystem fs = runtime.getDirectories().getFileSystem();

    ActionInputFileCache cache = null;
    for (BlazeModule module : runtime.getBlazeModules()) {
      ActionInputFileCache pluggable = module.createActionInputCache(cwd, fs);
      if (pluggable != null) {
        Preconditions.checkState(cache == null);
        cache = pluggable;
      }
    }

    if (cache == null) {
      cache = new SingleBuildFileCache(cwd, fs);
    }
    return cache;
  }

  private boolean enableIncrementalGraphPruning() {
    String envVar = runtime.getClientEnv().get("BLAZE_INTERNAL_USE_INCREMENTAL_GRAPH_PRUNING");
    return envVar == null || !envVar.equals("0");
  }

  private Reporter getReporter() {
    return runtime.getReporter();
  }

  private EventBus getEventBus() {
    return runtime.getEventBus();
  }

  private BuildView getView() {
    return runtime.getView();
  }

  private Path getWorkspace() {
    return runtime.getWorkspace();
  }

  private Path getExecRoot() {
    return runtime.getExecRoot();
  }
}
