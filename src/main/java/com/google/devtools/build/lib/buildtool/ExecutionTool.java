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
package com.google.devtools.build.lib.buildtool;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.NANOSECONDS;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Stopwatch;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorBuilder;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.CheckUpToDateFilter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.exec.SymlinkTreeStrategy;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContextImpl;
import com.google.devtools.build.lib.rules.test.TestActionContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.Builder;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class manages the execution phase. The entry point is {@link #executeBuild}.
 *
 * <p>This is only intended for use by {@link BuildTool}.
 *
 * <p>This class contains an ActionCache, and refers to the BlazeRuntime's BuildView and
 * PackageCache.
 *
 * @see BuildTool
 * @see com.google.devtools.build.lib.analysis.BuildView
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

  static final Logger log = Logger.getLogger(ExecutionTool.class.getName());

  private final CommandEnvironment env;
  private final BlazeRuntime runtime;
  private final BuildRequest request;
  private BlazeExecutor executor;
  private ActionInputFileCache fileCache;
  private final ImmutableList<ActionContextProvider> actionContextProviders;

  private Map<String, SpawnActionContext> spawnStrategyMap =
      new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
  private List<ActionContext> strategies = new ArrayList<>();

  ExecutionTool(CommandEnvironment env, BuildRequest request) throws ExecutorInitException {
    this.env = env;
    this.runtime = env.getRuntime();
    this.request = request;


    // Create tools before getting the strategies from the modules as some of them need tools to
    // determine whether the host actually supports certain strategies (e.g. sandboxing).
    createToolsSymlinks();

    ExecutorBuilder builder = new ExecutorBuilder();
    for (BlazeModule module : runtime.getBlazeModules()) {
      module.executorInit(env, request, builder);
    }
    builder.addActionContextProvider(
        new FilesetActionContextImpl.Provider(env.getReporter(), env.getWorkspaceName()));
    builder.addActionContext(new SymlinkTreeStrategy(
                env.getOutputService(), env.getBlazeWorkspace().getBinTools()));
    // TODO(philwo) - the ExecutionTool should not add arbitrary dependencies on its own, instead
    // these dependencies should be added to the ActionContextConsumer of the module that actually
    // depends on them.
    builder.addActionContextConsumer(
        new ActionContextConsumer() {
          @Override
          public ImmutableMap<String, String> getSpawnActionContexts() {
            return ImmutableMap.of();
          }

          @Override
          public Multimap<Class<? extends ActionContext>, String> getActionContexts() {
            return ImmutableMultimap.<Class<? extends ActionContext>, String>builder()
                .put(FilesetActionContext.class, "")
                .put(WorkspaceStatusAction.Context.class, "")
                .put(SymlinkTreeActionContext.class, "")
                .build();
          }
        });

    this.actionContextProviders = builder.getActionContextProviders();
    StrategyConverter strategyConverter = new StrategyConverter(actionContextProviders);

    for (ActionContextConsumer consumer : builder.getActionContextConsumers()) {
      // There are many different SpawnActions, and we want to control the action context they use
      // independently from each other, for example, to run genrules locally and Java compile action
      // in prod. Thus, for SpawnActions, we decide the action context to use not only based on the
      // context class, but also the mnemonic of the action.
      for (Map.Entry<String, String> entry : consumer.getSpawnActionContexts().entrySet()) {
        SpawnActionContext context =
            strategyConverter.getStrategy(SpawnActionContext.class, entry.getValue());
        if (context == null) {
          throw makeExceptionForInvalidStrategyValue(
              entry.getValue(),
              "spawn",
              strategyConverter.getValidValues(SpawnActionContext.class));
        }
        spawnStrategyMap.put(entry.getKey(), context);
      }

      for (Map.Entry<Class<? extends ActionContext>, String> entry :
          consumer.getActionContexts().entries()) {
        ActionContext context = strategyConverter.getStrategy(entry.getKey(), entry.getValue());
        if (context == null) {
          throw makeExceptionForInvalidStrategyValue(
              entry.getValue(),
              strategyConverter.getUserFriendlyName(entry.getKey()),
              strategyConverter.getValidValues(entry.getKey()));
        }
        strategies.add(context);
      }
    }

    String testStrategyValue = request.getOptions(ExecutionOptions.class).testStrategy;
    ActionContext context = strategyConverter.getStrategy(TestActionContext.class,
        testStrategyValue);
    if (context == null) {
      throw makeExceptionForInvalidStrategyValue(testStrategyValue, "test",
          strategyConverter.getValidValues(TestActionContext.class));
    }
    strategies.add(context);
  }

  private static ExecutorInitException makeExceptionForInvalidStrategyValue(String value,
      String strategy, String validValues) {
    return new ExecutorInitException(String.format(
        "'%s' is an invalid value for %s strategy. Valid values are: %s", value, strategy,
        validValues), ExitCode.COMMAND_LINE_ERROR);
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
        env.getExecRoot(),
        getReporter(),
        env.getEventBus(),
        runtime.getClock(),
        request,
        request.getOptions(ExecutionOptions.class).verboseFailures,
        request.getOptions(ExecutionOptions.class).showSubcommands,
        strategies,
        spawnStrategyMap,
        actionContextProviders);
  }

  void init() throws ExecutorInitException {
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
   * @param buildId UUID of the build id
   * @param analysisResult the analysis phase output
   * @param buildResult the mutable build result
   * @param packageRoots package roots collected from loading phase and BuildConfigurationCollection
   * creation
   */
  void executeBuild(UUID buildId, AnalysisResult analysisResult,
      BuildResult buildResult,
      BuildConfigurationCollection configurations,
      ImmutableMap<PackageIdentifier, Path> packageRoots,
      TopLevelArtifactContext topLevelArtifactContext)
      throws BuildFailedException, InterruptedException, TestExecException, AbruptExitException {
    Stopwatch timer = Stopwatch.createStarted();
    prepare(packageRoots, analysisResult.getWorkspaceName());

    ActionGraph actionGraph = analysisResult.getActionGraph();

    // Get top-level artifacts.
    ImmutableSet<Artifact> additionalArtifacts = analysisResult.getAdditionalArtifactsToBuild();

    OutputService outputService = env.getOutputService();
    ModifiedFileSet modifiedOutputFiles = ModifiedFileSet.EVERYTHING_MODIFIED;
    if (outputService != null) {
      modifiedOutputFiles = outputService.startBuild(buildId,
              request.getBuildOptions().finalizeActions);
    } else {
      // TODO(bazel-team): this could be just another OutputService
      startLocalOutputBuild(analysisResult.getWorkspaceName());
    }

    List<BuildConfiguration> targetConfigurations = configurations.getTargetConfigurations();
    BuildConfiguration targetConfiguration = targetConfigurations.size() == 1
        ? targetConfigurations.get(0) : null;
    if (targetConfigurations.size() == 1) {
      String productName = runtime.getProductName();
      String dirName = env.getWorkspaceName();
      String workspaceName = analysisResult.getWorkspaceName();
      OutputDirectoryLinksUtils.createOutputDirectoryLinks(
          dirName, env.getWorkspace(), env.getDirectories().getExecRoot(workspaceName),
          env.getDirectories().getOutputPath(workspaceName), getReporter(), targetConfiguration,
          request.getBuildOptions().getSymlinkPrefix(productName), productName);
    }

    ActionCache actionCache = getActionCache();
    SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
    Builder builder = createBuilder(
        request, actionCache, skyframeExecutor, modifiedOutputFiles,
        analysisResult.getWorkspaceName());

    //
    // Execution proper.  All statements below are logically nested in
    // begin/end pairs.  No early returns or exceptions please!
    //

    Collection<ConfiguredTarget> configuredTargets = buildResult.getActualTargets();
    env.getEventBus().post(new ExecutionStartingEvent(configuredTargets));

    getReporter().handle(Event.progress("Building..."));

    // Conditionally record dependency-checker log:
    ExplanationHandler explanationHandler =
        installExplanationHandler(request.getBuildOptions().explanationPath,
                                  request.getOptionsDescription());

    Set<ConfiguredTarget> builtTargets = new HashSet<>();
    Collection<AspectValue> aspects = analysisResult.getAspects();

    Iterable<Artifact> allArtifactsForProviders =
        Iterables.concat(
            additionalArtifacts,
            TopLevelArtifactHelper.getAllArtifactsToBuild(
                    analysisResult.getTargetsToBuild(), analysisResult.getTopLevelContext())
                .getAllArtifacts(),
            TopLevelArtifactHelper.getAllArtifactsToBuildFromAspects(
                    aspects, analysisResult.getTopLevelContext())
                .getAllArtifacts(),
            //TODO(dslomov): Artifacts to test from aspects?
            TopLevelArtifactHelper.getAllArtifactsToTest(analysisResult.getTargetsToTest()));

    if (request.isRunningInEmacs()) {
      // The syntax of this message is tightly constrained by lisp/progmodes/compile.el in emacs
      request.getOutErr().printErrLn("blaze: Entering directory `" + getExecRoot() + "/'");
    }
    boolean buildCompleted = false;
    try {
      for (ActionContextProvider actionContextProvider : actionContextProviders) {
        actionContextProvider.executionPhaseStarting(
            fileCache,
            actionGraph,
            allArtifactsForProviders);
      }
      executor.executionPhaseStarting();
      skyframeExecutor.drainChangedFiles();

      if (request.getViewOptions().discardAnalysisCache) {
        // Free memory by removing cache entries that aren't going to be needed. Note that in
        // skyframe full, this destroys the action graph as well, so we can only do it after the
        // action graph is no longer needed.
        env.getSkyframeBuildView().clearAnalysisCache(analysisResult.getTargetsToBuild());
      }

      configureResourceManager(request);

      Profiler.instance().markPhase(ProfilePhase.EXECUTE);

      builder.buildArtifacts(
          env.getReporter(),
          additionalArtifacts,
          analysisResult.getParallelTests(),
          analysisResult.getExclusiveTests(),
          analysisResult.getTargetsToBuild(),
          analysisResult.getAspects(),
          executor,
          builtTargets,
          request.getBuildOptions().explanationPath != null,
          env.getBlazeWorkspace().getLastExecutionTimeRange(),
          topLevelArtifactContext);
      buildCompleted = true;
    } catch (BuildFailedException | TestExecException e) {
      buildCompleted = true;
      throw e;
    } finally {
      env.recordLastExecutionTime();
      if (request.isRunningInEmacs()) {
        request.getOutErr().printErrLn("blaze: Leaving directory `" + getExecRoot() + "/'");
      }
      if (buildCompleted) {
        getReporter().handle(Event.progress("Building complete."));
      }

      env.getEventBus().post(new ExecutionFinishedEvent(ImmutableMap.<String, Long> of(), 0L,
          skyframeExecutor.getOutputDirtyFilesAndClear(),
          skyframeExecutor.getModifiedFilesDuringPreviousBuildAndClear()));

      executor.executionPhaseEnding();
      for (ActionContextProvider actionContextProvider : actionContextProviders) {
        actionContextProvider.executionPhaseEnding();
      }

      Profiler.instance().markPhase(ProfilePhase.FINISH);

      if (buildCompleted) {
        saveCaches(actionCache);
      }

      try (AutoProfiler p = AutoProfiler.profiled("Show results", ProfilerTask.INFO)) {
        buildResult.setSuccessfulTargets(
            determineSuccessfulTargets(configuredTargets, builtTargets, timer));
        BuildResultPrinter buildResultPrinter = new BuildResultPrinter(env);
        buildResultPrinter.showBuildResult(
            request, buildResult, configuredTargets, analysisResult.getAspects());
      }

      try (AutoProfiler p = AutoProfiler.profiled("Show artifacts", ProfilerTask.INFO)) {
        if (request.getBuildOptions().showArtifacts) {
          BuildResultPrinter buildResultPrinter = new BuildResultPrinter(env);
          buildResultPrinter.showArtifacts(
              request, configuredTargets, analysisResult.getAspects());
        }
      }

      if (explanationHandler != null) {
        uninstallExplanationHandler(explanationHandler);
      }
      // Finalize output service last, so that if we do throw an exception, we know all the other
      // code has already run.
      if (env.getOutputService() != null) {
        boolean isBuildSuccessful =
            buildResult.getSuccessfulTargets().size() == configuredTargets.size();
        env.getOutputService().finalizeBuild(isBuildSuccessful);
      }
    }
  }

  private void prepare(ImmutableMap<PackageIdentifier, Path> packageRoots, String workspaceName)
      throws ExecutorInitException {
    // Prepare for build.
    Profiler.instance().markPhase(ProfilePhase.PREPARE);

    // Create some tools symlinks / cleanup per-build state
    createActionLogDirectory();

    // Plant the symlink forest.
    try {
      new SymlinkForest(
          packageRoots, getExecRoot(), runtime.getProductName(), workspaceName)
          .plantSymlinkForest();
    } catch (IOException e) {
      throw new ExecutorInitException("Source forest creation failed", e);
    }
  }

  private void createToolsSymlinks() throws ExecutorInitException {
    try {
      env.getBlazeWorkspace().getBinTools().setupBuildTools();
    } catch (ExecException e) {
      throw new ExecutorInitException("Tools symlink creation failed", e);
    }
  }

  private void createActionLogDirectory() throws ExecutorInitException {
    Path directory = env.getDirectories().getActionConsoleOutputDirectory();
    try {
      if (directory.exists()) {
        FileSystemUtils.deleteTree(directory);
      }
      directory.createDirectory();
    } catch (IOException e) {
      throw new ExecutorInitException("Couldn't delete action output directory", e);
    }
  }

  /**
   * Prepare for a local output build.
   */
  private void startLocalOutputBuild(String workspaceName) throws ExecutorInitException {
    try (AutoProfiler p = AutoProfiler.profiled("Starting local output build", ProfilerTask.INFO)) {
      Path outputPath = env.getDirectories().getOutputPath(workspaceName);
      Path localOutputPath = env.getDirectories().getLocalOutputPath();

      if (outputPath.isSymbolicLink()) {
        try {
          // Remove the existing symlink first.
          outputPath.delete();
          if (localOutputPath.exists()) {
            // Pre-existing local output directory. Move to outputPath.
            localOutputPath.renameTo(outputPath);
          }
        } catch (IOException e) {
          throw new ExecutorInitException("Couldn't handle local output directory symlinks", e);
        }
      }
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
      getReporter().handle(Event.warn(String.format(
          "Cannot write explanation of rebuilds to file '%s': %s",
          explanationPath, e.getMessage())));
      return null;
    }
    getReporter().handle(
        Event.info("Writing explanation of rebuilds to '" + explanationPath + "'"));
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
   * An ErrorEventListener implementation that records DEPCHECKER events into a log
   * file, iff the --explain flag is specified during a build.
   */
  private static class ExplanationHandler implements EventHandler {
    private final PrintWriter log;

    private ExplanationHandler(OutputStream log, String optionsDescription) {
      this.log = new PrintWriter(log);
      this.log.println("Build options: " + optionsDescription);
    }


    @Override
    public void handle(Event event) {
      if (event.getKind() == EventKind.DEPCHECKER) {
        log.println(event.getMessage());
      }
    }
  }

  /**
   * Computes the result of the build. Sets the list of successful (up-to-date)
   * targets in the request object.
   *
   * @param configuredTargets The configured targets whose artifacts are to be
   *                          built.
   * @param timer A timer that was started when the execution phase started.
   */
  private Collection<ConfiguredTarget> determineSuccessfulTargets(
      Collection<ConfiguredTarget> configuredTargets, Set<ConfiguredTarget> builtTargets,
      Stopwatch timer) {
    // Maintain the ordering by copying builtTargets into a LinkedHashSet in the same iteration
    // order as configuredTargets.
    Collection<ConfiguredTarget> successfulTargets = new LinkedHashSet<>();
    for (ConfiguredTarget target : configuredTargets) {
      if (builtTargets.contains(target)) {
        successfulTargets.add(target);
      }
    }
    env.getEventBus().post(
        new ExecutionPhaseCompleteEvent(timer.stop().elapsed(MILLISECONDS)));
    return successfulTargets;
  }


  private ActionCache getActionCache() throws LocalEnvironmentException {
    try {
      return env.getPersistentActionCache();
    } catch (IOException e) {
      // TODO(bazel-team): (2010) Ideally we should just remove all cache data and reinitialize
      // caches.
      LoggingUtil.logToRemote(Level.WARNING, "Failed to initialize action cache: "
          + e.getMessage(), e);
      throw new LocalEnvironmentException("couldn't create action cache: " + e.getMessage()
          + ". If error persists, use 'blaze clean'");
    }
  }

  private Builder createBuilder(BuildRequest request,
      ActionCache actionCache,
      SkyframeExecutor skyframeExecutor,
      ModifiedFileSet modifiedOutputFiles,
      String workspaceName) {
    BuildRequest.BuildRequestOptions options = request.getBuildOptions();
    boolean verboseExplanations = options.verboseExplanations;
    boolean keepGoing = request.getViewOptions().keepGoing;

    Path actionOutputRoot = env.getDirectories().getActionConsoleOutputDirectory();
    Predicate<Action> executionFilter = CheckUpToDateFilter.fromOptions(
        request.getOptions(ExecutionOptions.class));

    // jobs should have been verified in BuildRequest#validateOptions().
    Preconditions.checkState(options.jobs >= -1);
    int actualJobs = options.jobs == 0 ? 1 : options.jobs;  // Treat 0 jobs as a single task.

    // Unfortunately, the exec root cache is not shared with caches in the remote execution
    // client.
    fileCache = createBuildSingleFileCache(env.getDirectories().getExecRoot(workspaceName));
    skyframeExecutor.setActionOutputRoot(actionOutputRoot);
    ArtifactFactory artifactFactory = env.getSkyframeBuildView().getArtifactFactory();
    return new SkyframeBuilder(skyframeExecutor,
        new ActionCacheChecker(actionCache, artifactFactory, executionFilter, verboseExplanations),
        keepGoing, actualJobs,
        request.getPackageCacheOptions().checkOutputFiles
            ? modifiedOutputFiles : ModifiedFileSet.NOTHING_MODIFIED,
        options.finalizeActions, fileCache, request.getBuildOptions().progressReportInterval);
  }

  private void configureResourceManager(BuildRequest request) {
    ResourceManager resourceMgr = ResourceManager.instance();
    ExecutionOptions options = request.getOptions(ExecutionOptions.class);
    ResourceSet resources;
    if (options.availableResources != null) {
      resources = options.availableResources;
      resourceMgr.setRamUtilizationPercentage(100);
    } else {
      resources = LocalHostCapacity.getLocalHostCapacity();
      resourceMgr.setRamUtilizationPercentage(options.ramUtilizationPercentage);
    }

    resourceMgr.setAvailableResources(ResourceSet.create(
        resources.getMemoryMb(),
        resources.getCpuUsage(),
        resources.getIoUsage(),
        request.getExecutionOptions().usingLocalTestJobs()
            ? request.getExecutionOptions().localTestJobs : Integer.MAX_VALUE
    ));
  }

  /**
   * Writes the cache files to disk, reporting any errors that occurred during
   * writing.
   */
  private void saveCaches(ActionCache actionCache) {
    long actionCacheSizeInBytes = 0;
    long actionCacheSaveTimeInMs;

    AutoProfiler p = AutoProfiler.profiledAndLogged("Saving action cache", ProfilerTask.INFO, log);
    try {
      actionCacheSizeInBytes = actionCache.save();
    } catch (IOException e) {
      getReporter().handle(Event.error("I/O error while writing action log: " + e.getMessage()));
    } finally {
      actionCacheSaveTimeInMs =
          MILLISECONDS.convert(p.completeAndGetElapsedTimeNanos(), NANOSECONDS);
    }
    env.getEventBus().post(new CachesSavedEvent(
        actionCacheSaveTimeInMs, actionCacheSizeInBytes));
  }

  private ActionInputFileCache createBuildSingleFileCache(Path execRoot) {
    String cwd = execRoot.getPathString();
    FileSystem fs = env.getDirectories().getFileSystem();

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

  private Reporter getReporter() {
    return env.getReporter();
  }

  private Path getWorkspace() {
    return env.getWorkspace();
  }

  private Path getExecRoot() {
    return env.getExecRoot();
  }
}
