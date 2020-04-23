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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicate;
import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.ConvenienceSymlink;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions.ConvenienceSymlinksMode;
import com.google.devtools.build.lib.buildtool.buildevent.ConvenienceSymlinksIdentifiedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecRootPreparedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.BlazeExecutor;
import com.google.devtools.build.lib.exec.CheckUpToDateFilter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ExecutorLifecycleListener;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.RemoteLocalFallbackRegistry;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.exec.SpawnStrategyResolver;
import com.google.devtools.build.lib.exec.SymlinkTreeStrategy;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.Builder;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * This class manages the execution phase. The entry point is {@link #executeBuild}.
 *
 * <p>This is only intended for use by {@link BuildTool}.
 *
 * <p>This class contains an ActionCache, and refers to the Blaze Runtime's BuildView and
 * PackageCache.
 *
 * @see BuildTool
 * @see com.google.devtools.build.lib.analysis.BuildView
 */
public class ExecutionTool {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final CommandEnvironment env;
  private final BlazeRuntime runtime;
  private final BuildRequest request;
  private BlazeExecutor executor;
  private final ActionInputPrefetcher prefetcher;
  private final ImmutableSet<ExecutorLifecycleListener> executorLifecycleListeners;
  private final SpawnActionContextMaps spawnActionContextMaps;

  ExecutionTool(CommandEnvironment env, BuildRequest request) throws ExecutorInitException {
    this.env = env;
    this.runtime = env.getRuntime();
    this.request = request;

    try {
      env.getExecRoot().createDirectoryAndParents();
    } catch (IOException e) {
      throw new ExecutorInitException("Execroot creation failed", e);
    }

    ExecutorBuilder executorBuilder = new ExecutorBuilder();
    ModuleActionContextRegistry.Builder actionContextRegistryBuilder =
        executorBuilder.asModuleActionContextRegistryBuilder(ModuleActionContextRegistry.builder());
    SpawnStrategyRegistry.Builder spawnStrategyRegistryBuilder =
        executorBuilder.asSpawnStrategyRegistryBuilder(SpawnStrategyRegistry.builder());
    actionContextRegistryBuilder.register(SpawnStrategyResolver.class, new SpawnStrategyResolver());
    executorBuilder.addStrategyByContext(SpawnStrategyResolver.class, "");

    for (BlazeModule module : runtime.getBlazeModules()) {
      try (SilentCloseable ignored = Profiler.instance().profile(module + ".executorInit")) {
        module.executorInit(env, request, executorBuilder);
      }

      try (SilentCloseable ignored =
          Profiler.instance().profile(module + ".registerActionContexts")) {
        module.registerActionContexts(actionContextRegistryBuilder, env, request);
      }

      try (SilentCloseable ignored =
          Profiler.instance().profile(module + ".registerSpawnStrategies")) {
        module.registerSpawnStrategies(spawnStrategyRegistryBuilder, env);
      }
    }
    actionContextRegistryBuilder.register(
        SymlinkTreeActionContext.class,
        new SymlinkTreeStrategy(env.getOutputService(), env.getBlazeWorkspace().getBinTools()));
    // TODO(philwo) - the ExecutionTool should not add arbitrary dependencies on its own, instead
    // these dependencies should be added to the ActionContextConsumer of the module that actually
    // depends on them.
    actionContextRegistryBuilder
        .restrictTo(WorkspaceStatusAction.Context.class, "")
        .restrictTo(SymlinkTreeActionContext.class, "");

    this.prefetcher = executorBuilder.getActionInputPrefetcher();
    this.executorLifecycleListeners = executorBuilder.getExecutorLifecycleListeners();

    // There are many different SpawnActions, and we want to control the action context they use
    // independently from each other, for example, to run genrules locally and Java compile action
    // in prod. Thus, for SpawnActions, we decide the action context to use not only based on the
    // context class, but also the mnemonic of the action.
    ExecutionOptions options = request.getOptions(ExecutionOptions.class);
    // TODO(jmmv): This should live in some testing-related Blaze module, not here.
    actionContextRegistryBuilder.restrictTo(TestActionContext.class, options.testStrategy);

    SpawnStrategyRegistry spawnStrategyRegistry = spawnStrategyRegistryBuilder.build();
    actionContextRegistryBuilder.register(SpawnStrategyRegistry.class, spawnStrategyRegistry);
    actionContextRegistryBuilder.register(DynamicStrategyRegistry.class, spawnStrategyRegistry);
    actionContextRegistryBuilder.register(RemoteLocalFallbackRegistry.class, spawnStrategyRegistry);

    executorBuilder.addActionContext(SpawnStrategyRegistry.class, spawnStrategyRegistry);
    executorBuilder.addStrategyByContext(SpawnStrategyRegistry.class, "");

    ModuleActionContextRegistry moduleActionContextRegistry = actionContextRegistryBuilder.build();
    executorBuilder.addActionContext(
        ModuleActionContextRegistry.class, moduleActionContextRegistry);
    executorBuilder.addStrategyByContext(ModuleActionContextRegistry.class, "");

    spawnActionContextMaps = executorBuilder.getSpawnActionContextMaps();

    if (options.availableResources != null && options.removeLocalResources) {
      throw new ExecutorInitException(
          "--local_resources is deprecated. Please use "
              + "--local_ram_resources and/or --local_cpu_resources");
    }
  }

  Executor getExecutor() throws ExecutorInitException {
    if (executor == null) {
      executor = createExecutor();
      for (ExecutorLifecycleListener executorLifecycleListener : executorLifecycleListeners) {
        executorLifecycleListener.executorCreated();
      }
    }
    return executor;
  }

  /** Creates an executor for the current set of blaze runtime, execution options, and request. */
  private BlazeExecutor createExecutor() throws ExecutorInitException {
    return new BlazeExecutor(
        runtime.getFileSystem(),
        env.getExecRoot(),
        getReporter(),
        runtime.getClock(),
        request,
        spawnActionContextMaps);
  }

  void init() throws ExecutorInitException {
    getExecutor();
  }

  void shutdown() {
    for (ExecutorLifecycleListener executorLifecycleListener : executorLifecycleListeners) {
      executorLifecycleListener.executionPhaseEnding();
    }
  }

  TestActionContext getTestActionContext() {
    return spawnActionContextMaps.getContext(TestActionContext.class);
  }

  /**
   * Performs the execution phase (phase 3) of the build, in which the Builder is applied to the
   * action graph to bring the targets up to date. (This function will return prior to
   * execution-proper if --nobuild was specified.)
   *
   * @param buildId UUID of the build id
   * @param analysisResult the analysis phase output
   * @param buildResult the mutable build result
   * @param packageRoots package roots collected from loading phase and BuildConfigurationCollection
   *     creation. May be empty if {@link
   *     SkyframeExecutor#getForcedSingleSourceRootIfNoExecrootSymlinkCreation} is false.
   */
  void executeBuild(
      UUID buildId,
      AnalysisResult analysisResult,
      BuildResult buildResult,
      PackageRoots packageRoots,
      TopLevelArtifactContext topLevelArtifactContext)
      throws BuildFailedException, InterruptedException, TestExecException, AbruptExitException {
    Stopwatch timer = Stopwatch.createStarted();
    prepare(packageRoots, analysisResult.getNonSymlinkedDirectoriesUnderExecRoot());

    ActionGraph actionGraph = analysisResult.getActionGraph();

    OutputService outputService = env.getOutputService();
    ModifiedFileSet modifiedOutputFiles = ModifiedFileSet.EVERYTHING_MODIFIED;
    if (outputService != null) {
      try (SilentCloseable c = Profiler.instance().profile("outputService.startBuild")) {
        modifiedOutputFiles =
            outputService.startBuild(
                env.getReporter(), buildId, request.getBuildOptions().finalizeActions);
      }
    } else {
      // TODO(bazel-team): this could be just another OutputService
      try (SilentCloseable c = Profiler.instance().profile("startLocalOutputBuild")) {
        startLocalOutputBuild();
      }
    }

    if (outputService == null || !outputService.actionFileSystemType().inMemoryFileSystem()) {
      // Must be created after the output path is created above.
      createActionLogDirectory();
    }

    handleConvenienceSymlinks(analysisResult);

    ActionCache actionCache = getActionCache();
    actionCache.resetStatistics();
    SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
    Builder builder;
    try (SilentCloseable c = Profiler.instance().profile("createBuilder")) {
      builder = createBuilder(request, actionCache, skyframeExecutor, modifiedOutputFiles);
    }

    //
    // Execution proper.  All statements below are logically nested in
    // begin/end pairs.  No early returns or exceptions please!
    //

    Collection<ConfiguredTarget> configuredTargets = buildResult.getActualTargets();
    try (SilentCloseable c = Profiler.instance().profile("ExecutionStartingEvent")) {
      env.getEventBus().post(new ExecutionStartingEvent(configuredTargets));
    }

    getReporter().handle(Event.progress("Building..."));

    // Conditionally record dependency-checker log:
    ExplanationHandler explanationHandler =
        installExplanationHandler(
            request.getBuildOptions().explanationPath, request.getOptionsDescription());

    Set<ConfiguredTargetKey> builtTargets = new HashSet<>();
    Set<AspectKey> builtAspects = new HashSet<>();

    if (request.isRunningInEmacs()) {
      // The syntax of this message is tightly constrained by lisp/progmodes/compile.el in emacs
      request
          .getOutErr()
          .printErrLn(runtime.getProductName() + ": Entering directory `" + getExecRoot() + "/'");
    }

    Throwable catastrophe = null;
    boolean buildCompleted = false;
    try {
      if (request.getViewOptions().discardAnalysisCache
          || !skyframeExecutor.tracksStateForIncrementality()) {
        // Free memory by removing cache entries that aren't going to be needed.
        try (SilentCloseable c = Profiler.instance().profile("clearAnalysisCache")) {
          env.getSkyframeBuildView()
              .clearAnalysisCache(
                  analysisResult.getTargetsToBuild(), analysisResult.getAspectsMap().keySet());
        }
      }

      for (ExecutorLifecycleListener executorLifecycleListener : executorLifecycleListeners) {
        try (SilentCloseable c =
            Profiler.instance().profile(executorLifecycleListener + ".executionPhaseStarting")) {
          executorLifecycleListener.executionPhaseStarting(
              actionGraph,
              // If this supplier is ever consumed by more than one ActionContextProvider, it can be
              // pulled out of the loop and made a memoizing supplier.
              () -> TopLevelArtifactHelper.makeTopLevelArtifactsToOwnerLabels(analysisResult));
        }
      }
      skyframeExecutor.drainChangedFiles();

      try (SilentCloseable c = Profiler.instance().profile("configureResourceManager")) {
        configureResourceManager(env.getLocalResourceManager(), request);
      }

      Profiler.instance().markPhase(ProfilePhase.EXECUTE);
      boolean shouldTrustRemoteArtifacts =
          env.getOutputService() == null
              ? false
              : env.getOutputService().shouldTrustRemoteArtifacts();
      builder.buildArtifacts(
          env.getReporter(),
          analysisResult.getTopLevelArtifactsToOwnerLabels().getArtifacts(),
          analysisResult.getParallelTests(),
          analysisResult.getExclusiveTests(),
          analysisResult.getTargetsToBuild(),
          analysisResult.getTargetsToSkip(),
          analysisResult.getAspectsMap().keySet(),
          executor,
          builtTargets,
          builtAspects,
          request,
          env.getBlazeWorkspace().getLastExecutionTimeRange(),
          topLevelArtifactContext,
          shouldTrustRemoteArtifacts);
      buildCompleted = true;
    } catch (BuildFailedException | TestExecException e) {
      buildCompleted = true;
      throw e;
    } catch (Error | RuntimeException e) {
      catastrophe = e;
    } finally {
      // These may flush logs, which may help if there is a catastrophic failure.
      for (ExecutorLifecycleListener executorLifecycleListener : executorLifecycleListeners) {
        executorLifecycleListener.executionPhaseEnding();
      }

      // Handlers process these events and others (e.g. CommandCompleteEvent), even in the event of
      // a catastrophic failure. Posting these is consistent with other behavior.
      env.getEventBus().post(skyframeExecutor.createExecutionFinishedEvent());

      env.getEventBus()
          .post(new ExecutionPhaseCompleteEvent(timer.stop().elapsed(TimeUnit.MILLISECONDS)));

      if (catastrophe != null) {
        Throwables.throwIfUnchecked(catastrophe);
      }
      // NOTE: No finalization activities below will run in the event of a catastrophic error!

      env.recordLastExecutionTime();

      if (request.isRunningInEmacs()) {
        request
            .getOutErr()
            .printErrLn(runtime.getProductName() + ": Leaving directory `" + getExecRoot() + "/'");
      }
      if (buildCompleted) {
        getReporter().handle(Event.progress("Building complete."));
      }

      if (buildCompleted) {
        saveActionCache(actionCache);
      }

      try (SilentCloseable c = Profiler.instance().profile("Show results")) {
        buildResult.setSuccessfulTargets(
            determineSuccessfulTargets(configuredTargets, builtTargets));
        buildResult.setSuccessfulAspects(
            determineSuccessfulAspects(analysisResult.getAspectsMap().keySet(), builtAspects));
        buildResult.setSkippedTargets(analysisResult.getTargetsToSkip());
        BuildResultPrinter buildResultPrinter = new BuildResultPrinter(env);
        buildResultPrinter.showBuildResult(
            request,
            buildResult,
            configuredTargets,
            analysisResult.getTargetsToSkip(),
            analysisResult.getAspectsMap());
      }

      try (SilentCloseable c = Profiler.instance().profile("Show artifacts")) {
        if (request.getBuildOptions().showArtifacts) {
          BuildResultPrinter buildResultPrinter = new BuildResultPrinter(env);
          buildResultPrinter.showArtifacts(
              request, configuredTargets, analysisResult.getAspectsMap().values());
        }
      }

      if (explanationHandler != null) {
        uninstallExplanationHandler(explanationHandler);
        try {
          explanationHandler.close();
        } catch (IOException _ignored) {
          // Ignored
        }
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

  private void prepare(
      PackageRoots packageRoots, ImmutableSortedSet<String> nonSymlinkedDirectoriesUnderExecRoot)
      throws AbruptExitException, InterruptedException {
    Optional<ImmutableMap<PackageIdentifier, Root>> packageRootMap =
        packageRoots.getPackageRootsMap();
    if (packageRootMap.isPresent()) {
      // Prepare for build.
      Profiler.instance().markPhase(ProfilePhase.PREPARE);

      // Plant the symlink forest.
      try (SilentCloseable c = Profiler.instance().profile("plantSymlinkForest")) {
        SymlinkForest symlinkForest =
            new SymlinkForest(
                packageRootMap.get(),
                getExecRoot(),
                runtime.getProductName(),
                nonSymlinkedDirectoriesUnderExecRoot,
                request.getOptions(StarlarkSemanticsOptions.class)
                    .experimentalSiblingRepositoryLayout);
        symlinkForest.plantSymlinkForest();
      } catch (IOException e) {
        throw new ExecutorInitException("Source forest creation failed", e);
      }
    }
    env.getEventBus().post(new ExecRootPreparedEvent(packageRootMap));
  }

  private void createActionLogDirectory() throws ExecutorInitException {
    Path directory = env.getActionTempsDirectory();
    if (directory.exists()) {
      try {
        directory.deleteTree();
      } catch (IOException e) {
        throw new ExecutorInitException("Couldn't delete action output directory", e);
      }
    }

    try {
      directory.createDirectoryAndParents();
    } catch (IOException e) {
      throw new ExecutorInitException("Couldn't create action output directory", e);
    }

    try {
      env.getPersistentActionOutsDirectory().createDirectoryAndParents();
    } catch (IOException e) {
      throw new ExecutorInitException("Couldn't create persistent action output directory", e);
    }
  }

  /**
   * Obtains the {@link BuildConfiguration} for a given {@link BuildOptions} for the purpose of
   * symlink creation.
   *
   * <p>In the event of a {@link InvalidConfigurationException}, a warning is emitted and null is
   * returned.
   */
  @Nullable
  private static BuildConfiguration getConfiguration(
      SkyframeExecutor executor, Reporter reporter, BuildOptions options) {
    try {
      return executor.getConfiguration(reporter, options, /*keepGoing=*/ false);
    } catch (InvalidConfigurationException e) {
      reporter.handle(
          Event.warn(
              "Couldn't get configuration for convenience symlink creation: " + e.getMessage()));
      return null;
    }
  }

  /**
   * Handles what action to perform on the convenience symlinks. If the the mode is {@link
   * ConvenienceSymlinksMode.IGNORE}, then skip any creating or cleaning of convenience symlinks.
   * Otherwise, manage the convenience symlinks and then post a {@link
   * ConvenienceSymlinksIdentifiedEvent} build event.
   */
  private void handleConvenienceSymlinks(AnalysisResult analysisResult) {
    ImmutableList<ConvenienceSymlink> convenienceSymlinks = ImmutableList.of();
    if (request.getBuildOptions().experimentalConvenienceSymlinks
        != ConvenienceSymlinksMode.IGNORE) {
      convenienceSymlinks = createConvenienceSymlinks(request.getBuildOptions(), analysisResult);
    }
    if (request.getBuildOptions().experimentalConvenienceSymlinksBepEvent) {
      env.getEventBus().post(new ConvenienceSymlinksIdentifiedEvent(convenienceSymlinks));
    }
  }

  /**
   * Creates convenience symlinks based on the target configurations.
   *
   * <p>Exactly what target configurations we consider depends on the value of {@code
   * --use_top_level_targets_for_symlinks}. If this flag is false, we use the top-level target
   * configuration as represented by the command line prior to processing any target. If the flag is
   * true, we instead use the configurations OF the top-level targets -- meaning that we account for
   * the effects of any rule transitions these targets may have.
   *
   * <p>For each type of convenience symlink, if all the considered configurations agree on what
   * path the symlink should point to, it gets created; otherwise, the symlink is not created, and
   * in fact gets removed if it was already present from a previous invocation.
   */
  private ImmutableList<ConvenienceSymlink> createConvenienceSymlinks(
      BuildRequestOptions buildRequestOptions, AnalysisResult analysisResult) {
    SkyframeExecutor executor = env.getSkyframeExecutor();
    Reporter reporter = env.getReporter();

    // Gather configurations to consider.
    Set<BuildConfiguration> targetConfigurations =
        buildRequestOptions.useTopLevelTargetsForSymlinks()
            ? analysisResult.getTargetsToBuild().stream()
                .map(ConfiguredTarget::getConfigurationKey)
                .filter(configuration -> configuration != null)
                .distinct()
                .map((key) -> executor.getConfiguration(reporter, key))
                .collect(toImmutableSet())
            : ImmutableSet.copyOf(
                analysisResult.getConfigurationCollection().getTargetConfigurations());

    String productName = runtime.getProductName();
    try (SilentCloseable c =
        Profiler.instance().profile("OutputDirectoryLinksUtils.createOutputDirectoryLinks")) {
      return OutputDirectoryLinksUtils.createOutputDirectoryLinks(
          runtime.getRuleClassProvider().getSymlinkDefinitions(),
          buildRequestOptions,
          env.getWorkspaceName(),
          env.getWorkspace(),
          env.getDirectories(),
          getReporter(),
          targetConfigurations,
          options -> getConfiguration(executor, reporter, options),
          productName);
    }
  }

  /** Prepare for a local output build. */
  private void startLocalOutputBuild() throws ExecutorInitException {
    try (SilentCloseable c = Profiler.instance().profile("Starting local output build")) {
      Path outputPath = env.getDirectories().getOutputPath(env.getWorkspaceName());
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
   * If a path is supplied, creates and installs an ExplanationHandler. Returns an instance on
   * success. Reports an error and returns null otherwise.
   */
  private ExplanationHandler installExplanationHandler(
      PathFragment explanationPath, String allOptions) {
    if (explanationPath == null) {
      return null;
    }
    ExplanationHandler handler;
    try {
      handler =
          new ExplanationHandler(
              getWorkspace().getRelative(explanationPath).getOutputStream(), allOptions);
    } catch (IOException e) {
      getReporter()
          .handle(
              Event.warn(
                  String.format(
                      "Cannot write explanation of rebuilds to file '%s': %s",
                      explanationPath, e.getMessage())));
      return null;
    }
    getReporter()
        .handle(Event.info("Writing explanation of rebuilds to '" + explanationPath + "'"));
    getReporter().addHandler(handler);
    return handler;
  }

  /** Uninstalls the specified ExplanationHandler (if any) and closes the log file. */
  private void uninstallExplanationHandler(ExplanationHandler handler) {
    if (handler != null) {
      getReporter().removeHandler(handler);
      handler.log.close();
    }
  }

  /**
   * An ErrorEventListener implementation that records DEPCHECKER events into a log file, iff the
   * --explain flag is specified during a build.
   */
  private static class ExplanationHandler implements EventHandler, AutoCloseable {
    private final PrintWriter log;

    private ExplanationHandler(OutputStream log, String optionsDescription) {
      this.log = new PrintWriter(new OutputStreamWriter(log, StandardCharsets.UTF_8));
      this.log.println("Build options: " + optionsDescription);
    }

    @Override
    public void close() throws IOException {
      this.log.close();
    }

    @Override
    public void handle(Event event) {
      if (event.getKind() == EventKind.DEPCHECKER) {
        log.println(event.getMessage());
      }
    }
  }

  /**
   * Computes the result of the build. Sets the list of successful (up-to-date) targets in the
   * request object.
   *
   * @param configuredTargets The configured targets whose artifacts are to be built.
   */
  private Collection<ConfiguredTarget> determineSuccessfulTargets(
      Collection<ConfiguredTarget> configuredTargets, Set<ConfiguredTargetKey> builtTargets) {
    // Maintain the ordering by copying builtTargets into a LinkedHashSet in the same iteration
    // order as configuredTargets.
    Collection<ConfiguredTarget> successfulTargets = new LinkedHashSet<>();
    for (ConfiguredTarget target : configuredTargets) {
      if (builtTargets.contains(ConfiguredTargetKey.inTargetConfig(target))) {
        successfulTargets.add(target);
      }
    }
    return successfulTargets;
  }

  private static ImmutableSet<AspectKey> determineSuccessfulAspects(
      ImmutableSet<AspectKey> aspects, Set<AspectKey> builtAspects) {
    // Maintain the ordering.
    return aspects.stream().filter(builtAspects::contains).collect(ImmutableSet.toImmutableSet());
  }

  /** Get action cache if present or reload it from the on-disk cache. */
  private ActionCache getActionCache() throws LocalEnvironmentException {
    try {
      return env.getPersistentActionCache();
    } catch (IOException e) {
      // TODO(bazel-team): (2010) Ideally we should just remove all cache data and reinitialize
      // caches.
      LoggingUtil.logToRemote(
          Level.WARNING, "Failed to initialize action cache: " + e.getMessage(), e);
      throw new LocalEnvironmentException(
          "couldn't create action cache: "
              + e.getMessage()
              + ". If error persists, use 'bazel clean'");
    }
  }

  private Builder createBuilder(
      BuildRequest request,
      ActionCache actionCache,
      SkyframeExecutor skyframeExecutor,
      ModifiedFileSet modifiedOutputFiles) {
    BuildRequestOptions options = request.getBuildOptions();

    skyframeExecutor.setActionOutputRoot(
        env.getActionTempsDirectory(), env.getPersistentActionOutsDirectory());

    Predicate<Action> executionFilter =
        CheckUpToDateFilter.fromOptions(request.getOptions(ExecutionOptions.class));
    ArtifactFactory artifactFactory = env.getSkyframeBuildView().getArtifactFactory();
    return new SkyframeBuilder(
        skyframeExecutor,
        env.getLocalResourceManager(),
        new ActionCacheChecker(
            actionCache,
            artifactFactory,
            skyframeExecutor.getActionKeyContext(),
            executionFilter,
            ActionCacheChecker.CacheConfig.builder()
                .setEnabled(options.useActionCache)
                .setVerboseExplanations(options.verboseExplanations)
                .build()),
        env.getTopDownActionCache(),
        request.getPackageOptions().checkOutputFiles
            ? modifiedOutputFiles
            : ModifiedFileSet.NOTHING_MODIFIED,
        env.getFileCache(),
        prefetcher);
  }

  @VisibleForTesting
  public static void configureResourceManager(ResourceManager resourceMgr, BuildRequest request) {
    ExecutionOptions options = request.getOptions(ExecutionOptions.class);
    ResourceSet resources;
    if (options.availableResources != null && !options.removeLocalResources) {
      logger.atWarning().log(
          "--local_resources will be deprecated. Please use --local_ram_resources "
              + "and/or --local_cpu_resources.");
      resources = options.availableResources;
    } else {
      resources =
          ResourceSet.createWithRamCpu(options.localRamResources, options.localCpuResources);
    }
    resourceMgr.setUseLocalMemoryEstimate(options.localMemoryEstimate);

    resourceMgr.setAvailableResources(
        ResourceSet.create(
            resources.getMemoryMb(),
            resources.getCpuUsage(),
            request.getExecutionOptions().usingLocalTestJobs()
                ? request.getExecutionOptions().localTestJobs
                : Integer.MAX_VALUE));
  }

  /**
   * Writes the action cache files to disk, reporting any errors that occurred during writing and
   * capturing statistics.
   */
  private void saveActionCache(ActionCache actionCache) {
    ActionCacheStatistics.Builder builder = ActionCacheStatistics.newBuilder();
    actionCache.mergeIntoActionCacheStatistics(builder);

    AutoProfiler p =
        GoogleAutoProfilerUtils.profiledAndLogged("Saving action cache", ProfilerTask.INFO);
    try {
      builder.setSizeInBytes(actionCache.save());
    } catch (IOException e) {
      builder.setSizeInBytes(0);
      getReporter().handle(Event.error("I/O error while writing action log: " + e.getMessage()));
    } finally {
      builder.setSaveTimeInMs(
          TimeUnit.MILLISECONDS.convert(p.completeAndGetElapsedTimeNanos(), TimeUnit.NANOSECONDS));
    }

    env.getEventBus().post(builder.build());
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
