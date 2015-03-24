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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.collect.Table;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionContextConsumer;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.CheckUpToDateFilter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.exec.SymlinkTreeStrategy;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContextImpl;
import com.google.devtools.build.lib.rules.test.TestActionContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.skyframe.Builder;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
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

  private Map<String, SpawnActionContext> spawnStrategyMap = new HashMap<>();
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

    actionContextProviders.add(new FilesetActionContextImpl.Provider(
        runtime.getReporter(), runtime.getWorkspaceName()));

    strategies.add(new SymlinkTreeStrategy(runtime.getOutputService(), runtime.getBinTools()));

    StrategyConverter strategyConverter = new StrategyConverter(actionContextProviders);
    strategies.add(strategyConverter.getStrategy(FilesetActionContext.class, ""));
    strategies.add(strategyConverter.getStrategy(WorkspaceStatusAction.Context.class, ""));

    for (ActionContextConsumer consumer : actionContextConsumers) {
      // There are many different SpawnActions, and we want to control the action context they use
      // independently from each other, for example, to run genrules locally and Java compile action
      // in prod. Thus, for SpawnActions, we decide the action context to use not only based on the
      // context class, but also the mnemonic of the action.
      for (Map.Entry<String, String> entry : consumer.getSpawnActionContexts().entrySet()) {
        SpawnActionContext context =
            strategyConverter.getStrategy(SpawnActionContext.class, entry.getValue());
        if (context == null) {
          throw makeExceptionForInvalidStrategyValue(entry.getValue(), "spawn",
              strategyConverter.getValidValues(SpawnActionContext.class));
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
          throw makeExceptionForInvalidStrategyValue(entry.getValue(),
              strategyConverter.getUserFriendlyName(entry.getKey()),
              strategyConverter.getValidValues(entry.getKey()));
        }
      }
    }

    // If tests are to be run during build, too, we have to explicitly load the test action context.
    if (request.shouldRunTests()) {
      String testStrategyValue = request.getOptions(ExecutionOptions.class).testStrategy;
      ActionContext context = strategyConverter.getStrategy(TestActionContext.class,
          testStrategyValue);
      if (context == null) {
        throw makeExceptionForInvalidStrategyValue(testStrategyValue, "test",
            strategyConverter.getValidValues(TestActionContext.class));
      }
      strategies.add(context);
    }
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
        runtime.getDirectories().getExecRoot(),
        runtime.getDirectories().getOutputPath(),
        getReporter(),
        getEventBus(),
        runtime.getClock(),
        request,
        request.getOptions(ExecutionOptions.class).verboseFailures,
        request.getOptions(ExecutionOptions.class).showSubcommands,
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
   * @param analysisResult the analysis phase output
   * @param buildResult the mutable build result
   * @param skyframeExecutor the skyframe executor (if any)
   * @param packageRoots package roots collected from loading phase and BuildConfigutaionCollection
   * creation
   */
  void executeBuild(AnalysisResult analysisResult,
      BuildResult buildResult, @Nullable SkyframeExecutor skyframeExecutor,
      BuildConfigurationCollection configurations,
      ImmutableMap<PathFragment, Path> packageRoots)
      throws BuildFailedException, InterruptedException, AbruptExitException, TestExecException,
      ViewCreationFailedException {
    Stopwatch timer = Stopwatch.createStarted();
    prepare(packageRoots, configurations);

    ActionGraph actionGraph = analysisResult.getActionGraph();

    // Get top-level artifacts.
    ImmutableSet<Artifact> additionalArtifacts = analysisResult.getAdditionalArtifactsToBuild();

    // Create symlinks only after we've verified that we're actually
    // supposed to build something.
    if (getWorkspace().getFileSystem().supportsSymbolicLinks()) {
      List<BuildConfiguration> targetConfigurations =
          getView().getConfigurationCollection().getTargetConfigurations();
      // TODO(bazel-team): This is not optimal - we retain backwards compatibility in the case where
      // there's only a single configuration, but we don't create any symlinks in the multi-config
      // case. Can we do better? [multi-config]
      if (targetConfigurations.size() == 1) {
        OutputDirectoryLinksUtils.createOutputDirectoryLinks(
            runtime.getWorkspaceName(), getWorkspace(), getExecRoot(),
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

    ActionCache actionCache = getActionCache();
    Builder builder = createBuilder(request, executor, actionCache, skyframeExecutor);

    //
    // Execution proper.  All statements below are logically nested in
    // begin/end pairs.  No early returns or exceptions please!
    //

    Collection<ConfiguredTarget> configuredTargets = buildResult.getActualTargets();
    getEventBus().post(new ExecutionStartingEvent(configuredTargets));

    getReporter().handle(Event.progress("Building..."));

    // Conditionally record dependency-checker log:
    ExplanationHandler explanationHandler =
        installExplanationHandler(request.getBuildOptions().explanationPath,
                                  request.getOptionsDescription());

    Set<ConfiguredTarget> builtTargets = new HashSet<>();
    boolean interrupted = false;
    try {
      Iterable<Artifact> allArtifactsForProviders = Iterables.concat(
          additionalArtifacts,
          TopLevelArtifactHelper.getAllArtifactsToBuild(
              analysisResult.getTargetsToBuild(), analysisResult.getTopLevelContext())
              .getAllArtifacts(),
          TopLevelArtifactHelper.getAllArtifactsToTest(analysisResult.getTargetsToTest()));
      if (request.isRunningInEmacs()) {
        // The syntax of this message is tightly constrained by lisp/progmodes/compile.el in emacs
        request.getOutErr().printErrLn("blaze: Entering directory `" + getExecRoot() + "/'");
      }
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
        getView().clearAnalysisCache(analysisResult.getTargetsToBuild());
        actionGraph = null;
      }

      configureResourceManager(request);

      Profiler.instance().markPhase(ProfilePhase.EXECUTE);

      builder.buildArtifacts(additionalArtifacts,
          analysisResult.getParallelTests(),
          analysisResult.getExclusiveTests(),
          analysisResult.getTargetsToBuild(),
          executor, builtTargets,
          request.getBuildOptions().explanationPath != null,
          runtime.getLastExecutionTimeRange());

    } catch (InterruptedException e) {
      interrupted = true;
      throw e;
    } finally {
      runtime.recordLastExecutionTime();
      if (request.isRunningInEmacs()) {
        request.getOutErr().printErrLn("blaze: Leaving directory `" + getExecRoot() + "/'");
      }
      if (!interrupted) {
        getReporter().handle(Event.progress("Building complete."));
      }

      runtime.getEventBus().post(new ExecutionFinishedEvent(ImmutableMap.<String, Long> of(), 0L,
          skyframeExecutor.getOutputDirtyFiles(),
          skyframeExecutor.getModifiedFilesDuringPreviousBuild()));

      // Disable system load polling (noop if it was not enabled).
      ResourceManager.instance().setAutoSensing(false);
      executor.executionPhaseEnding();
      for (ActionContextProvider actionContextProvider : actionContextProviders) {
        actionContextProvider.executionPhaseEnding();
      }

      Profiler.instance().markPhase(ProfilePhase.FINISH);

      if (!interrupted) {
        saveCaches(actionCache);
      }

      long startTime = Profiler.nanoTimeMaybe();
      determineSuccessfulTargets(buildResult, configuredTargets, builtTargets, timer);
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

  private void prepare(ImmutableMap<PathFragment, Path> packageRoots,
      BuildConfigurationCollection configurations)
      throws ViewCreationFailedException {
    // Prepare for build.
    Profiler.instance().markPhase(ProfilePhase.PREPARE);

    // Create some tools symlinks / cleanup per-build state
    createActionLogDirectory();

    // Plant the symlink forest.
    plantSymlinkForest(packageRoots, configurations);
  }

  private void createToolsSymlinks() throws ExecutorInitException {
    try {
      runtime.getBinTools().setupBuildTools();
    } catch (ExecException e) {
      throw new ExecutorInitException("Tools symlink creation failed: "
          + e.getMessage() + "; build aborted", e);
    }
  }

  private void plantSymlinkForest(ImmutableMap<PathFragment, Path> packageRoots,
      BuildConfigurationCollection configurations) throws ViewCreationFailedException {
    try {
      FileSystemUtils.deleteTreesBelowNotPrefixed(getExecRoot(),
          new String[] { ".", "_", Constants.PRODUCT_NAME + "-"});
      // Delete the build configuration's temporary directories
      for (BuildConfiguration configuration : configurations.getTargetConfigurations()) {
        configuration.prepareForExecutionPhase();
      }
      FileSystemUtils.plantLinkForest(packageRoots, getExecRoot());
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
  private void determineSuccessfulTargets(BuildResult result,
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
    getEventBus().post(
        new ExecutionPhaseCompleteEvent(timer.stop().elapsed(TimeUnit.MILLISECONDS)));
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
        // Suppress display of source files (because we do no work to build them).
        continue;
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

    TopLevelArtifactContext context = request.getTopLevelArtifactContext();
    for (ConfiguredTarget target : succeeded) {
      Label label = target.getLabel();
      // For up-to-date targets report generated artifacts, but only
      // if they have associated action and not middleman artifacts.
      boolean headerFlag = true;
      for (Artifact artifact :
          TopLevelArtifactHelper.getAllArtifactsToBuild(target, context).getImportantArtifacts()) {
        if (!artifact.isSourceArtifact() && !artifact.isMiddlemanArtifact()) {
          if (headerFlag) {
            outErr.printErr("Target " + label + " up-to-date:\n");
            headerFlag = false;
          }
          outErr.printErrLn("  " +
              OutputDirectoryLinksUtils.getPrettyPath(artifact.getPath(),
                  runtime.getWorkspaceName(), getWorkspace(), request.getSymlinkPrefix()));
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
      OutputGroupProvider topLevelProvider =
          target.getProvider(OutputGroupProvider.class);
      if (topLevelProvider != null) {
        for (Artifact temp : topLevelProvider.getOutputGroup(OutputGroupProvider.TEMP_FILES)) {
          if (temp.getPath().exists()) {
            outErr.printErrLn("  See temp at " +
                OutputDirectoryLinksUtils.getPrettyPath(temp.getPath(),
                    runtime.getWorkspaceName(), getWorkspace(), request.getSymlinkPrefix()));
          }
        }
      }
    }
    if (!failed.isEmpty() && !request.getOptions(ExecutionOptions.class).verboseFailures) {
      outErr.printErr("Use --verbose_failures to see the command lines of failed build steps.\n");
    }
  }

  private ActionCache getActionCache() throws LocalEnvironmentException {
    try {
      return runtime.getPersistentActionCache();
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
      Executor executor,
      ActionCache actionCache,
      SkyframeExecutor skyframeExecutor) {
    BuildRequest.BuildRequestOptions options = request.getBuildOptions();
    boolean verboseExplanations = options.verboseExplanations;
    boolean keepGoing = request.getViewOptions().keepGoing;

    Path actionOutputRoot = runtime.getDirectories().getActionConsoleOutputDirectory();
    Predicate<Action> executionFilter = CheckUpToDateFilter.fromOptions(
        request.getOptions(ExecutionOptions.class));

    // jobs should have been verified in BuildRequest#validateOptions().
    Preconditions.checkState(options.jobs >= -1);
    int actualJobs = options.jobs == 0 ? 1 : options.jobs;  // Treat 0 jobs as a single task.

    // Unfortunately, the exec root cache is not shared with caches in the remote execution
    // client.
    fileCache = createBuildSingleFileCache(executor.getExecRoot());
    skyframeExecutor.setActionOutputRoot(actionOutputRoot);
    return new SkyframeBuilder(skyframeExecutor,
        new ActionCacheChecker(actionCache, getView().getArtifactFactory(), executionFilter,
            verboseExplanations),
        keepGoing, actualJobs, options.checkOutputFiles, fileCache,
        request.getBuildOptions().progressReportInterval);
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
      if (options.useResourceAutoSense) {
        getReporter().handle(
            Event.warn("Not using resource autosense due to known responsiveness issues"));
      }
      ResourceManager.instance().setAutoSensing(/*autosense=*/false);
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
    long actionCacheSaveTime;

    long startTime = BlazeClock.nanoTime();
    try {
      LOG.info("saving action cache...");
      actionCacheSizeInBytes = actionCache.save();
      LOG.info("action cache saved");
    } catch (IOException e) {
      getReporter().handle(Event.error("I/O error while writing action log: " + e.getMessage()));
    } finally {
      long stopTime = BlazeClock.nanoTime();
      actionCacheSaveTime =
          TimeUnit.MILLISECONDS.convert(stopTime - startTime, TimeUnit.NANOSECONDS);
      Profiler.instance().logSimpleTask(startTime, stopTime,
                                        ProfilerTask.INFO, "Saving action cache");
    }

    runtime.getEventBus().post(new CachesSavedEvent(
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
