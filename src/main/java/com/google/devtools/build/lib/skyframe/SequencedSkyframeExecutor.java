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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionException;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.concurrent.QuiescingExecutors;
import com.google.devtools.build.lib.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryConsumingOutputHandler;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.rewinding.RewindableGraphInconsistencyReceiver;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.DelegatingGraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.EmittedEventState;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EventFilter;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsProvider;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A SkyframeExecutor that implicitly assumes that builds can be done incrementally from the most
 * recent build. In other words, builds are "sequenced".
 */
public class SequencedSkyframeExecutor extends SkyframeExecutor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final int MODIFIED_OUTPUT_PATHS_SAMPLE_SIZE = 100;

  /**
   * If false, the graph will not store state useful for incremental builds, saving memory but
   * leaving the graph un-reusable. Subsequent builds will therefore not be incremental.
   *
   * <p>Avoids storing edges entirely and dereferences each action after execution.
   */
  private boolean trackIncrementalState = true;

  private boolean evaluatorNeedsReset = false;
  private boolean lastCommandKeptState = false;
  private boolean needGcAfterResettingEvaluator = false;

  private final AtomicInteger outputDirtyFiles = new AtomicInteger();
  private final ArrayBlockingQueue<String> outputDirtyFilesExecPathSample =
      new ArrayBlockingQueue<>(MODIFIED_OUTPUT_PATHS_SAMPLE_SIZE);
  private final AtomicInteger modifiedFilesDuringPreviousBuild = new AtomicInteger();

  private Duration outputTreeDiffCheckingDuration = Duration.ofSeconds(-1L);

  // Use delegation so that the underlying inconsistency receiver can be changed per-command without
  // recreating the evaluator.
  protected final DelegatingGraphInconsistencyReceiver inconsistencyReceiver =
      new DelegatingGraphInconsistencyReceiver(GraphInconsistencyReceiver.THROWING);

  protected SequencedSkyframeExecutor(
      Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit,
      PackageFactory pkgFactory,
      FileSystem fileSystem,
      BlazeDirectories directories,
      ActionKeyContext actionKeyContext,
      Factory workspaceStatusActionFactory,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      SyscallCache syscallCache,
      SkyFunction ignoredPackagePrefixesFunction,
      CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy,
      ImmutableList<BuildFileName> buildFilesByPriority,
      ExternalPackageHelper externalPackageHelper,
      @Nullable SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile,
      boolean shouldUseRepoDotBazel,
      SkyKeyStateReceiver skyKeyStateReceiver,
      BugReporter bugReporter) {
    super(
        skyframeExecutorConsumerOnInit,
        pkgFactory,
        fileSystem,
        directories,
        actionKeyContext,
        workspaceStatusActionFactory,
        extraSkyFunctions,
        syscallCache,
        ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
        ignoredPackagePrefixesFunction,
        crossRepositoryLabelViolationStrategy,
        buildFilesByPriority,
        externalPackageHelper,
        actionOnIOExceptionReadingBuildFile,
        shouldUseRepoDotBazel,
        /* shouldUnblockCpuWorkWhenFetchingDeps= */ false,
        new PackageProgressReceiver(),
        new ConfiguredTargetProgressReceiver(),
        skyKeyStateReceiver,
        bugReporter,
        diffAwarenessFactories,
        workspaceInfoFromDiffReceiver,
        new SequencedRecordingDifferencer(),
        repositoryHelpersHolder);
  }

  @Override
  public void resetEvaluator() {
    super.resetEvaluator();
    diffAwarenessManager.reset();
  }

  @Override
  protected MemoizingEvaluator createEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      SkyframeProgressReceiver progressReceiver,
      EmittedEventState emittedEventState) {
    return new InMemoryMemoizingEvaluator(
        skyFunctions,
        recordingDiffer,
        progressReceiver,
        inconsistencyReceiver,
        trackIncrementalState ? DEFAULT_EVENT_FILTER_WITH_ACTIONS : EventFilter.NO_STORAGE,
        emittedEventState,
        trackIncrementalState,
        /* usePooledInterning= */ true);
  }

  @Override
  protected Injectable injectable() {
    return recordingDiffer;
  }

  @VisibleForTesting
  public RecordingDifferencer getDifferencerForTesting() {
    return recordingDiffer;
  }

  @Override
  protected SkyframeProgressReceiver newSkyframeProgressReceiver() {
    return new SequencedSkyframeProgressReceiver();
  }

  /** A {@link SkyframeProgressReceiver} tracks dirty {@link FileValue.Key}s. */
  protected class SequencedSkyframeProgressReceiver extends SkyframeProgressReceiver {
    @Override
    public void dirtied(SkyKey skyKey, DirtyType dirtyType) {
      super.dirtied(skyKey, dirtyType);
      if (skyKey instanceof FileValue.Key) {
        incrementalBuildMonitor.reportInvalidatedFileValue();
      }
    }
  }

  @Nullable
  @Override
  public WorkspaceInfoFromDiff sync(
      ExtendedEventHandler eventHandler,
      PathPackageLocator packageLocator,
      UUID commandId,
      Map<String, String> clientEnv,
      Map<String, String> repoEnvOption,
      TimestampGranularityMonitor tsgm,
      QuiescingExecutors executors,
      OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    inconsistencyReceiver.setDelegate(getGraphInconsistencyReceiverForCommand(options));
    if (evaluatorNeedsReset) {
      // Recreate MemoizingEvaluator so that graph is recreated with correct edge-clearing status,
      // or if the graph doesn't have edges, so that a fresh graph can be used.
      resetEvaluator();
      evaluatorNeedsReset = false;
      if (needGcAfterResettingEvaluator) {
        // Collect weakly reachable objects to avoid resurrection. See b/291641466.
        try (var profiler =
            GoogleAutoProfilerUtils.logged(
                "manual GC to clean up from --keep_state_after_build command")) {
          System.gc();
        }
        needGcAfterResettingEvaluator = false;
      }
    }
    super.sync(
        eventHandler,
        packageLocator,
        commandId,
        clientEnv,
        repoEnvOption,
        tsgm,
        executors,
        options);
    long startTime = System.nanoTime();
    WorkspaceInfoFromDiff workspaceInfo = handleDiffs(eventHandler, options);
    long stopTime = System.nanoTime();
    Profiler.instance().logSimpleTask(startTime, stopTime, ProfilerTask.INFO, "handleDiffs");
    long duration = stopTime - startTime;
    sourceDiffCheckingDuration = duration > 0 ? Duration.ofNanos(duration) : Duration.ZERO;
    return workspaceInfo;
  }

  private GraphInconsistencyReceiver getGraphInconsistencyReceiverForCommand(
      OptionsProvider options) throws AbruptExitException {
    if (rewindingEnabled(options)) {
      // Currently incompatible with Skymeld i.e. this code path won't be run in Skymeld mode. We
      // may need to combine these GraphInconsistencyReceiver implementations in the future.
      var rewindableReceiver = new RewindableGraphInconsistencyReceiver();
      rewindableReceiver.setHeuristicallyDropNodes(heuristicallyDropNodes);
      return rewindableReceiver;
    }
    if (isMergedSkyframeAnalysisExecution()
        && ((options.getOptions(AnalysisOptions.class) != null
                && options.getOptions(AnalysisOptions.class).discardAnalysisCache)
            || !trackIncrementalState
            || heuristicallyDropNodes)) {
      return new SkymeldInconsistencyReceiver(heuristicallyDropNodes);
    }
    if (heuristicallyDropNodes) {
      return new NodeDroppingInconsistencyReceiver();
    }
    return GraphInconsistencyReceiver.THROWING;
  }

  private boolean rewindingEnabled(OptionsProvider options) throws AbruptExitException {
    var buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    return buildRequestOptions != null && buildRequestOptions.rewindLostInputs;
  }

  /**
   * The value types whose builders have direct access to the package locator, rather than accessing
   * it via an explicit Skyframe dependency. They need to be invalidated if the package locator
   * changes.
   */
  private static final ImmutableSet<SkyFunctionName> PACKAGE_LOCATOR_DEPENDENT_VALUES =
      ImmutableSet.of(
          FileStateKey.FILE_STATE,
          FileValue.FILE,
          SkyFunctions.DIRECTORY_LISTING_STATE,
          SkyFunctions.PREPARE_DEPS_OF_PATTERN,
          SkyFunctions.TARGET_PATTERN,
          SkyFunctions.TARGET_PATTERN_PHASE);

  @Override
  protected void onPkgLocatorChange(PathPackageLocator oldLocator, PathPackageLocator pkgLocator) {
    invalidate(SkyFunctionName.functionIsIn(PACKAGE_LOCATOR_DEPENDENT_VALUES));
  }

  void invalidate(Predicate<SkyKey> pred) {
    recordingDiffer.invalidate(Iterables.filter(memoizingEvaluator.getValues().keySet(), pred));
  }

  /** Sets the packages that should be treated as deleted and ignored. */
  @Override
  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  public void setDeletedPackages(Iterable<PackageIdentifier> pkgs) {
    ImmutableSet<PackageIdentifier> newDeletedPackagesSet = ImmutableSet.copyOf(pkgs);

    Set<PackageIdentifier> newlyDeletedOrNotDeletedPackages =
        Sets.symmetricDifference(deletedPackages.get(), newDeletedPackagesSet);
    if (!newlyDeletedOrNotDeletedPackages.isEmpty()) {
      // PackageLookupValue is a HERMETIC node type, so we can't invalidate it.
      memoizingEvaluator.delete(
          k -> PackageLookupValue.appliesToKey(k, newlyDeletedOrNotDeletedPackages::contains));
    }

    deletedPackages.set(newDeletedPackagesSet);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Necessary conditions to not store graph edges are either
   *
   * <ol>
   *   <li>batch (since incremental builds are not possible) and discard_analysis_cache (since
   *       otherwise user isn't concerned about saving memory this way).
   *   <li>track_incremental_state set to false.
   * </ol>
   */
  @Override
  public void decideKeepIncrementalState(
      boolean batch,
      boolean keepStateAfterBuild,
      boolean shouldTrackIncrementalState,
      boolean heuristicallyDropNodes,
      boolean discardAnalysisCache,
      EventHandler eventHandler) {
    Preconditions.checkState(!active);
    boolean oldValueOfTrackIncrementalState = trackIncrementalState;

    // First check if the incrementality state should be kept around during the build.
    boolean explicitlyRequestedNoIncrementalData = !shouldTrackIncrementalState;
    boolean implicitlyRequestedNoIncrementalData = (batch && discardAnalysisCache);
    trackIncrementalState =
        !explicitlyRequestedNoIncrementalData && !implicitlyRequestedNoIncrementalData;
    if (explicitlyRequestedNoIncrementalData != implicitlyRequestedNoIncrementalData) {
      if (!explicitlyRequestedNoIncrementalData) {
        eventHandler.handle(
            Event.warn(
                "--batch and --discard_analysis_cache specified, but --notrack_incremental_state "
                    + "not specified: incrementality data is implicitly discarded, but you may need"
                    + " to specify --notrack_incremental_state in the future if you want to "
                    + "maximize memory savings."));
      }
      if (!batch && keepStateAfterBuild) {
        eventHandler.handle(
            Event.warn(
                "--notrack_incremental_state was specified, but without "
                    + "--nokeep_state_after_build. Inmemory state from this build will not be "
                    + "reusable, but it will not get fully wiped until the beginning of the next "
                    + "build. Use --nokeep_state_after_build to clean up eagerly."));
      }
    }

    if (trackIncrementalState) {
      if (heuristicallyDropNodes) {
        eventHandler.handle(
            Event.warn(
                "--heuristically_drop_nodes was specified with track incremental state also being"
                    + " true. The flag is ignored and no node is heuristically dropped in the track"
                    + " incremental mode."));
      }
      this.heuristicallyDropNodes = false;
    } else {
      this.heuristicallyDropNodes = heuristicallyDropNodes;
    }

    // Now check if it is necessary to wipe the previous state. We do this if either the previous
    // or current command requires the build to have been isolated.
    if (oldValueOfTrackIncrementalState != trackIncrementalState) {
      logger.atInfo().log("Set incremental state to %b", trackIncrementalState);
      evaluatorNeedsReset = true;
    } else if (!trackIncrementalState) {
      evaluatorNeedsReset = true;
    }
    if (evaluatorNeedsReset && lastCommandKeptState) {
      needGcAfterResettingEvaluator = true;
    }
    lastCommandKeptState = keepStateAfterBuild;
  }

  @Override
  public boolean tracksStateForIncrementality() {
    return trackIncrementalState;
  }

  @Override
  public void clearAnalysisCacheImpl(
      ImmutableSet<ConfiguredTarget> topLevelTargets, ImmutableSet<AspectKey> topLevelAspects) {
    discardPreExecutionCache(
        topLevelTargets,
        topLevelAspects,
        trackIncrementalState ? DiscardType.ANALYSIS_REFS_ONLY : DiscardType.ALL);
  }

  @Override
  public void invalidateTransientErrors() {
    checkActive();
    recordingDiffer.invalidateTransientErrors();
  }

  @Override
  public void detectModifiedOutputFiles(
      ModifiedFileSet modifiedOutputFiles,
      @Nullable Range<Long> lastExecutionTimeRange,
      RemoteArtifactChecker remoteArtifactChecker,
      int fsvcThreads)
      throws InterruptedException {
    long startTime = System.nanoTime();
    FilesystemValueChecker fsvc =
        new FilesystemValueChecker(
            Preconditions.checkNotNull(tsgm.get()), syscallCache, fsvcThreads);
    BatchStat batchStatter = outputService == null ? null : outputService.getBatchStatter();
    recordingDiffer.invalidate(
        fsvc.getDirtyActionValues(
            memoizingEvaluator.getValues(),
            batchStatter,
            modifiedOutputFiles,
            remoteArtifactChecker,
            (maybeModifiedTime, artifact) -> {
              modifiedFiles.incrementAndGet();
              int dirtyOutputsCount = outputDirtyFiles.incrementAndGet();
              if (lastExecutionTimeRange != null
                  && lastExecutionTimeRange.contains(maybeModifiedTime)) {
                modifiedFilesDuringPreviousBuild.incrementAndGet();
              }
              if (dirtyOutputsCount <= MODIFIED_OUTPUT_PATHS_SAMPLE_SIZE) {
                outputDirtyFilesExecPathSample.offer(artifact.getExecPathString());
              }
            }));
    logger.atInfo().log("Found %d modified files from last build", modifiedFiles.get());
    long stopTime = System.nanoTime();
    Profiler.instance()
        .logSimpleTask(startTime, stopTime, ProfilerTask.INFO, "detectModifiedOutputFiles");
    long duration = stopTime - startTime;
    outputTreeDiffCheckingDuration = duration > 0 ? Duration.ofNanos(duration) : Duration.ZERO;
  }

  @Override
  public List<RuleStat> getRuleStats(ExtendedEventHandler eventHandler)
      throws InterruptedException {
    Map<String, RuleStat> ruleStats = new HashMap<>();
    for (Map.Entry<SkyKey, SkyValue> skyKeyAndValue :
        memoizingEvaluator.getDoneValues().entrySet()) {
      SkyValue value = skyKeyAndValue.getValue();
      SkyKey key = skyKeyAndValue.getKey();
      SkyFunctionName functionName = key.functionName();
      if (value instanceof RuleConfiguredTargetValue) {
        RuleConfiguredTargetValue ctValue = (RuleConfiguredTargetValue) value;
        ConfiguredTarget configuredTarget = ctValue.getConfiguredTarget();
        if (configuredTarget instanceof RuleConfiguredTarget) {

          Rule rule;
          try {
            rule = (Rule) getPackageManager().getTarget(eventHandler, configuredTarget.getLabel());
          } catch (NoSuchPackageException | NoSuchTargetException e) {
            throw new IllegalStateException(
                "Failed to get Rule target from package when calculating stats.", e);
          }
          RuleClass ruleClass = rule.getRuleClassObject();
          RuleStat ruleStat =
              ruleStats.computeIfAbsent(
                  ruleClass.getKey(), k -> new RuleStat(k, ruleClass.getName(), true));
          ruleStat.addRule(ctValue.getNumActions());
        }
      } else if (functionName.equals(SkyFunctions.ASPECT)) {
        AspectValue aspectValue = (AspectValue) value;
        AspectClass aspectClass = aspectValue.getAspect().getAspectClass();
        RuleStat ruleStat =
            ruleStats.computeIfAbsent(
                aspectClass.getKey(), k -> new RuleStat(k, aspectClass.getName(), false));
        ruleStat.addRule(aspectValue.getNumActions());
      }
    }
    return new ArrayList<>(ruleStats.values());
  }

  public void dumpSkyframeStateInParallel(
      ActionGraphDump actionGraphDump, AqueryConsumingOutputHandler aqueryConsumingOutputHandler)
      throws CommandLineExpansionException, IOException, TemplateExpansionException {
    ImmutableList.Builder<Callable<Void>> tasks = ImmutableList.builder();

    try {
      for (Map.Entry<SkyKey, SkyValue> skyKeyAndValue :
          memoizingEvaluator.getDoneValues().entrySet()) {
        SkyKey key = skyKeyAndValue.getKey();
        SkyValue skyValue = skyKeyAndValue.getValue();
        if (skyValue == null) {
          // The skyValue may be null in case analysis of the previous build failed.
          continue;
        }
        if (skyValue instanceof RuleConfiguredTargetValue) {
          tasks.add(
              () -> {
                var configuredTarget = (RuleConfiguredTargetValue) skyValue;
                // Only dumps the value for non-delegating keys.
                if (configuredTarget.getConfiguredTarget().getLookupKey().equals(key)) {
                  actionGraphDump.dumpConfiguredTarget(configuredTarget);
                }
                return null;
              });
        } else if (key.functionName().equals(SkyFunctions.ASPECT)) {
          AspectValue aspectValue = (AspectValue) skyValue;
          AspectKey aspectKey = (AspectKey) key;
          ConfiguredTargetValue configuredTargetValue =
              (ConfiguredTargetValue)
                  memoizingEvaluator.getExistingValue(aspectKey.getBaseConfiguredTargetKey());
          tasks.add(
              () -> {
                actionGraphDump.dumpAspect(aspectValue, configuredTargetValue);
                return null;
              });
        }
      }
      ForkJoinPool executor =
          NamedForkJoinPool.newNamedPool(
              "action-graph-dump", Runtime.getRuntime().availableProcessors());
      try {
        Future<Void> consumerFuture = executor.submit(aqueryConsumingOutputHandler.startConsumer());
        List<Future<Void>> futures = executor.invokeAll(tasks.build());
        for (Future<Void> future : futures) {
          future.get();
        }
        aqueryConsumingOutputHandler.stopConsumer(/* discardRemainingTasks= */ false);
        // Get any possible exception from the consumer.
        consumerFuture.get();
      } catch (ExecutionException e) {
        aqueryConsumingOutputHandler.stopConsumer(/* discardRemainingTasks= */ true);
        Throwable cause = Throwables.getRootCause(e);
        throwIfInstanceOf(cause, CommandLineExpansionException.class);
        throwIfInstanceOf(cause, TemplateExpansionException.class);
        throwIfInstanceOf(cause, IOException.class);
        throwIfInstanceOf(cause, InterruptedException.class);
        throwIfUnchecked(cause);
        throw new IllegalStateException("Unexpected exception type: ", e);
      } finally {
        executor.shutdown();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  /** Support for aquery output. */
  public void dumpSkyframeState(ActionGraphDump actionGraphDump)
      throws CommandLineExpansionException, IOException, TemplateExpansionException {

    for (Map.Entry<SkyKey, SkyValue> skyKeyAndValue :
        memoizingEvaluator.getDoneValues().entrySet()) {
      SkyKey key = skyKeyAndValue.getKey();
      SkyValue skyValue = skyKeyAndValue.getValue();
      if (skyValue == null) {
        // The skyValue may be null in case analysis of the previous build failed.
        continue;
      }
      try {
        if (skyValue instanceof RuleConfiguredTargetValue) {
          var configuredTarget = (RuleConfiguredTargetValue) skyValue;
          // Only dumps the value for non-delegating keys.
          if (configuredTarget.getConfiguredTarget().getLookupKey().equals(key)) {
            actionGraphDump.dumpConfiguredTarget(configuredTarget);
          }
        } else if (key.functionName().equals(SkyFunctions.ASPECT)) {
          AspectValue aspectValue = (AspectValue) skyValue;
          AspectKey aspectKey = (AspectKey) key;
          ConfiguredTargetValue configuredTargetValue =
              (ConfiguredTargetValue)
                  memoizingEvaluator.getExistingValue(aspectKey.getBaseConfiguredTargetKey());
          actionGraphDump.dumpAspect(aspectValue, configuredTargetValue);
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("No interruption in sequenced evaluation", e);
      }
    }
  }

  /**
   * In addition to calling the superclass method, deletes all analysis-related values from the
   * Skyframe cache. This is done to save memory (e.g. on a configuration change); since the
   * configuration is part of the key, these key/value pairs will be sitting around doing nothing
   * until the configuration changes back to the previous value.
   *
   * <p>The next evaluation will delete all invalid values.
   */
  @Override
  public void handleAnalysisInvalidatingChange() {
    super.handleAnalysisInvalidatingChange();
    memoizingEvaluator.delete(this::shouldDeleteOnAnalysisInvalidatingChange);
  }

  // Also remove ActionLookupData since all such nodes depend on ActionLookupKey nodes and deleting
  // en masse is cheaper than deleting via graph traversal (b/192863968).
  @ForOverride
  protected boolean shouldDeleteOnAnalysisInvalidatingChange(SkyKey k) {
    return k instanceof ArtifactNestedSetKey
        || k instanceof ActionLookupKey
        || (k instanceof BuildConfigurationKey
            && getSkyframeBuildView().getBuildConfiguration() != null
            && !k.equals(getSkyframeBuildView().getBuildConfiguration().getKey()))
        || k instanceof ActionLookupData;
  }

  /**
   * Deletes all ConfiguredTarget values from the Skyframe cache.
   *
   * <p>After the execution of this method all invalidated and marked for deletion values (and the
   * values depending on them) will be deleted from the cache.
   *
   * <p>WARNING: Note that a call to this method leaves legacy data inconsistent with Skyframe. The
   * next build should clear the legacy caches.
   */
  @Override
  protected void dropConfiguredTargetsNow(final ExtendedEventHandler eventHandler) {
    handleAnalysisInvalidatingChange();
    // Run the invalidator to actually delete the values.
    try {
      progressReceiver.ignoreInvalidations = true;
      Uninterruptibles.callUninterruptibly(
          () -> {
            EvaluationContext evaluationContext =
                newEvaluationContextBuilder()
                    .setKeepGoing(false)
                    .setParallelism(ResourceUsage.getAvailableProcessors())
                    .setEventHandler(eventHandler)
                    .build();
            memoizingEvaluator.evaluate(ImmutableList.of(), evaluationContext);
            return null;
          });
    } catch (Exception e) {
      throw new IllegalStateException(e);
    } finally {
      progressReceiver.ignoreInvalidations = false;
    }
  }

  @Override
  protected ExecutionFinishedEvent.Builder createExecutionFinishedEventInternal() {
    ExecutionFinishedEvent.Builder builder =
        ExecutionFinishedEvent.builder()
            .setOutputDirtyFiles(outputDirtyFiles.getAndSet(0))
            .setOutputDirtyFileExecPathSample(ImmutableList.copyOf(outputDirtyFilesExecPathSample))
            .setOutputModifiedFilesDuringPreviousBuild(
                modifiedFilesDuringPreviousBuild.getAndSet(0))
            .setSourceDiffCheckingDuration(sourceDiffCheckingDuration)
            .setNumSourceFilesCheckedBecauseOfMissingDiffs(
                numSourceFilesCheckedBecauseOfMissingDiffs)
            .setOutputTreeDiffCheckingDuration(outputTreeDiffCheckingDuration);
    outputDirtyFilesExecPathSample.clear();
    sourceDiffCheckingDuration = Duration.ZERO;
    outputTreeDiffCheckingDuration = Duration.ZERO;
    return builder;
  }

  @Override
  public void deleteOldNodes(long versionWindowForDirtyGc) {
    // TODO(bazel-team): perhaps we should come up with a separate GC class dedicated to maintaining
    // value garbage. If we ever do so, this logic should be moved there.
    if (trackIncrementalState) {
      memoizingEvaluator.deleteDirty(versionWindowForDirtyGc);
    }
  }

  @Override
  protected void dumpPackages(PrintStream out) {
    Iterable<SkyKey> packageSkyKeys =
        Iterables.filter(
            memoizingEvaluator.getValues().keySet(),
            SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE));
    out.println(Iterables.size(packageSkyKeys) + " packages");
    for (SkyKey packageSkyKey : packageSkyKeys) {
      PackageValue pkgVal = ((PackageValue) memoizingEvaluator.getValues().get(packageSkyKey));
      if (pkgVal != null) {
        Package pkg =
            ((PackageValue) memoizingEvaluator.getValues().get(packageSkyKey)).getPackage();
        pkg.dump(out);
      } else {
        out.println("  Package " + packageSkyKey + " is in error.");
      }
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder class for {@link SequencedSkyframeExecutor}.
   *
   * <p>Allows addition of the new arguments to {@link SequencedSkyframeExecutor} constructor
   * without the need to modify all the places, where {@link SequencedSkyframeExecutor} is
   * constructed (if the default value can be provided for the new argument in Builder).
   */
  public static final class Builder {
    PackageFactory pkgFactory;
    FileSystem fileSystem;
    BlazeDirectories directories;
    ActionKeyContext actionKeyContext;
    private CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy;
    private ImmutableList<BuildFileName> buildFilesByPriority;
    private ExternalPackageHelper externalPackageHelper;
    private ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile;
    private boolean shouldUseRepoDotBazel = true;

    // Fields with default values.
    private ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions = ImmutableMap.of();
    private Factory workspaceStatusActionFactory;
    private Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories = ImmutableList.of();
    private WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver =
        (ignored1, ignored2) -> {};
    @Nullable private SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder = null;
    private Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit = skyframeExecutor -> {};
    private SkyFunction ignoredPackagePrefixesFunction;
    private BugReporter bugReporter = BugReporter.defaultInstance();
    private SkyKeyStateReceiver skyKeyStateReceiver = SkyKeyStateReceiver.NULL_INSTANCE;
    private SyscallCache syscallCache = null;

    private Builder() {}

    public SequencedSkyframeExecutor build() {
      // Check that the values were explicitly set.
      Preconditions.checkNotNull(pkgFactory);
      Preconditions.checkNotNull(fileSystem);
      Preconditions.checkNotNull(directories);
      Preconditions.checkNotNull(actionKeyContext);
      Preconditions.checkNotNull(crossRepositoryLabelViolationStrategy);
      Preconditions.checkNotNull(buildFilesByPriority);
      Preconditions.checkNotNull(externalPackageHelper);
      Preconditions.checkNotNull(actionOnIOExceptionReadingBuildFile);
      Preconditions.checkNotNull(ignoredPackagePrefixesFunction);

      SequencedSkyframeExecutor skyframeExecutor =
          new SequencedSkyframeExecutor(
              skyframeExecutorConsumerOnInit,
              pkgFactory,
              fileSystem,
              directories,
              actionKeyContext,
              workspaceStatusActionFactory,
              diffAwarenessFactories,
              workspaceInfoFromDiffReceiver,
              extraSkyFunctions,
              Preconditions.checkNotNull(syscallCache),
              ignoredPackagePrefixesFunction,
              crossRepositoryLabelViolationStrategy,
              buildFilesByPriority,
              externalPackageHelper,
              repositoryHelpersHolder,
              actionOnIOExceptionReadingBuildFile,
              shouldUseRepoDotBazel,
              skyKeyStateReceiver,
              bugReporter);
      skyframeExecutor.init();
      return skyframeExecutor;
    }

    @CanIgnoreReturnValue
    public Builder setPkgFactory(PackageFactory pkgFactory) {
      this.pkgFactory = pkgFactory;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setFileSystem(FileSystem fileSystem) {
      this.fileSystem = fileSystem;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setDirectories(BlazeDirectories directories) {
      this.directories = directories;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setActionKeyContext(ActionKeyContext actionKeyContext) {
      this.actionKeyContext = actionKeyContext;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setIgnoredPackagePrefixesFunction(SkyFunction ignoredPackagePrefixesFunction) {
      this.ignoredPackagePrefixesFunction = ignoredPackagePrefixesFunction;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBugReporter(BugReporter bugReporter) {
      this.bugReporter = bugReporter;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExtraSkyFunctions(
        ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions) {
      this.extraSkyFunctions = extraSkyFunctions;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setWorkspaceStatusActionFactory(@Nullable Factory workspaceStatusActionFactory) {
      this.workspaceStatusActionFactory = workspaceStatusActionFactory;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setDiffAwarenessFactories(
        Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories) {
      this.diffAwarenessFactories = diffAwarenessFactories;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setWorkspaceInfoFromDiffReceiver(
        WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver) {
      this.workspaceInfoFromDiffReceiver = workspaceInfoFromDiffReceiver;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setRepositoryHelpersHolder(
        SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder) {
      this.repositoryHelpersHolder = repositoryHelpersHolder;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setCrossRepositoryLabelViolationStrategy(
        CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy) {
      this.crossRepositoryLabelViolationStrategy = crossRepositoryLabelViolationStrategy;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBuildFilesByPriority(ImmutableList<BuildFileName> buildFilesByPriority) {
      this.buildFilesByPriority = buildFilesByPriority;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExternalPackageHelper(ExternalPackageHelper externalPackageHelper) {
      this.externalPackageHelper = externalPackageHelper;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setActionOnIOExceptionReadingBuildFile(
        ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile) {
      this.actionOnIOExceptionReadingBuildFile = actionOnIOExceptionReadingBuildFile;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setShouldUseRepoDotBazel(boolean shouldUseRepoDotBazel) {
      this.shouldUseRepoDotBazel = shouldUseRepoDotBazel;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSkyframeExecutorConsumerOnInit(
        Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit) {
      this.skyframeExecutorConsumerOnInit = skyframeExecutorConsumerOnInit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSkyKeyStateReceiver(SkyKeyStateReceiver skyKeyStateReceiver) {
      this.skyKeyStateReceiver = Preconditions.checkNotNull(skyKeyStateReceiver);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSyscallCache(SyscallCache syscallCache) {
      this.syscallCache = syscallCache;
      return this;
    }
  }
}
