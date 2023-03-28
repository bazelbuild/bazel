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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionException;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
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
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.DiffAwarenessManager.ProcessableModifiedFileSet;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.BasicFilesystemDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.ExternalDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.MissingDiffDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.UnionDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFilesKnowledge;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.skyframe.FilesystemValueChecker.ImmutableBatchDirtyResult;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.rewinding.RewindableGraphInconsistencyReceiver;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EventFilter;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.ParsedOptionDescription;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A SkyframeExecutor that implicitly assumes that builds can be done incrementally from the most
 * recent build. In other words, builds are "sequenced".
 */
public final class SequencedSkyframeExecutor extends SkyframeExecutor {

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

  // This is intentionally not kept in sync with the evaluator: we may reset the evaluator without
  // ever losing injected/invalidated data here. This is safe because the worst that will happen is
  // that on the next build we try to inject/invalidate some nodes that aren't needed for the build.
  private final RecordingDifferencer recordingDiffer = new SequencedRecordingDifferencer();
  private final DiffAwarenessManager diffAwarenessManager;
  // If this is null then workspace header pre-calculation won't happen.
  @Nullable private final SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder;
  private Set<String> previousClientEnvironment = ImmutableSet.of();

  private final AtomicInteger modifiedFiles = new AtomicInteger();
  private final AtomicInteger outputDirtyFiles = new AtomicInteger();
  private final ArrayBlockingQueue<String> outputDirtyFilesExecPathSample =
      new ArrayBlockingQueue<>(MODIFIED_OUTPUT_PATHS_SAMPLE_SIZE);
  private final AtomicInteger modifiedFilesDuringPreviousBuild = new AtomicInteger();

  private Duration sourceDiffCheckingDuration = Duration.ofSeconds(-1L);
  private int numSourceFilesCheckedBecauseOfMissingDiffs;
  private Duration outputTreeDiffCheckingDuration = Duration.ofSeconds(-1L);

  private final WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver;
  private GraphInconsistencyReceiver inconsistencyReceiver = GraphInconsistencyReceiver.THROWING;

  private SequencedSkyframeExecutor(
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
        /* shouldUnblockCpuWorkWhenFetchingDeps= */ false,
        new PackageProgressReceiver(),
        new ConfiguredTargetProgressReceiver(),
        skyKeyStateReceiver,
        bugReporter);
    this.diffAwarenessManager = new DiffAwarenessManager(diffAwarenessFactories);
    this.repositoryHelpersHolder = repositoryHelpersHolder;
    this.workspaceInfoFromDiffReceiver = workspaceInfoFromDiffReceiver;
  }

  @Override
  public void resetEvaluator() {
    super.resetEvaluator();
    diffAwarenessManager.reset();
  }

  @Override
  protected InMemoryMemoizingEvaluator createEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      SkyframeProgressReceiver progressReceiver,
      NestedSetVisitor.VisitedState emittedEventState) {
    return new InMemoryMemoizingEvaluator(
        skyFunctions,
        recordingDiffer,
        progressReceiver,
        inconsistencyReceiver,
        trackIncrementalState ? DEFAULT_EVENT_FILTER_WITH_ACTIONS : EventFilter.NO_STORAGE,
        emittedEventState,
        trackIncrementalState,
        /* usePooledSkyKeyInterning= */ true);
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
    return new SkyframeProgressReceiver() {
      @Override
      public void invalidated(SkyKey skyKey, InvalidationState state) {
        super.invalidated(skyKey, state);
        if (state == InvalidationState.DIRTY && skyKey instanceof FileValue.Key) {
          incrementalBuildMonitor.reportInvalidatedFileValue();
        }
      }
    };
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
    if (evaluatorNeedsReset) {
      if (rewindingPermitted(options)) {
        var rewindableReceiver = new RewindableGraphInconsistencyReceiver();
        rewindableReceiver.setHeuristicallyDropNodes(heuristicallyDropNodes);
        this.inconsistencyReceiver = rewindableReceiver;
      } else {
        inconsistencyReceiver =
            heuristicallyDropNodes
                ? new NodeDroppingInconsistencyReceiver()
                : GraphInconsistencyReceiver.THROWING;
      }

      // Recreate MemoizingEvaluator so that graph is recreated with correct edge-clearing status,
      // or if the graph doesn't have edges, so that a fresh graph can be used.
      resetEvaluator();
      evaluatorNeedsReset = false;
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

  private boolean rewindingPermitted(OptionsProvider options) {
    // Rewinding is only supported with no incremental state and no action cache.
    if (trackIncrementalState) {
      return false;
    }
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    return buildRequestOptions != null
        && !buildRequestOptions.useActionCache
        && buildRequestOptions.rewindLostInputs;
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

  @Override
  protected void invalidate(Predicate<SkyKey> pred) {
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

  /** Uses diff awareness on all the package paths to invalidate changed files. */
  @VisibleForTesting
  public void handleDiffsForTesting(ExtendedEventHandler eventHandler)
      throws InterruptedException, AbruptExitException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.checkOutputFiles = false;
    ClassToInstanceMap<OptionsBase> options =
        ImmutableClassToInstanceMap.of(PackageOptions.class, packageOptions);
    handleDiffs(
        eventHandler,
        new OptionsProvider() {
          @Nullable
          @Override
          public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
            return options.getInstance(optionsClass);
          }

          @Override
          public ImmutableMap<String, Object> getStarlarkOptions() {
            return ImmutableMap.of();
          }

          @Override
          public ImmutableMap<String, Object> getExplicitStarlarkOptions(
              java.util.function.Predicate<? super ParsedOptionDescription> filter) {
            return ImmutableMap.of();
          }
        });
  }

  @Nullable
  private WorkspaceInfoFromDiff handleDiffs(
      ExtendedEventHandler eventHandler, OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    TimestampGranularityMonitor tsgm = this.tsgm.get();
    modifiedFiles.set(0);
    numSourceFilesCheckedBecauseOfMissingDiffs = 0;

    WorkspaceInfoFromDiff workspaceInfo = null;
    Map<Root, DiffAwarenessManager.ProcessableModifiedFileSet> modifiedFilesByPathEntry =
        Maps.newHashMap();
    Set<Pair<Root, DiffAwarenessManager.ProcessableModifiedFileSet>>
        pathEntriesWithoutDiffInformation = Sets.newHashSet();
    ImmutableList<Root> pkgRoots = pkgLocator.get().getPathEntries();
    for (Root pathEntry : pkgRoots) {
      DiffAwarenessManager.ProcessableModifiedFileSet modifiedFileSet =
          diffAwarenessManager.getDiff(eventHandler, pathEntry, options);
      if (pkgRoots.size() == 1) {
        workspaceInfo = modifiedFileSet.getWorkspaceInfo();
        workspaceInfoFromDiffReceiver.syncWorkspaceInfoFromDiff(
            pathEntry.asPath().asFragment(), workspaceInfo);
      }
      if (modifiedFileSet.getModifiedFileSet().treatEverythingAsModified()) {
        pathEntriesWithoutDiffInformation.add(Pair.of(pathEntry, modifiedFileSet));
      } else {
        modifiedFilesByPathEntry.put(pathEntry, modifiedFileSet);
      }
    }
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    int fsvcThreads = buildRequestOptions == null ? 200 : buildRequestOptions.fsvcThreads;
    handleDiffsWithCompleteDiffInformation(tsgm, modifiedFilesByPathEntry, fsvcThreads);
    RepositoryOptions repoOptions = options.getOptions(RepositoryOptions.class);
    handleDiffsWithMissingDiffInformation(
        eventHandler,
        tsgm,
        pathEntriesWithoutDiffInformation,
        options.getOptions(PackageOptions.class).checkOutputFiles,
        repoOptions == null || repoOptions.checkExternalRepositoryFiles,
        fsvcThreads);
    handleClientEnvironmentChanges();
    return workspaceInfo;
  }

  /** Invalidates entries in the client environment. */
  private void handleClientEnvironmentChanges() {
    // Remove deleted client environmental variables.
    Iterable<SkyKey> deletedKeys =
        Sets.difference(previousClientEnvironment, clientEnv.get().keySet()).stream()
            .map(ClientEnvironmentFunction::key)
            .collect(toImmutableList());
    recordingDiffer.invalidate(deletedKeys);
    previousClientEnvironment = clientEnv.get().keySet();
    // Inject current client environmental values. We can inject unconditionally without fearing
    // over-invalidation; skyframe will not invalidate an injected key if the key's new value is the
    // same as the old value.
    ImmutableMap.Builder<SkyKey, SkyValue> newValuesBuilder = ImmutableMap.builder();
    for (Map.Entry<String, String> entry : clientEnv.get().entrySet()) {
      newValuesBuilder.put(
          ClientEnvironmentFunction.key(entry.getKey()),
          new ClientEnvironmentValue(entry.getValue()));
    }
    recordingDiffer.inject(newValuesBuilder.buildOrThrow());
  }

  /**
   * Invalidates files under path entries whose corresponding {@link DiffAwareness} gave an exact
   * diff. Removes entries from the given map as they are processed. All of the files need to be
   * invalidated, so the map should be empty upon completion of this function.
   */
  private void handleDiffsWithCompleteDiffInformation(
      TimestampGranularityMonitor tsgm,
      Map<Root, ProcessableModifiedFileSet> modifiedFilesByPathEntry,
      int fsvcThreads)
      throws InterruptedException, AbruptExitException {
    for (Root pathEntry : ImmutableSet.copyOf(modifiedFilesByPathEntry.keySet())) {
      DiffAwarenessManager.ProcessableModifiedFileSet processableModifiedFileSet =
          modifiedFilesByPathEntry.get(pathEntry);
      ModifiedFileSet modifiedFileSet = processableModifiedFileSet.getModifiedFileSet();
      Preconditions.checkState(!modifiedFileSet.treatEverythingAsModified(), pathEntry);
      handleChangedFiles(
          ImmutableList.of(pathEntry),
          getDiff(tsgm, modifiedFileSet, pathEntry, fsvcThreads),
          /*numSourceFilesCheckedIfDiffWasMissing=*/ 0);
      processableModifiedFileSet.markProcessed();
    }
  }

  /**
   * Finds and invalidates changed files under path entries whose corresponding {@link
   * DiffAwareness} said all files may have been modified.
   */
  private void handleDiffsWithMissingDiffInformation(
      ExtendedEventHandler eventHandler,
      TimestampGranularityMonitor tsgm,
      Set<Pair<Root, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation,
      boolean checkOutputFiles,
      boolean checkExternalRepositoryFiles,
      int fsvcThreads)
      throws InterruptedException {

    ExternalFilesKnowledge externalFilesKnowledge = externalFilesHelper.getExternalFilesKnowledge();
    if (!pathEntriesWithoutDiffInformation.isEmpty()
        || (checkOutputFiles && externalFilesKnowledge.anyOutputFilesSeen)
        || (checkExternalRepositoryFiles && repositoryHelpersHolder != null)
        || (checkExternalRepositoryFiles && externalFilesKnowledge.anyFilesInExternalReposSeen)
        || externalFilesKnowledge.tooManyNonOutputExternalFilesSeen) {
      // We freshly compute knowledge of the presence of external files in the skyframe graph. We
      // use a fresh ExternalFilesHelper instance and only set the real instance's knowledge *after*
      // we are done with the graph scan, lest an interrupt during the graph scan causes us to
      // incorrectly think there are no longer any external files.
      ExternalFilesHelper tmpExternalFilesHelper =
          externalFilesHelper.cloneWithFreshExternalFilesKnowledge();

      // Before running the FilesystemValueChecker, ensure that all values marked for invalidation
      // have actually been invalidated (recall that invalidation happens at the beginning of the
      // next evaluate() call), because checking those is a waste of time.
      EvaluationContext evaluationContext =
          newEvaluationContextBuilder()
              .setKeepGoing(false)
              .setParallelism(DEFAULT_THREAD_COUNT)
              .setEventHandler(eventHandler)
              .build();
      memoizingEvaluator.evaluate(ImmutableList.of(), evaluationContext);

      FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, syscallCache, fsvcThreads);
      // We need to manually check for changes to known files. This entails finding all dirty file
      // system values under package roots for which we don't have diff information. If at least
      // one path entry doesn't have diff information, then we're going to have to iterate over
      // the skyframe values at least once no matter what.
      Set<Root> diffPackageRootsUnderWhichToCheck = new HashSet<>();
      for (Pair<Root, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
          pathEntriesWithoutDiffInformation) {
        diffPackageRootsUnderWhichToCheck.add(pair.getFirst());
      }

      EnumSet<FileType> fileTypesToCheck = EnumSet.noneOf(FileType.class);
      Iterable<SkyValueDirtinessChecker> dirtinessCheckers = ImmutableList.of();

      if (!diffPackageRootsUnderWhichToCheck.isEmpty()) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(
                    new MissingDiffDirtinessChecker(diffPackageRootsUnderWhichToCheck)));
      }
      if (checkExternalRepositoryFiles && repositoryHelpersHolder != null) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(repositoryHelpersHolder.repositoryDirectoryDirtinessChecker()));
      }
      if (checkExternalRepositoryFiles) {
        fileTypesToCheck = EnumSet.of(FileType.EXTERNAL_REPO);
      }
      if (externalFilesKnowledge.tooManyNonOutputExternalFilesSeen
          || !externalFilesKnowledge.nonOutputExternalFilesSeen.isEmpty()) {
        fileTypesToCheck.add(FileType.EXTERNAL);
      }
      // See the comment for FileType.OUTPUT for why we need to consider output files here.
      if (checkOutputFiles) {
        fileTypesToCheck.add(FileType.OUTPUT);
      }
      if (!fileTypesToCheck.isEmpty()) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(
                    new ExternalDirtinessChecker(tmpExternalFilesHelper, fileTypesToCheck)));
      }
      Preconditions.checkArgument(!Iterables.isEmpty(dirtinessCheckers));

      logger.atInfo().log(
          "About to scan skyframe graph checking for filesystem nodes of types %s",
          Iterables.toString(fileTypesToCheck));
      ImmutableBatchDirtyResult batchDirtyResult;
      try (SilentCloseable c = Profiler.instance().profile("fsvc.getDirtyKeys")) {
        batchDirtyResult =
            fsvc.getDirtyKeys(
                memoizingEvaluator.getValues(),
                new UnionDirtinessChecker(ImmutableList.copyOf(dirtinessCheckers)));
      }
      handleChangedFiles(
          diffPackageRootsUnderWhichToCheck,
          batchDirtyResult,
          /*numSourceFilesCheckedIfDiffWasMissing=*/ batchDirtyResult.getNumKeysChecked());
      // We use the knowledge gained during the graph scan that just completed. Otherwise, naively,
      // once an external file gets into the Skyframe graph, we'll overly-conservatively always
      // think the graph needs to be scanned.
      externalFilesHelper.setExternalFilesKnowledge(
          tmpExternalFilesHelper.getExternalFilesKnowledge());
    } else if (!externalFilesKnowledge.nonOutputExternalFilesSeen.isEmpty()) {
      logger.atInfo().log(
          "About to scan %d external files",
          externalFilesKnowledge.nonOutputExternalFilesSeen.size());
      FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, syscallCache, fsvcThreads);
      ImmutableBatchDirtyResult batchDirtyResult;
      try (SilentCloseable c = Profiler.instance().profile("fsvc.getDirtyExternalKeys")) {
        Map<SkyKey, SkyValue> externalDirtyNodes = new ConcurrentHashMap<>();
        for (RootedPath path : externalFilesKnowledge.nonOutputExternalFilesSeen) {
          SkyKey key = FileStateValue.key(path);
          SkyValue value = memoizingEvaluator.getExistingValue(key);
          if (value != null) {
            externalDirtyNodes.put(key, value);
          }
          key = DirectoryListingStateValue.key(path);
          value = memoizingEvaluator.getExistingValue(key);
          if (value != null) {
            externalDirtyNodes.put(key, value);
          }
        }
        batchDirtyResult =
            fsvc.getDirtyKeys(
                externalDirtyNodes,
                new ExternalDirtinessChecker(externalFilesHelper, EnumSet.of(FileType.EXTERNAL)));
      }
      handleChangedFiles(
          ImmutableList.of(), batchDirtyResult, batchDirtyResult.getNumKeysChecked());
    }
    for (Pair<Root, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
        pathEntriesWithoutDiffInformation) {
      pair.getSecond().markProcessed();
    }
  }

  private void handleChangedFiles(
      Collection<Root> diffPackageRootsUnderWhichToCheck,
      Differencer.Diff diff,
      int numSourceFilesCheckedIfDiffWasMissing) {
    int numWithoutNewValues = diff.changedKeysWithoutNewValues().size();
    Iterable<SkyKey> keysToBeChangedLaterInThisBuild = diff.changedKeysWithoutNewValues();
    Map<SkyKey, SkyValue> changedKeysWithNewValues = diff.changedKeysWithNewValues();

    logDiffInfo(
        diffPackageRootsUnderWhichToCheck,
        keysToBeChangedLaterInThisBuild,
        numWithoutNewValues,
        changedKeysWithNewValues);

    recordingDiffer.invalidate(keysToBeChangedLaterInThisBuild);
    recordingDiffer.inject(changedKeysWithNewValues);
    modifiedFiles.addAndGet(
        getNumberOfModifiedFiles(keysToBeChangedLaterInThisBuild)
            + getNumberOfModifiedFiles(changedKeysWithNewValues.keySet()));
    numSourceFilesCheckedBecauseOfMissingDiffs += numSourceFilesCheckedIfDiffWasMissing;
    incrementalBuildMonitor.accrue(keysToBeChangedLaterInThisBuild);
    incrementalBuildMonitor.accrue(changedKeysWithNewValues.keySet());
  }

  private static final int MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG = 10;

  private static void logDiffInfo(
      Iterable<Root> pathEntries,
      Iterable<SkyKey> changedWithoutNewValue,
      int numWithoutNewValues,
      Map<SkyKey, ? extends SkyValue> changedWithNewValue) {
    int numModified = changedWithNewValue.size() + numWithoutNewValues;
    StringBuilder result =
        new StringBuilder("DiffAwareness found ")
            .append(numModified)
            .append(" modified source files and directory listings");
    if (!Iterables.isEmpty(pathEntries)) {
      result.append(" for ");
      result.append(Joiner.on(", ").join(pathEntries));
    }

    if (numModified > 0) {
      Iterable<SkyKey> allModifiedKeys =
          Iterables.concat(changedWithoutNewValue, changedWithNewValue.keySet());
      Iterable<SkyKey> trimmed =
          Iterables.limit(allModifiedKeys, MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG);

      result.append(": ").append(Joiner.on(", ").join(trimmed));

      if (numModified > MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG) {
        result.append(", ...");
      }
    }

    logger.atInfo().log("%s", result);
  }

  private static int getNumberOfModifiedFiles(Iterable<SkyKey> modifiedValues) {
    // We are searching only for changed files, DirectoryListingValues don't depend on
    // child values, that's why they are invalidated separately
    return Iterables.size(
        Iterables.filter(modifiedValues, SkyFunctionName.functionIs(FileStateKey.FILE_STATE)));
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
    // or current incrementalStateRetentionStrategy requires the build to have been isolated.
    if (oldValueOfTrackIncrementalState != trackIncrementalState) {
      logger.atInfo().log("Set incremental state to %b", trackIncrementalState);
      evaluatorNeedsReset = true;
    } else if (!trackIncrementalState) {
      evaluatorNeedsReset = true;
    }
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
  protected void invalidateFilesUnderPathForTestingImpl(
      ExtendedEventHandler eventHandler, ModifiedFileSet modifiedFileSet, Root pathEntry)
      throws InterruptedException, AbruptExitException {
    TimestampGranularityMonitor tsgm = this.tsgm.get();
    Differencer.Diff diff;
    if (modifiedFileSet.treatEverythingAsModified()) {
      diff =
          new FilesystemValueChecker(tsgm, syscallCache, /* numThreads= */ 200)
              .getDirtyKeys(memoizingEvaluator.getValues(), new BasicFilesystemDirtinessChecker());
    } else {
      diff = getDiff(tsgm, modifiedFileSet, pathEntry, /* fsvcThreads= */ 200);
    }
    recordingDiffer.invalidate(diff.changedKeysWithoutNewValues());
    recordingDiffer.inject(diff.changedKeysWithNewValues());
    // Blaze invalidates transient errors on every build.
    invalidateTransientErrors();
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
      boolean trustRemoteArtifacts,
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
            trustRemoteArtifacts,
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

  /** Support for aquery output. */
  public void dumpSkyframeState(
      com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump actionGraphDump)
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
          actionGraphDump.dumpConfiguredTarget((RuleConfiguredTargetValue) skyValue);
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
    deleteAnalysisNodes();
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
    memoizingEvaluator.deleteDirty(versionWindowForDirtyGc);
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
