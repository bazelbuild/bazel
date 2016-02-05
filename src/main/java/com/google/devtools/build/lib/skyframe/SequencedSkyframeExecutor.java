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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.BasicFilesystemDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.ExternalDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.MissingDiffDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.UnionDirtinessChecker;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EvaluatorSupplier;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.logging.Logger;

/**
 * A SkyframeExecutor that implicitly assumes that builds can be done incrementally from the most
 * recent build. In other words, builds are "sequenced".
 */
public final class SequencedSkyframeExecutor extends SkyframeExecutor {

  private static final Logger LOG = Logger.getLogger(SequencedSkyframeExecutor.class.getName());

  /** Lower limit for number of loaded packages to consider clearing CT values. */
  private int valueCacheEvictionLimit = -1;

  /** Union of labels of loaded packages since the last eviction of CT values. */
  private Set<PackageIdentifier> allLoadedPackages = ImmutableSet.of();
  private boolean lastAnalysisDiscarded = false;

  // Can only be set once (to false) over the lifetime of this object. If false, the graph will not
  // store edges, saving memory but making incremental builds impossible.
  private boolean keepGraphEdges = true;

  private RecordingDifferencer recordingDiffer;
  private final DiffAwarenessManager diffAwarenessManager;
  private final Iterable<SkyValueDirtinessChecker> customDirtinessCheckers;

  private SequencedSkyframeExecutor(
      EvaluatorSupplier evaluatorSupplier,
      PackageFactory pkgFactory,
      TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      BinTools binTools,
      Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues,
      Iterable<SkyValueDirtinessChecker> customDirtinessCheckers) {
    super(
        evaluatorSupplier,
        pkgFactory,
        tsgm,
        directories,
        binTools,
        workspaceStatusActionFactory,
        buildInfoFactories,
        allowedMissingInputs,
        preprocessorFactorySupplier,
        extraSkyFunctions,
        extraPrecomputedValues,
        false);
    this.diffAwarenessManager = new DiffAwarenessManager(diffAwarenessFactories);
    this.customDirtinessCheckers = customDirtinessCheckers;
  }

  public static SequencedSkyframeExecutor create(
      PackageFactory pkgFactory,
      TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      BinTools binTools,
      Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues,
      Iterable<SkyValueDirtinessChecker> customDirtinessCheckers) {
    SequencedSkyframeExecutor skyframeExecutor =
        new SequencedSkyframeExecutor(
            InMemoryMemoizingEvaluator.SUPPLIER,
            pkgFactory,
            tsgm,
            directories,
            binTools,
            workspaceStatusActionFactory,
            buildInfoFactories,
            diffAwarenessFactories,
            allowedMissingInputs,
            preprocessorFactorySupplier,
            extraSkyFunctions,
            extraPrecomputedValues,
            customDirtinessCheckers);
    skyframeExecutor.init();
    return skyframeExecutor;
  }

  @VisibleForTesting
  public static SequencedSkyframeExecutor create(PackageFactory pkgFactory,
      TimestampGranularityMonitor tsgm, BlazeDirectories directories, BinTools binTools,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories) {
    return create(
        pkgFactory,
        tsgm,
        directories,
        binTools,
        workspaceStatusActionFactory,
        buildInfoFactories,
        diffAwarenessFactories,
        Predicates.<PathFragment>alwaysFalse(),
        Preprocessor.Factory.Supplier.NullSupplier.INSTANCE,
        ImmutableMap.<SkyFunctionName, SkyFunction>of(),
        ImmutableList.<PrecomputedValue.Injected>of(),
        ImmutableList.<SkyValueDirtinessChecker>of());
  }

  @Override
  protected BuildDriver newBuildDriver() {
    return new SequentialBuildDriver(memoizingEvaluator);
  }

  @Override
  protected void init() {
    // Note that we need to set recordingDiffer first since SkyframeExecutor#init calls
    // SkyframeExecutor#evaluatorDiffer.
    recordingDiffer = new RecordingDifferencer();
    super.init();
  }

  @Override
  public void resetEvaluator() {
    super.resetEvaluator();
    diffAwarenessManager.reset();
  }

  @Override
  protected Differencer evaluatorDiffer() {
    return recordingDiffer;
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
  public void sync(EventHandler eventHandler, PackageCacheOptions packageCacheOptions,
      Path outputBase, Path workingDirectory, String defaultsPackageContents, UUID commandId)
          throws InterruptedException, AbruptExitException {
    this.valueCacheEvictionLimit = packageCacheOptions.minLoadedPkgCountForCtNodeEviction;
    super.sync(eventHandler, packageCacheOptions, outputBase, workingDirectory,
        defaultsPackageContents, commandId);
    handleDiffs(eventHandler);
  }

  /**
   * The value types whose builders have direct access to the package locator, rather than accessing
   * it via an explicit Skyframe dependency. They need to be invalidated if the package locator
   * changes.
   */
  private static final Set<SkyFunctionName> PACKAGE_LOCATOR_DEPENDENT_VALUES =
      ImmutableSet.of(
          SkyFunctions.AST_FILE_LOOKUP,
          SkyFunctions.FILE_STATE,
          SkyFunctions.FILE,
          SkyFunctions.DIRECTORY_LISTING_STATE,
          SkyFunctions.TARGET_PATTERN,
          SkyFunctions.PREPARE_DEPS_OF_PATTERN,
          SkyFunctions.WORKSPACE_FILE,
          SkyFunctions.EXTERNAL_PACKAGE);

  @Override
  protected void onNewPackageLocator(PathPackageLocator oldLocator, PathPackageLocator pkgLocator) {
    invalidate(SkyFunctionName.functionIsIn(PACKAGE_LOCATOR_DEPENDENT_VALUES));
  }

  @Override
  protected void invalidate(Predicate<SkyKey> pred) {
    recordingDiffer.invalidate(Iterables.filter(memoizingEvaluator.getValues().keySet(), pred));
  }

  private void invalidateDeletedPackages(Iterable<PackageIdentifier> deletedPackages) {
    ArrayList<SkyKey> packagesToInvalidate = Lists.newArrayList();
    for (PackageIdentifier deletedPackage : deletedPackages) {
      packagesToInvalidate.add(PackageLookupValue.key(deletedPackage));
    }
    recordingDiffer.invalidate(packagesToInvalidate);
  }

  /**
   * Sets the packages that should be treated as deleted and ignored.
   */
  @Override
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void setDeletedPackages(Iterable<PackageIdentifier> pkgs) {
    // Invalidate the old deletedPackages as they may exist now.
    invalidateDeletedPackages(deletedPackages.get());
    deletedPackages.set(ImmutableSet.copyOf(pkgs));
    // Invalidate the new deletedPackages as we need to pretend that they don't exist now.
    invalidateDeletedPackages(deletedPackages.get());
  }

  /**
   * Uses diff awareness on all the package paths to invalidate changed files.
   */
  @VisibleForTesting
  public void handleDiffs(EventHandler eventHandler) throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    modifiedFiles = 0;
    Map<Path, DiffAwarenessManager.ProcessableModifiedFileSet> modifiedFilesByPathEntry =
        Maps.newHashMap();
    Set<Pair<Path, DiffAwarenessManager.ProcessableModifiedFileSet>>
        pathEntriesWithoutDiffInformation = Sets.newHashSet();
    for (Path pathEntry : pkgLocator.get().getPathEntries()) {
      DiffAwarenessManager.ProcessableModifiedFileSet modifiedFileSet =
          diffAwarenessManager.getDiff(eventHandler, pathEntry);
      if (modifiedFileSet.getModifiedFileSet().treatEverythingAsModified()) {
        pathEntriesWithoutDiffInformation.add(Pair.of(pathEntry, modifiedFileSet));
      } else {
        modifiedFilesByPathEntry.put(pathEntry, modifiedFileSet);
      }
    }
    handleDiffsWithCompleteDiffInformation(modifiedFilesByPathEntry);
    handleDiffsWithMissingDiffInformation(eventHandler, pathEntriesWithoutDiffInformation);
  }

  /**
   * Invalidates files under path entries whose corresponding {@link DiffAwareness} gave an exact
   * diff. Removes entries from the given map as they are processed. All of the files need to be
   * invalidated, so the map should be empty upon completion of this function.
   */
  private void handleDiffsWithCompleteDiffInformation(
      Map<Path, DiffAwarenessManager.ProcessableModifiedFileSet> modifiedFilesByPathEntry)
          throws InterruptedException {
    for (Path pathEntry : ImmutableSet.copyOf(modifiedFilesByPathEntry.keySet())) {
      DiffAwarenessManager.ProcessableModifiedFileSet processableModifiedFileSet =
          modifiedFilesByPathEntry.get(pathEntry);
      ModifiedFileSet modifiedFileSet = processableModifiedFileSet.getModifiedFileSet();
      Preconditions.checkState(!modifiedFileSet.treatEverythingAsModified(), pathEntry);
      handleChangedFiles(ImmutableList.of(pathEntry),
          getDiff(modifiedFileSet.modifiedSourceFiles(), pathEntry));
      processableModifiedFileSet.markProcessed();
    }
  }

  /**
   * Finds and invalidates changed files under path entries whose corresponding
   * {@link DiffAwareness} said all files may have been modified.
   */
  private void handleDiffsWithMissingDiffInformation(EventHandler eventHandler,
      Set<Pair<Path, DiffAwarenessManager.ProcessableModifiedFileSet>>
          pathEntriesWithoutDiffInformation) throws InterruptedException {
    if (pathEntriesWithoutDiffInformation.isEmpty()
        && Iterables.isEmpty(customDirtinessCheckers)
        && !externalFilesHelper.isExternalFileSeen()) {
      // Avoid a full graph scan if we have good diff information for all path entries, there are
      // no custom checkers that need to look at the whole graph, and no external (not under any
      // path) files need to be checked.
      return;
    }
    // Before running the FilesystemValueChecker, ensure that all values marked for invalidation
    // have actually been invalidated (recall that invalidation happens at the beginning of the
    // next evaluate() call), because checking those is a waste of time.
    buildDriver.evaluate(ImmutableList.<SkyKey>of(), false,
        DEFAULT_THREAD_COUNT, eventHandler);

    FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, null);
    // We need to manually check for changes to known files. This entails finding all dirty file
    // system values under package roots for which we don't have diff information. If at least
    // one path entry doesn't have diff information, then we're going to have to iterate over
    // the skyframe values at least once no matter what.
    Set<Path> diffPackageRootsUnderWhichToCheck = new HashSet<>();
    for (Pair<Path, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
        pathEntriesWithoutDiffInformation) {
      diffPackageRootsUnderWhichToCheck.add(pair.getFirst());
    }

    Differencer.Diff diff =
        fsvc.getDirtyKeys(
            memoizingEvaluator.getValues(),
            new UnionDirtinessChecker(
                Iterables.concat(
                    customDirtinessCheckers,
                    ImmutableList.<SkyValueDirtinessChecker>of(
                        new ExternalDirtinessChecker(pkgLocator.get()),
                        new MissingDiffDirtinessChecker(diffPackageRootsUnderWhichToCheck)))));
    handleChangedFiles(diffPackageRootsUnderWhichToCheck, diff);

    for (Pair<Path, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
        pathEntriesWithoutDiffInformation) {
      pair.getSecond().markProcessed();
    }
  }

  private void handleChangedFiles(
      Collection<Path> diffPackageRootsUnderWhichToCheck, Differencer.Diff diff) {
    Collection<SkyKey> changedKeysWithoutNewValues = diff.changedKeysWithoutNewValues();
    Map<SkyKey, SkyValue> changedKeysWithNewValues = diff.changedKeysWithNewValues();

    logDiffInfo(diffPackageRootsUnderWhichToCheck, changedKeysWithoutNewValues,
        changedKeysWithNewValues);

    recordingDiffer.invalidate(changedKeysWithoutNewValues);
    recordingDiffer.inject(changedKeysWithNewValues);
    modifiedFiles += getNumberOfModifiedFiles(changedKeysWithoutNewValues);
    modifiedFiles += getNumberOfModifiedFiles(changedKeysWithNewValues.keySet());
    incrementalBuildMonitor.accrue(changedKeysWithoutNewValues);
    incrementalBuildMonitor.accrue(changedKeysWithNewValues.keySet());
  }

  private static void logDiffInfo(Iterable<Path> pathEntries,
                                  Collection<SkyKey> changedWithoutNewValue,
                                  Map<SkyKey, ? extends SkyValue> changedWithNewValue) {
    int numModified = changedWithNewValue.size() + changedWithoutNewValue.size();
    StringBuilder result = new StringBuilder("DiffAwareness found ")
        .append(numModified)
        .append(" modified source files and directory listings for ")
        .append(Joiner.on(", ").join(pathEntries));

    if (numModified > 0) {
      Iterable<SkyKey> allModifiedKeys = Iterables.concat(changedWithoutNewValue,
          changedWithNewValue.keySet());
      Iterable<SkyKey> trimmed = Iterables.limit(allModifiedKeys, 5);

      result.append(": ")
          .append(Joiner.on(", ").join(trimmed));

      if (numModified > 5) {
        result.append(", ...");
      }
    }

    LOG.info(result.toString());
  }

  private static int getNumberOfModifiedFiles(Iterable<SkyKey> modifiedValues) {
    // We are searching only for changed files, DirectoryListingValues don't depend on
    // child values, that's why they are invalidated separately
    return Iterables.size(Iterables.filter(modifiedValues,
        SkyFunctionName.functionIs(SkyFunctions.FILE_STATE)));
  }

  @Override
  public void decideKeepIncrementalState(boolean batch, BuildView.Options viewOptions) {
    Preconditions.checkState(!active);
    if (viewOptions == null) {
      // Some blaze commands don't include the view options. Don't bother with them.
      return;
    }
    if (batch && viewOptions.keepGoing && viewOptions.discardAnalysisCache) {
      Preconditions.checkState(keepGraphEdges, "May only be called once if successful");
      keepGraphEdges = false;
      // Graph will be recreated on next sync.
    }
  }

  @Override
  public boolean hasIncrementalState() {
    // TODO(bazel-team): Combine this method with clearSkyframeRelevantCaches() once legacy
    // execution is removed [skyframe-execution].
    return keepGraphEdges;
  }

  @Override
  public void invalidateFilesUnderPathForTesting(EventHandler eventHandler,
      ModifiedFileSet modifiedFileSet, Path pathEntry) throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    Differencer.Diff diff;
    if (modifiedFileSet.treatEverythingAsModified()) {
      diff = new FilesystemValueChecker(tsgm, null).getDirtyKeys(memoizingEvaluator.getValues(),
          new BasicFilesystemDirtinessChecker());
    } else {
      diff = getDiff(modifiedFileSet.modifiedSourceFiles(), pathEntry);
    }
    syscalls.set(newPerBuildSyscallCache(/*concurrencyLevel=*/42));
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
  protected void invalidateDirtyActions(Iterable<SkyKey> dirtyActionValues) {
    recordingDiffer.invalidate(dirtyActionValues);
  }

  /**
   * Save memory by removing references to configured targets and actions in Skyframe.
   *
   * <p>These values must be recreated on subsequent builds. We do not clear the top-level target
   * values, since their configured targets are needed for the target completion middleman values.
   *
   * <p>The values are not deleted during this method call, because they are needed for the
   * execution phase. Instead, their data is cleared. The next build will delete the values (and
   * recreate them if necessary).
   */
  private void discardAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    try (AutoProfiler p = AutoProfiler.logged("discarding analysis cache", LOG)) {
      lastAnalysisDiscarded = true;
      for (Map.Entry<SkyKey, SkyValue> entry : memoizingEvaluator.getValues().entrySet()) {
        if (!entry.getKey().functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
          continue;
        }
        ConfiguredTargetValue ctValue = (ConfiguredTargetValue) entry.getValue();
        // ctValue may be null if target was not successfully analyzed.
        if (ctValue != null && !topLevelTargets.contains(ctValue.getConfiguredTarget())) {
          ctValue.clear();
        }
      }
    }
  }

  @Override
  public void clearAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    discardAnalysisCache(topLevelTargets);
  }

  @Override
  public void dropConfiguredTargets() {
    skyframeBuildView.clearInvalidatedConfiguredTargets();
    skyframeBuildView.clearLegacyData();
    memoizingEvaluator.delete(
        // We delete any value that can hold an action -- all subclasses of ActionLookupValue -- as
        // well as ActionExecutionValues, since they do not depend on ActionLookupValues.
        SkyFunctionName.functionIsIn(ImmutableSet.of(
            SkyFunctions.CONFIGURED_TARGET,
            SkyFunctions.ACTION_LOOKUP,
            SkyFunctions.BUILD_INFO,
            SkyFunctions.TARGET_COMPLETION,
            SkyFunctions.BUILD_INFO_COLLECTION,
            SkyFunctions.ACTION_EXECUTION))
    );
  }

  /**
   * Deletes all ConfiguredTarget values from the Skyframe cache.
   *
   * <p>After the execution of this method all invalidated and marked for deletion values
   * (and the values depending on them) will be deleted from the cache.
   *
   * <p>WARNING: Note that a call to this method leaves legacy data inconsistent with Skyframe.
   * The next build should clear the legacy caches.
   */
  private void dropConfiguredTargetsNow(final EventHandler eventHandler) {
    dropConfiguredTargets();
    // Run the invalidator to actually delete the values.
    try {
      progressReceiver.ignoreInvalidations = true;
      Uninterruptibles.callUninterruptibly(new Callable<Void>() {
        @Override
        public Void call() throws InterruptedException {
          buildDriver.evaluate(ImmutableList.<SkyKey>of(), false,
              ResourceUsage.getAvailableProcessors(), eventHandler);
          return null;
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);
    } finally {
      progressReceiver.ignoreInvalidations = false;
    }
  }

  /**
   * Returns true if the old set of Packages is a subset or superset of the new one.
   *
   * <p>Compares the names of packages instead of the Package objects themselves (Package doesn't
   * yet override #equals). Since packages store their names as a String rather than a Label, it's
   * easier to use strings here.
   */
  @VisibleForTesting
  static boolean isBuildSubsetOrSupersetOfPreviousBuild(Set<PackageIdentifier> oldPackages,
      Set<PackageIdentifier> newPackages) {
    if (newPackages.size() <= oldPackages.size()) {
      return Sets.difference(newPackages, oldPackages).isEmpty();
    } else if (oldPackages.size() < newPackages.size()) {
      // No need to check for <= here, since the first branch does that already.
      // If size(A) = size(B), then then A\B = 0 iff B\A = 0
      return Sets.difference(oldPackages, newPackages).isEmpty();
    } else {
      return false;
    }
  }

  @Override
  public void updateLoadedPackageSet(Set<PackageIdentifier> loadedPackages) {
    Preconditions.checkState(valueCacheEvictionLimit >= 0,
        "should have called setMinLoadedPkgCountForCtValueEviction earlier");

    // Make a copy to avoid nesting SetView objects. It also computes size(), which we need below.
    Set<PackageIdentifier> union = ImmutableSet.copyOf(
        Sets.union(allLoadedPackages, loadedPackages));

    if (union.size() < valueCacheEvictionLimit
        || isBuildSubsetOrSupersetOfPreviousBuild(allLoadedPackages, loadedPackages)) {
      allLoadedPackages = union;
    } else {
      dropConfiguredTargets();
      allLoadedPackages = loadedPackages;
    }
  }

  @Override
  public void deleteOldNodes(long versionWindowForDirtyGc) {
    // TODO(bazel-team): perhaps we should come up with a separate GC class dedicated to maintaining
    // value garbage. If we ever do so, this logic should be moved there.
    memoizingEvaluator.deleteDirty(versionWindowForDirtyGc);
  }

  @Override
  public void dumpPackages(PrintStream out) {
    Iterable<SkyKey> packageSkyKeys = Iterables.filter(memoizingEvaluator.getValues().keySet(),
        SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE));
    out.println(Iterables.size(packageSkyKeys) + " packages");
    for (SkyKey packageSkyKey : packageSkyKeys) {
      Package pkg = ((PackageValue) memoizingEvaluator.getValues().get(packageSkyKey)).getPackage();
      pkg.dump(out);
    }
  }
}
