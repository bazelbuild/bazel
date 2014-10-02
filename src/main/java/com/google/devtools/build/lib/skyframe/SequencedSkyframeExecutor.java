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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BuildView;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.PrintStream;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;

/**
 * A SkyframeExecutor that implicitly assumes that builds can be done incrementally from the most
 * recent build. In other words, builds are "sequenced".
 */
public final class SequencedSkyframeExecutor extends SkyframeExecutor {
  /** Lower limit for number of loaded packages to consider clearing CT values. */
  private int valueCacheEvictionLimit = -1;

  /** Union of labels of loaded packages since the last eviction of CT values. */
  private Set<PathFragment> allLoadedPackages = ImmutableSet.of();

  private boolean lastAnalysisDiscarded = false;

  // Can only be set once (to false) over the lifetime of this object. If false, the graph will not
  // store edges, saving memory but making incremental builds impossible.
  private boolean keepGraphEdges = true;

  public SequencedSkyframeExecutor(Reporter reporter, PackageFactory pkgFactory,
      boolean skyframeBuild, TimestampGranularityMonitor tsgm, BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier, Clock clock) {
    super(reporter, InMemoryMemoizingEvaluator.SUPPLIER, pkgFactory, skyframeBuild, tsgm,
        directories, workspaceStatusActionFactory, buildInfoFactories, diffAwarenessFactories,
        allowedMissingInputs, preprocessorFactorySupplier, clock);
  }

  @VisibleForTesting
  public SequencedSkyframeExecutor(Reporter reporter, PackageFactory pkgFactory,
      TimestampGranularityMonitor tsgm, BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories) {
    this(reporter, pkgFactory, true, tsgm, directories, workspaceStatusActionFactory,
        buildInfoFactories, diffAwarenessFactories, Predicates.<PathFragment>alwaysFalse(),
        Preprocessor.Factory.Supplier.NullSupplier.INSTANCE, BlazeClock.instance());
  }

  @Override
  public void resetEvaluator() {
    resetEvaluatorInternal(/*bootstrapping=*/false);
  }

  public void sync(PackageCacheOptions packageCacheOptions, Path workingDirectory,
                   String defaultsPackageContents, UUID commandId) throws InterruptedException {
    this.valueCacheEvictionLimit = packageCacheOptions.minLoadedPkgCountForCtNodeEviction;
    super.sync(packageCacheOptions, workingDirectory, defaultsPackageContents, commandId);
  }

  @Override
  public void handleDiffs() throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }
    super.handleDiffs();
  }

  @Override
  public void decideKeepIncrementalState(boolean batch, BuildView.Options viewOptions) {
    Preconditions.checkState(!active);
    if (viewOptions == null) {
      // Some blaze commands don't include the view options. Don't bother with them.
      return;
    }
    if (skyframeBuild && batch && viewOptions.keepGoing && viewOptions.discardAnalysisCache) {
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
  public void invalidateFilesUnderPathForTesting(ModifiedFileSet modifiedFileSet, Path pathEntry)
      throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }
    super.invalidateFilesUnderPathForTesting(modifiedFileSet, pathEntry);
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

  @Override
  public void clearAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    if (!skyframeBuild()) {
      dropConfiguredTargetsNow();
    } else {
      discardAnalysisCache(topLevelTargets);
    }
  }

  @Override
  public void dropConfiguredTargets() {
    if (skyframeBuildView != null) {
      skyframeBuildView.clearInvalidatedConfiguredTargets();
    }
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
  private void dropConfiguredTargetsNow() {
    dropConfiguredTargets();
    // Run the invalidator to actually delete the values.
    try {
      progressReceiver.ignoreInvalidations = true;
      callUninterruptibly(new Callable<Void>() {
        @Override
        public Void call() throws InterruptedException {
          buildDriver.evaluate(ImmutableList.<SkyKey>of(), false,
              ResourceUsage.getAvailableProcessors(), reporter);
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
  static boolean isBuildSubsetOrSupersetOfPreviousBuild(Set<PathFragment> oldPackages,
      Set<PathFragment> newPackages) {
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
  public void updateLoadedPackageSet(Set<PathFragment> loadedPackages) {
    Preconditions.checkState(valueCacheEvictionLimit >= 0,
        "should have called setMinLoadedPkgCountForCtValueEviction earlier");

    // Make a copy to avoid nesting SetView objects. It also computes size(), which we need below.
    Set<PathFragment> union = ImmutableSet.copyOf(Sets.union(allLoadedPackages, loadedPackages));

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
