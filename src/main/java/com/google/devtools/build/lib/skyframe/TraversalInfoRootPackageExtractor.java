// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver.MAX_PACKAGES_BULK_GET;

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ParallelVisitor;
import com.google.devtools.build.lib.cmdline.QueryExceptionMarkerInterface;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Looks up values under {@link TraversalInfo}s of given roots in a {@link WalkableGraph}. */
public class TraversalInfoRootPackageExtractor implements RootPackageExtractor {

  private static final Comparator<TraversalInfo> TRAVERSAL_INFO_COMPARATOR =
      Comparator.comparing(ti -> ti.rootedDir.getRootRelativePath());

  private static final int PACKAGE_ID_OUTPUT_BATCH_SIZE = 100;
  private static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

  @Override
  public void streamPackagesFromRoots(
      SafeBatchCallback<PackageIdentifier> results,
      WalkableGraph graph,
      List<Root> roots,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> forbiddenSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException {
    TreeSet<TraversalInfo> dirsToCheckForPackages = new TreeSet<>(TRAVERSAL_INFO_COMPARATOR);
    for (Root root : roots) {
      RootedPath rootedDir = RootedPath.toRootedPath(root, directory);
      dirsToCheckForPackages.add(
          new TraversalInfo(rootedDir, forbiddenSubdirectories, excludedSubdirectories));
    }
    PackageCollectingParallelVisitor visitor =
        new PackageCollectingParallelVisitor(
            results,
            /*visitBatchSize=*/ MAX_PACKAGES_BULK_GET,
            /*processResultsBatchSize=*/ PACKAGE_ID_OUTPUT_BATCH_SIZE,
            /*minPendingTasks=*/ 3 * DEFAULT_THREAD_COUNT,
            /*resultBatchSize=*/ PACKAGE_ID_OUTPUT_BATCH_SIZE,
            eventHandler,
            repository,
            graph);
    visitor.visitAndWaitForCompletion(dirsToCheckForPackages);
  }

  private static final ExecutorService PACKAGE_ID_COLLECTING_EXECUTOR =
      Executors.newFixedThreadPool(
          /*numThreads=*/ DEFAULT_THREAD_COUNT,
          new ThreadFactoryBuilder().setNameFormat("package-id-traversal-%d").build());

  /**
   * A ParallelVisitor that reports every {@link PackageIdentifier} by querying the WalkableGraph
   * for a {@link CollectPackagesUnderDirectoryValue} for each {@link TraversalInfo} it visits.
   */
  static class PackageCollectingParallelVisitor
      extends ParallelVisitor<
          TraversalInfo,
          TraversalInfo,
          TraversalInfo,
          PackageIdentifier,
          QueryExceptionMarkerInterface.MarkerRuntimeException,
          SafeBatchCallback<PackageIdentifier>> {

    private final ExtendedEventHandler eventHandler;
    private final RepositoryName repository;
    private final WalkableGraph graph;

    PackageCollectingParallelVisitor(
        SafeBatchCallback<PackageIdentifier> callback,
        int visitBatchSize,
        int processResultsBatchSize,
        int minPendingTasks,
        int resultBatchSize,
        ExtendedEventHandler eventHandler,
        RepositoryName repository,
        WalkableGraph graph) {
      super(
          callback,
          QueryExceptionMarkerInterface.MarkerRuntimeException.class,
          visitBatchSize,
          processResultsBatchSize,
          minPendingTasks,
          resultBatchSize,
          PACKAGE_ID_COLLECTING_EXECUTOR,
          VisitTaskStatusCallback.NULL_INSTANCE);
      this.eventHandler = eventHandler;
      this.repository = repository;
      this.graph = graph;
    }

    @Override
    protected Iterable<PackageIdentifier> outputKeysToOutputValues(
        Iterable<TraversalInfo> targetKeys) {
      ImmutableList.Builder<PackageIdentifier> results =
          ImmutableList.builderWithExpectedSize(resultBatchSize);
      for (TraversalInfo resultInfo : targetKeys) {
        results.add(
            PackageIdentifier.create(repository, resultInfo.rootedDir.getRootRelativePath()));
      }
      return results.build();
    }

    @Override
    protected Visit getVisitResult(Iterable<TraversalInfo> dirsToCheckForPackages)
        throws InterruptedException {
      ImmutableMap.Builder<TraversalInfo, SkyKey> traversalToKeyMapBuilder = ImmutableMap.builder();
      for (TraversalInfo traversalInfo : dirsToCheckForPackages) {
        traversalToKeyMapBuilder.put(
            traversalInfo,
            CollectPackagesUnderDirectoryValue.key(
                repository, traversalInfo.rootedDir, traversalInfo.forbiddenSubdirectories));
      }
      ImmutableMap<TraversalInfo, SkyKey> traversalToKeyMap =
          traversalToKeyMapBuilder.buildOrThrow();
      Map<SkyKey, SkyValue> values = graph.getSuccessfulValues(traversalToKeyMap.values());

      // NOTE: Use a TreeSet to ensure a deterministic (sorted) iteration order when we recurse.
      List<TraversalInfo> resultPackageIds = new ArrayList<>();
      TreeSet<TraversalInfo> subdirsToCheckForPackages = new TreeSet<>(TRAVERSAL_INFO_COMPARATOR);
      for (Map.Entry<TraversalInfo, SkyKey> entry : traversalToKeyMap.entrySet()) {
        TraversalInfo info = entry.getKey();
        SkyKey key = entry.getValue();
        SkyValue val = values.get(key);
        CollectPackagesUnderDirectoryValue collectPackagesValue =
            (CollectPackagesUnderDirectoryValue) val;
        if (collectPackagesValue != null) {
          if (collectPackagesValue.isDirectoryPackage()) {
            resultPackageIds.add(info);
          }

          if (collectPackagesValue.getErrorMessage() != null) {
            eventHandler.handle(Event.error(collectPackagesValue.getErrorMessage()));
          }

          ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages =
              collectPackagesValue.getSubdirectoryTransitivelyContainsPackagesOrErrors();
          for (RootedPath subdirectory : subdirectoryTransitivelyContainsPackages.keySet()) {
            if (subdirectoryTransitivelyContainsPackages.get(subdirectory)) {
              PathFragment subdirectoryRelativePath = subdirectory.getRootRelativePath();
              ImmutableSet<PathFragment> forbiddenSubdirectoriesBeneathThisSubdirectory =
                  info.forbiddenSubdirectories.stream()
                      .filter(pathFragment -> pathFragment.startsWith(subdirectoryRelativePath))
                      .collect(toImmutableSet());
              ImmutableSet<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
                  info.excludedSubdirectories.stream()
                      .filter(pathFragment -> pathFragment.startsWith(subdirectoryRelativePath))
                      .collect(toImmutableSet());
              if (!excludedSubdirectoriesBeneathThisSubdirectory.contains(
                  subdirectoryRelativePath)) {
                subdirsToCheckForPackages.add(
                    new TraversalInfo(
                        subdirectory,
                        forbiddenSubdirectoriesBeneathThisSubdirectory,
                        excludedSubdirectoriesBeneathThisSubdirectory));
              }
            }
          }
        }
      }
      return new Visit(
          /*keysToUseForResult=*/ resultPackageIds, /*keysToVisit=*/ subdirsToCheckForPackages);
    }

    @Override
    protected Iterable<TraversalInfo> preprocessInitialVisit(Iterable<TraversalInfo> infos) {
      return infos;
    }

    @Override
    protected Iterable<TraversalInfo> noteAndReturnUniqueVisitationKeys(
        Iterable<TraversalInfo> prospectiveVisitationKeys) {
      return prospectiveVisitationKeys;
    }
  }

  /** Value type used as visitation and output key for {@link PackageCollectingParallelVisitor}. */
  private static final class TraversalInfo {
    final RootedPath rootedDir;
    // Set of forbidden directories. The graph is assumed to be prepopulated with
    // CollectPackagesUnderDirectoryValue nodes whose keys have forbidden packages embedded in
    // them. Therefore, we need to be careful to request and use the same sort of keys here in our
    // traversal.
    final ImmutableSet<PathFragment> forbiddenSubdirectories;
    // Set of directories, targets under which should be excluded from the traversal results.
    // Excluded directory information isn't part of the graph keys in the prepopulated graph, so we
    // need to perform the filtering ourselves.
    final ImmutableSet<PathFragment> excludedSubdirectories;

    private TraversalInfo(
        RootedPath rootedDir,
        ImmutableSet<PathFragment> forbiddenSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories) {
      this.rootedDir = rootedDir;
      this.forbiddenSubdirectories = forbiddenSubdirectories;
      this.excludedSubdirectories = excludedSubdirectories;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(rootedDir, forbiddenSubdirectories, excludedSubdirectories);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj instanceof TraversalInfo otherTraversal) {
        return Objects.equal(rootedDir, otherTraversal.rootedDir)
            && Objects.equal(forbiddenSubdirectories, otherTraversal.forbiddenSubdirectories)
            && Objects.equal(excludedSubdirectories, otherTraversal.excludedSubdirectories);
      }
      return false;
    }
  }
}
