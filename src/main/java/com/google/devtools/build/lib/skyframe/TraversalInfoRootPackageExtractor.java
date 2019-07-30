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

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.Consumer;

/** Looks up values under {@link TraversalInfo}s of given roots in a {@link WalkableGraph}. */
public class TraversalInfoRootPackageExtractor implements RootPackageExtractor {

  private static final Comparator<TraversalInfo> TRAVERSAL_INFO_COMPARATOR =
      Comparator.comparing(ti -> ti.rootedDir.getRootRelativePath());

  @Override
  public void streamPackagesFromRoots(
      Consumer<PathFragment> results,
      WalkableGraph graph,
      List<Root> roots,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException {
    for (Root root : roots) {
      RootedPath rootedDir = RootedPath.toRootedPath(root, directory);
      TraversalInfo info =
          new TraversalInfo(rootedDir, blacklistedSubdirectories, excludedSubdirectories);
      TreeSet<TraversalInfo> dirsToCheckForPackages = new TreeSet<>(TRAVERSAL_INFO_COMPARATOR);
      dirsToCheckForPackages.add(info);
      collectPackagesUnder(results, graph, eventHandler, repository, dirsToCheckForPackages);
    }
  }

  private void collectPackagesUnder(
      Consumer<PathFragment> results,
      WalkableGraph graph,
      ExtendedEventHandler eventHandler,
      final RepositoryName repository,
      TreeSet<TraversalInfo> dirsToCheckForPackages)
      throws InterruptedException {
    // NOTE: Maps.asMap returns a Map<T> view whose entrySet() order matches the underlying Set<T>.
    Map<TraversalInfo, SkyKey> traversalToKeyMap =
        Maps.asMap(
            dirsToCheckForPackages,
            traversalInfo ->
                CollectPackagesUnderDirectoryValue.key(
                    repository, traversalInfo.rootedDir, traversalInfo.blacklistedSubdirectories));
    Map<SkyKey, SkyValue> values = graph.getSuccessfulValues(traversalToKeyMap.values());

    // NOTE: Use a TreeSet to ensure a deterministic (sorted) iteration order when we recurse.
    TreeSet<TraversalInfo> subdirsToCheckForPackages = new TreeSet<>(TRAVERSAL_INFO_COMPARATOR);
    for (Map.Entry<TraversalInfo, SkyKey> entry : traversalToKeyMap.entrySet()) {
      TraversalInfo info = entry.getKey();
      SkyKey key = entry.getValue();
      SkyValue val = values.get(key);
      CollectPackagesUnderDirectoryValue collectPackagesValue =
          (CollectPackagesUnderDirectoryValue) val;
      if (collectPackagesValue != null) {
        if (collectPackagesValue.isDirectoryPackage()) {
          results.accept(info.rootedDir.getRootRelativePath());
        }

        if (collectPackagesValue.getErrorMessage() != null) {
          eventHandler.handle(Event.error(collectPackagesValue.getErrorMessage()));
        }

        ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages =
            collectPackagesValue.getSubdirectoryTransitivelyContainsPackagesOrErrors();
        for (RootedPath subdirectory : subdirectoryTransitivelyContainsPackages.keySet()) {
          if (subdirectoryTransitivelyContainsPackages.get(subdirectory)) {
            PathFragment subdirectoryRelativePath = subdirectory.getRootRelativePath();
            ImmutableSet<PathFragment> blacklistedSubdirectoriesBeneathThisSubdirectory =
                info.blacklistedSubdirectories
                    .stream()
                    .filter(pathFragment -> pathFragment.startsWith(subdirectoryRelativePath))
                    .collect(toImmutableSet());
            ImmutableSet<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
                info.excludedSubdirectories
                    .stream()
                    .filter(pathFragment -> pathFragment.startsWith(subdirectoryRelativePath))
                    .collect(toImmutableSet());
            if (!excludedSubdirectoriesBeneathThisSubdirectory.contains(subdirectoryRelativePath)) {
              subdirsToCheckForPackages.add(
                  new TraversalInfo(
                      subdirectory,
                      blacklistedSubdirectoriesBeneathThisSubdirectory,
                      excludedSubdirectoriesBeneathThisSubdirectory));
            }
          }
        }
      }
    }

    if (!subdirsToCheckForPackages.isEmpty()) {
      collectPackagesUnder(results, graph, eventHandler, repository, subdirsToCheckForPackages);
    }
  }

  private static final class TraversalInfo {
    final RootedPath rootedDir;
    // Set of blacklisted directories. The graph is assumed to be prepopulated with
    // CollectPackagesUnderDirectoryValue nodes whose keys have blacklisted packages embedded in
    // them. Therefore, we need to be careful to request and use the same sort of keys here in our
    // traversal.
    final ImmutableSet<PathFragment> blacklistedSubdirectories;
    // Set of directories, targets under which should be excluded from the traversal results.
    // Excluded directory information isn't part of the graph keys in the prepopulated graph, so we
    // need to perform the filtering ourselves.
    final ImmutableSet<PathFragment> excludedSubdirectories;

    private TraversalInfo(
        RootedPath rootedDir,
        ImmutableSet<PathFragment> blacklistedSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories) {
      this.rootedDir = rootedDir;
      this.blacklistedSubdirectories = blacklistedSubdirectories;
      this.excludedSubdirectories = excludedSubdirectories;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(rootedDir, blacklistedSubdirectories, excludedSubdirectories);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj instanceof TraversalInfo) {
        TraversalInfo otherTraversal = (TraversalInfo) obj;
        return Objects.equal(rootedDir, otherTraversal.rootedDir)
            && Objects.equal(blacklistedSubdirectories, otherTraversal.blacklistedSubdirectories)
            && Objects.equal(excludedSubdirectories, otherTraversal.excludedSubdirectories);
      }
      return false;
    }
  }
}
