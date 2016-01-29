// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Objects;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A {@link RecursivePackageProvider} backed by a {@link WalkableGraph}, used by {@code
 * SkyQueryEnvironment} to look up the packages and targets matching the universe that's been
 * preloaded in {@code graph}.
 */
@ThreadSafe
public final class GraphBackedRecursivePackageProvider implements RecursivePackageProvider {

  private final WalkableGraph graph;
  private final PathPackageLocator pkgPath;
  private final ImmutableList<TargetPatternKey> universeTargetPatternKeys;

  public GraphBackedRecursivePackageProvider(WalkableGraph graph,
      ImmutableList<TargetPatternKey> universeTargetPatternKeys,
      PathPackageLocator pkgPath) {
    this.pkgPath = pkgPath;
    this.graph = Preconditions.checkNotNull(graph);
    this.universeTargetPatternKeys = Preconditions.checkNotNull(universeTargetPatternKeys);
  }

  @Override
  public Package getPackage(EventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException {
    SkyKey pkgKey = PackageValue.key(packageName);

    PackageValue pkgValue;
    if (graph.exists(pkgKey)) {
      pkgValue = (PackageValue) graph.getValue(pkgKey);
      if (pkgValue == null) {
        throw (NoSuchPackageException)
            Preconditions.checkNotNull(graph.getException(pkgKey), pkgKey);
      }
    } else {
      // If the package key does not exist in the graph, then it must not correspond to any package,
      // because the SkyQuery environment has already loaded the universe.
      throw new BuildFileNotFoundException(packageName, "BUILD file not found on package path");
    }
    return pkgValue.getPackage();
  }

  @Override
  public Map<PackageIdentifier, Package> bulkGetPackages(EventHandler eventHandler,
      Iterable<PackageIdentifier> pkgIds) throws NoSuchPackageException {
    Set<SkyKey> pkgKeys = ImmutableSet.copyOf(PackageValue.keys(pkgIds));

    ImmutableMap.Builder<PackageIdentifier, Package> pkgResults = ImmutableMap.builder();
    Map<SkyKey, SkyValue> packages = graph.getSuccessfulValues(pkgKeys);
    for (PackageIdentifier pkgId : pkgIds) {
      PackageValue pkgValue = (PackageValue) packages.get(PackageValue.key(pkgId));
      pkgResults.put(pkgId, Preconditions.checkNotNull(pkgValue.getPackage(), pkgId));
    }

    SetView<SkyKey> unknownKeys = Sets.difference(pkgKeys, packages.keySet());
    for (Map.Entry<SkyKey, Exception> missingOrExceptionEntry :
        graph.getMissingAndExceptions(unknownKeys).entrySet()) {
      PackageIdentifier pkgIdentifier =
          (PackageIdentifier) missingOrExceptionEntry.getKey().argument();
      Exception exception = missingOrExceptionEntry.getValue();
      if (exception == null) {
        // If the package key does not exist in the graph, then it must not correspond to any
        // package, because the SkyQuery environment has already loaded the universe.
        throw new BuildFileNotFoundException(pkgIdentifier, "BUILD file not found on package path");
      }
      Throwables.propagateIfInstanceOf(exception, NoSuchPackageException.class);
      Throwables.propagate(exception);
    }
    return pkgResults.build();
  }


  @Override
  public boolean isPackage(EventHandler eventHandler, PackageIdentifier packageName) {
    SkyKey packageLookupKey = PackageLookupValue.key(packageName);
    if (!graph.exists(packageLookupKey)) {
      // If the package lookup key does not exist in the graph, then it must not correspond to any
      // package, because the SkyQuery environment has already loaded the universe.
      return false;
    }
    PackageLookupValue packageLookupValue = (PackageLookupValue) graph.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      Exception exception = Preconditions.checkNotNull(graph.getException(packageLookupKey),
          "During package lookup for '%s', got null for exception", packageName);
      if (exception instanceof NoSuchPackageException
          || exception instanceof InconsistentFilesystemException) {
        eventHandler.handle(Event.error(exception.getMessage()));
        return false;
      } else {
        throw new IllegalStateException("During package lookup for '" + packageName
            + "', got unexpected exception type", exception);
      }
    }
    return packageLookupValue.packageExists();
  }

  @Override
  public Iterable<PathFragment> getPackagesUnderDirectory(
      RepositoryName repository, PathFragment directory,
      ImmutableSet<PathFragment> excludedSubdirectories) {
    PathFragment.checkAllPathsAreUnder(excludedSubdirectories, directory);

    // Check that this package is covered by at least one of our universe patterns.
    boolean inUniverse = false;
    for (TargetPatternKey patternKey : universeTargetPatternKeys) {
      TargetPattern pattern = patternKey.getParsedPattern();
      boolean isTBD = pattern.getType().equals(Type.TARGETS_BELOW_DIRECTORY);
      PackageIdentifier packageIdentifier = PackageIdentifier.create(
          repository, directory);
      if (isTBD && pattern.containsBelowDirectory(packageIdentifier)) {
        inUniverse = true;
        break;
      }
    }

    if (!inUniverse) {
      return ImmutableList.of();
    }

    List<Path> roots = new ArrayList<>();
    if (repository.isDefault()) {
      roots.addAll(pkgPath.getPathEntries());
    } else {
      RepositoryDirectoryValue repositoryValue =
            (RepositoryDirectoryValue) graph.getValue(RepositoryDirectoryValue.key(repository));
      if (repositoryValue == null) {
        // If this key doesn't exist, the repository is outside the universe, so we return
        // "nothing".
        return ImmutableList.of();
      }
      roots.add(repositoryValue.getPath());
    }

    // If we found a TargetsBelowDirectory pattern in the universe that contains this directory,
    // then we can look for packages in and under it in the graph. If we didn't find one, then the
    // directory wasn't in the universe, so return an empty list.
    ImmutableList.Builder<PathFragment> builder = ImmutableList.builder();
    for (Path root : roots) {
      RootedPath rootedDir = RootedPath.toRootedPath(root, directory);
      TraversalInfo info = new TraversalInfo(rootedDir, excludedSubdirectories);
      collectPackagesUnder(repository, ImmutableSet.of(info), builder);
    }
    return builder.build();
  }

  private void collectPackagesUnder(
      final RepositoryName repository,
      Set<TraversalInfo> traversals,
      ImmutableList.Builder<PathFragment> builder) {
    Map<TraversalInfo, SkyKey> traversalToKeyMap =
        Maps.asMap(
            traversals,
            new Function<TraversalInfo, SkyKey>() {
              @Override
              public SkyKey apply(TraversalInfo traversalInfo) {
                return CollectPackagesUnderDirectoryValue.key(
                    repository, traversalInfo.rootedDir, traversalInfo.excludedSubdirectories);
              }
            });
    Map<SkyKey, SkyValue> values = graph.getSuccessfulValues(traversalToKeyMap.values());

    ImmutableSet.Builder<TraversalInfo> subdirTraversalBuilder = ImmutableSet.builder();
    for (Map.Entry<TraversalInfo, SkyKey> entry : traversalToKeyMap.entrySet()) {
      TraversalInfo info = entry.getKey();
      SkyKey key = entry.getValue();
      SkyValue val = values.get(key);
      CollectPackagesUnderDirectoryValue collectPackagesValue =
          (CollectPackagesUnderDirectoryValue) val;
      if (collectPackagesValue != null) {
        if (collectPackagesValue.isDirectoryPackage()) {
          builder.add(info.rootedDir.getRelativePath());
        }

        ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages =
            collectPackagesValue.getSubdirectoryTransitivelyContainsPackages();
        for (RootedPath subdirectory : subdirectoryTransitivelyContainsPackages.keySet()) {
          if (subdirectoryTransitivelyContainsPackages.get(subdirectory)) {
            PathFragment subdirectoryRelativePath = subdirectory.getRelativePath();
            ImmutableSet<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
                PathFragment.filterPathsStartingWith(
                    info.excludedSubdirectories, subdirectoryRelativePath);
            subdirTraversalBuilder.add(
                new TraversalInfo(subdirectory, excludedSubdirectoriesBeneathThisSubdirectory));
          }
        }
      }
    }

    ImmutableSet<TraversalInfo> subdirTraversals = subdirTraversalBuilder.build();
    if (!subdirTraversals.isEmpty()) {
      collectPackagesUnder(repository, subdirTraversals, builder);
    }
  }

  @Override
  public Target getTarget(EventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException {
    return getPackage(eventHandler, label.getPackageIdentifier()).getTarget(label.getName());
  }

  private static final class TraversalInfo {
    private final RootedPath rootedDir;
    private final ImmutableSet<PathFragment> excludedSubdirectories;

    private TraversalInfo(RootedPath rootedDir, ImmutableSet<PathFragment> excludedSubdirectories) {
      this.rootedDir = rootedDir;
      this.excludedSubdirectories = excludedSubdirectories;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(rootedDir, excludedSubdirectories);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj instanceof TraversalInfo) {
        TraversalInfo otherTraversal = (TraversalInfo) obj;
        return Objects.equal(rootedDir, otherTraversal.rootedDir)
            && Objects.equal(excludedSubdirectories, otherTraversal.excludedSubdirectories);
      }
      return false;
    }
  }
}
