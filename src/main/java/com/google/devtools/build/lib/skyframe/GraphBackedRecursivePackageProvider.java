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

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.concurrent.BatchCallback;
import com.google.devtools.build.lib.concurrent.ParallelVisitor.UnusedException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.AbstractRecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Map;
import java.util.Set;

/**
 * A {@link com.google.devtools.build.lib.pkgcache.RecursivePackageProvider} backed by a {@link
 * WalkableGraph}, used by {@code SkyQueryEnvironment} to look up the packages and targets matching
 * the universe that's been preloaded in {@code graph}.
 */
@ThreadSafe
public final class GraphBackedRecursivePackageProvider extends AbstractRecursivePackageProvider {

  private final WalkableGraph graph;
  private final ImmutableList<Root> pkgRoots;
  private final RootPackageExtractor rootPackageExtractor;
  private final ImmutableList<TargetPattern> universeTargetPatterns;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public GraphBackedRecursivePackageProvider(
      WalkableGraph graph,
      ImmutableList<TargetPattern> universeTargetPatterns,
      PathPackageLocator pkgPath,
      RootPackageExtractor rootPackageExtractor) {
    this.graph = Preconditions.checkNotNull(graph);
    this.pkgRoots = pkgPath.getPathEntries();
    this.universeTargetPatterns = Preconditions.checkNotNull(universeTargetPatterns);
    this.rootPackageExtractor = rootPackageExtractor;
  }

  @Override
  public Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException, InterruptedException {
    SkyKey pkgKey = PackageValue.key(packageName);

    PackageValue pkgValue = (PackageValue) graph.getValue(pkgKey);
    if (pkgValue != null) {
      return pkgValue.getPackage();
    }
    NoSuchPackageException nspe = (NoSuchPackageException) graph.getException(pkgKey);
    if (nspe != null) {
      throw nspe;
    }
    if (graph.isCycle(pkgKey)) {
      throw new NoSuchPackageException(packageName, "Package depends on a cycle");
    } else {
      // If the package key does not exist in the graph, then it must not correspond to any package,
      // because the SkyQuery environment has already loaded the universe.
      throw new BuildFileNotFoundException(packageName, "BUILD file not found on package path");
    }
  }

  @Override
  public Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds)
      throws NoSuchPackageException, InterruptedException {
    Set<SkyKey> pkgKeys = ImmutableSet.copyOf(PackageValue.keys(pkgIds));

    ImmutableMap.Builder<PackageIdentifier, Package> pkgResults = ImmutableMap.builder();
    Map<SkyKey, SkyValue> packages = graph.getSuccessfulValues(pkgKeys);
    for (Map.Entry<SkyKey, SkyValue> pkgEntry : packages.entrySet()) {
      PackageIdentifier pkgId = (PackageIdentifier) pkgEntry.getKey().argument();
      PackageValue pkgValue = (PackageValue) pkgEntry.getValue();
      pkgResults.put(pkgId, Preconditions.checkNotNull(pkgValue.getPackage(), pkgId));
    }

    SetView<SkyKey> unknownKeys = Sets.difference(pkgKeys, packages.keySet());
    if (!Iterables.isEmpty(unknownKeys)) {
      logger.atWarning().log(
          "Unable to find %s in the batch lookup of %s. Successfully looked up %s",
          unknownKeys, pkgKeys, packages.keySet());
    }
    for (Map.Entry<SkyKey, Exception> missingOrExceptionEntry :
        graph.getMissingAndExceptions(unknownKeys).entrySet()) {
      PackageIdentifier pkgIdentifier =
          (PackageIdentifier) missingOrExceptionEntry.getKey().argument();
      Exception exception = missingOrExceptionEntry.getValue();
      if (exception == null) {
        // If the package key does not exist in the graph, then it must not correspond to any
        // package, because the SkyQuery environment has already loaded the universe.
        throw new BuildFileNotFoundException(pkgIdentifier, "Package not found");
      }
      Throwables.propagateIfInstanceOf(exception, NoSuchPackageException.class);
      Throwables.propagate(exception);
    }
    return pkgResults.build();
  }

  @Override
  public boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws InterruptedException {
    SkyKey packageLookupKey = PackageLookupValue.key(packageName);
    PackageLookupValue packageLookupValue = (PackageLookupValue) graph.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      // Package lookups can't depend on Skyframe cycles.
      Preconditions.checkState(!graph.isCycle(packageLookupKey), packageLookupKey);
      Exception exception = graph.getException(packageLookupKey);
      if (exception == null) {
        // If the package lookup key does not exist in the graph, then it must not correspond to any
        // package, because the SkyQuery environment has already loaded the universe.
        return false;
      } else {
        if (exception instanceof NoSuchPackageException
            || exception instanceof InconsistentFilesystemException) {
          eventHandler.handle(Event.error(exception.getMessage()));
          return false;
        } else {
          throw new IllegalStateException(
              "During package lookup for '" + packageName + "', got unexpected exception type",
              exception);
        }
      }
    }
    return packageLookupValue.packageExists();
  }

  private ImmutableList<Root> checkValidDirectoryAndGetRoots(
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException {
    if (ignoredSubdirectories.contains(directory) || excludedSubdirectories.contains(directory)) {
      return ImmutableList.of();
    }

    // Check that this package is covered by at least one of our universe patterns.
    boolean inUniverse = false;
    for (TargetPattern pattern : universeTargetPatterns) {
      if (!pattern.getType().equals(TargetPattern.Type.TARGETS_BELOW_DIRECTORY)) {
        continue;
      }
      PackageIdentifier packageIdentifier = PackageIdentifier.create(repository, directory);
      if (((TargetsBelowDirectory) pattern)
          .containsAllTransitiveSubdirectories(packageIdentifier)) {
        inUniverse = true;
        break;
      }
    }

    if (!inUniverse) {
      return ImmutableList.of();
    }

    if (repository.isMain()) {
      return pkgRoots;
    } else {
      RepositoryDirectoryValue repositoryValue =
          (RepositoryDirectoryValue) graph.getValue(RepositoryDirectoryValue.key(repository));
      if (repositoryValue == null || !repositoryValue.repositoryExists()) {
        // If this key doesn't exist, the repository is outside the universe, so we return
        // "nothing".
        return ImmutableList.of();
      }
      return ImmutableList.of(Root.fromPath(repositoryValue.getPath()));
    }
  }

  @Override
  public void streamPackagesUnderDirectory(
      BatchCallback<PackageIdentifier, UnusedException> results,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException {
    ImmutableList<Root> roots =
        checkValidDirectoryAndGetRoots(
            repository, directory, ignoredSubdirectories, excludedSubdirectories);

    rootPackageExtractor.streamPackagesFromRoots(
        results,
        graph,
        roots,
        eventHandler,
        repository,
        directory,
        ignoredSubdirectories,
        excludedSubdirectories);
  }
}
