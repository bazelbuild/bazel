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
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.AbstractRecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Map;

/**
 * A {@link com.google.devtools.build.lib.pkgcache.RecursivePackageProvider} backed by a {@link
 * WalkableGraph}, used by {@code SkyQueryEnvironment} to look up the packages and targets matching
 * the universe that's been preloaded in {@code graph}.
 */
@ThreadSafe
public final class GraphBackedRecursivePackageProvider extends AbstractRecursivePackageProvider {

  /**
   * Helper interface for clients of GraphBackedRecursivePackageProvider to indicate what universe
   * packages should be resolved in.
   *
   * <p>Client can either specify a fixed set of target patterns (using {@link #of}), or specify
   * that all targets are valid (using {@link #all}).
   */
  public interface UniverseTargetPattern {
    ImmutableList<TargetPattern> patterns();

    boolean allowAll();

    static UniverseTargetPattern of(ImmutableList<TargetPattern> patterns) {
      return new UniverseTargetPattern() {
        @Override
        public ImmutableList<TargetPattern> patterns() {
          return patterns;
        }

        @Override
        public boolean allowAll() {
          return false;
        }
      };
    }

    static UniverseTargetPattern all() {
      return new UniverseTargetPattern() {
        @Override
        public ImmutableList<TargetPattern> patterns() {
          return ImmutableList.of();
        }

        @Override
        public boolean allowAll() {
          return true;
        }
      };
    }
  }

  private final WalkableGraph graph;
  private final ImmutableList<Root> pkgRoots;
  private final RootPackageExtractor rootPackageExtractor;
  private final UniverseTargetPattern universeTargetPatterns;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public GraphBackedRecursivePackageProvider(
      WalkableGraph graph,
      UniverseTargetPattern universeTargetPatterns,
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
    PackageValue pkgValue = (PackageValue) graph.getValue(packageName);
    if (pkgValue != null) {
      return pkgValue.getPackage();
    }
    NoSuchPackageException nspe = (NoSuchPackageException) graph.getException(packageName);
    if (nspe != null) {
      throw nspe;
    }
    if (graph.isCycle(packageName)) {
      throw new NoSuchPackageException(packageName, "Package depends on a cycle");
    } else {
      // If the package key does not exist in the graph, then it must not correspond to any package,
      // because the SkyQuery environment has already loaded the universe.
      throw new BuildFileNotFoundException(packageName, "BUILD file not found on package path");
    }
  }

  @Override
  public ImmutableMap<PackageIdentifier, Package> bulkGetPackages(
      Iterable<PackageIdentifier> pkgIds) throws NoSuchPackageException, InterruptedException {
    ImmutableSet<SkyKey> pkgKeys = ImmutableSet.copyOf(pkgIds);

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
    return pkgResults.buildOrThrow();
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
        if (exception instanceof NoSuchPackageException) {
          eventHandler.handle(Event.error(exception.getMessage()));
          return false;
        } else {
          // InconsistentFilesystemException can theoretically be thrown by PackageLookupFunction.
          // However, such exceptions are catastrophic. If we evaluated this PackageLookupFunction
          // immediately prior to doing the current graph traversal, we should have already failed
          // catastrophically. On the other hand, if PackageLookupFunction was evaluated on a
          // previous evaluation, it would not have been committed to the graph, since a
          // catastrophe triggers error bubbling, which does not commit nodes to the graph.
          throw new IllegalStateException(
              "During package lookup for '" + packageName + "', got unexpected exception type",
              exception);
        }
      }
    }
    return packageLookupValue.packageExists();
  }

  private ImmutableList<Root> checkValidDirectoryAndGetRoots(
      RepositoryName repository, PathFragment directory) throws InterruptedException {

    // Check that this package is covered by at least one of our universe patterns.
    boolean inUniverse = false;
    if (universeTargetPatterns.allowAll()) {
      inUniverse = true;
    } else {
      for (TargetPattern pattern : universeTargetPatterns.patterns()) {
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
      SafeBatchCallback<PackageIdentifier> results,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException, QueryException {
    rootPackageExtractor.streamPackagesFromRoots(
        results,
        graph,
        checkValidDirectoryAndGetRoots(repository, directory),
        eventHandler,
        repository,
        directory,
        ignoredSubdirectories,
        excludedSubdirectories);
  }
}
