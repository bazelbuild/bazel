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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.io.ProcessPackageDirectoryException;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.AbstractRecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * A {@link com.google.devtools.build.lib.pkgcache.RecursivePackageProvider} backed by an {@link
 * Environment}. Its methods may throw {@link MissingDepException} if the package values this
 * depends on haven't been calculated and added to its environment.
 *
 * <p>This implementation never emits events through the {@link ExtendedEventHandler}s passed to its
 * methods. Instead, it emits events through its environment's {@link Environment#getListener()}.
 *
 * <p>This implementation suppresses most {@link NoSuchPackageException}s discovered during package
 * loading, since target pattern expansion may tolerate point failures in packages. The first one
 * found is stored so that inside a nokeep-going build it can be retrieved, wrapped, and rethrown.
 * The exception(!) to the rule is errors loading a package via {@link #getPackage}, since the
 * corresponding target pattern does throw eagerly if the package cannot be loaded.
 *
 * <p>On the other hand, exceptions indicating a bad filesystem are propagated eagerly, since they
 * are catastrophic failures that should terminate the evaluation.
 */
public final class EnvironmentBackedRecursivePackageProvider
    extends AbstractRecursivePackageProvider {

  private final Environment env;
  private final AtomicBoolean encounteredPackageErrors = new AtomicBoolean(false);
  private final AtomicReference<NoSuchPackageException> noSuchPackageException =
      new AtomicReference<>();

  EnvironmentBackedRecursivePackageProvider(Environment env) {
    this.env = env;
  }

  /**
   * Whether any of the calls to {@link #getPackage}, {@link #getTarget}, {@link #bulkGetPackages},
   * or {@link
   * com.google.devtools.build.lib.pkgcache.RecursivePackageProvider#streamPackagesUnderDirectory}
   * encountered a package in error.
   *
   * <p>The client of {@link EnvironmentBackedRecursivePackageProvider} may want to check this. See
   * comments in {@link #getPackage} for details.
   */
  boolean encounteredPackageErrors() {
    return encounteredPackageErrors.get();
  }

  @Nullable
  NoSuchPackageException maybeGetNoSuchPackageException() {
    return noSuchPackageException.get();
  }

  @Override
  public Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException, MissingDepException, InterruptedException {
    PackageValue pkgValue;
    try {
      pkgValue = (PackageValue) env.getValueOrThrow(packageName, NoSuchPackageException.class);
      if (pkgValue == null) {
        throw new MissingDepException();
      }
    } catch (NoSuchPackageException e) {
      encounteredPackageErrors.set(true);
      throw e;
    }

    Package pkg = pkgValue.getPackage();
    if (pkg.containsErrors()) {
      // If this is a nokeep_going build, we must shut the build down by throwing an exception. To
      // do that, we request a node that will throw an exception, and then try to catch it and
      // continue. This gives the framework notification to shut down the build if it should.
      try {
        env.getValueOrThrow(
            PackageErrorFunction.key(packageName), BuildFileContainsErrorsException.class);
        Preconditions.checkState(env.valuesMissing(), "Should have thrown for %s", packageName);
        throw new MissingDepException();
      } catch (BuildFileContainsErrorsException e) {
        // If this is a keep_going build, then the user of this RecursivePackageProvider has two
        // options for handling the "package in error" case. The user must either inspect the
        // package returned by this method, or else determine whether any errors have been seen via
        // the "encounteredPackageErrors" method.
        encounteredPackageErrors.set(true);
        noSuchPackageException.compareAndSet(null, e);
      }
    }
    return pkgValue.getPackage();
  }

  @Override
  public Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds)
      throws NoSuchPackageException, InterruptedException {
    ImmutableMap.Builder<PackageIdentifier, Package> builder = ImmutableMap.builder();
    for (PackageIdentifier pkgId : pkgIds) {
      builder.put(pkgId, getPackage(env.getListener(), pkgId));
    }
    return builder.buildOrThrow();
  }

  @SuppressWarnings("ThrowsUncheckedException") // Good for callers to know about MissingDep.
  @Override
  public boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageId)
      throws MissingDepException, InconsistentFilesystemException, InterruptedException {
    SkyKey packageLookupKey = PackageLookupValue.key(packageId);
    try {
      PackageLookupValue packageLookupValue =
          (PackageLookupValue)
              env.getValueOrThrow(
                  packageLookupKey,
                  NoSuchPackageException.class,
                  InconsistentFilesystemException.class);
      if (packageLookupValue == null) {
        throw new MissingDepException();
      }
      return packageLookupValue.packageExists();
    } catch (NoSuchPackageException e) {
      noSuchPackageException.compareAndSet(null, e);
      encounteredPackageErrors.set(true);
      return false;
    }
  }

  @Override
  public void streamPackagesUnderDirectory(
      SafeBatchCallback<PackageIdentifier> results,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      IgnoredSubdirectories ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException, NoSuchPackageException, ProcessPackageDirectoryException {
    PathPackageLocator packageLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    if (packageLocator == null) {
      throw new MissingDepException();
    }

    List<Root> roots = new ArrayList<>();
    if (repository.isMain()) {
      roots.addAll(packageLocator.getPathEntries());
    } else {
      RepositoryDirectoryValue repositoryValue =
          (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(repository));
      if (repositoryValue == null) {
        throw new MissingDepException();
      }

      if (!repositoryValue.repositoryExists()) {
        eventHandler.handle(
            Event.error(
                String.format(
                    "No such repository '%s': %s", repository, repositoryValue.getErrorMsg())));
        return;
      }
      roots.add(Root.fromPath(repositoryValue.getPath()));
    }

    IgnoredSubdirectories filteredIgnoredSubdirectories =
        ignoredSubdirectories.filterForDirectory(directory);

    Iterable<RecursivePkgValue.Key> recursivePackageKeys =
        Iterables.transform(
            roots,
            r ->
                RecursivePkgValue.key(
                    repository,
                    RootedPath.toRootedPath(r, directory),
                    filteredIgnoredSubdirectories));
    SkyframeLookupResult recursivePackageValues = env.getValuesAndExceptions(recursivePackageKeys);
    NoSuchPackageException firstNspe = null;
    for (RecursivePkgValue.Key key : recursivePackageKeys) {
      RecursivePkgValue lookup;
      try {
        lookup =
            (RecursivePkgValue)
                recursivePackageValues.getOrThrow(
                    key, NoSuchPackageException.class, ProcessPackageDirectoryException.class);
      } catch (NoSuchPackageException e) {
        // NoSuchPackageException can happen during error bubbling in a no-keep-going build.
        if (firstNspe == null) {
          firstNspe = e;
        }
        encounteredPackageErrors.set(true);
        noSuchPackageException.compareAndSet(null, e);
        continue;
      }
      if (lookup == null) {
        continue;
      }
      if (lookup.hasErrors()) {
        encounteredPackageErrors.set(true);
      }

      if (env.valuesMissing()) {
        // If values are missing, we're only checking for errors, not constructing a result.
        continue;
      }
      for (String packageName : lookup.getPackages().toList()) {
        // TODO(bazel-team): Make RecursivePkgValue return NestedSet<PathFragment> so this transform
        // is unnecessary.
        PathFragment packageNamePathFragment = PathFragment.create(packageName);
        if (!Iterables.any(excludedSubdirectories, packageNamePathFragment::startsWith)) {
          results.process(
              ImmutableList.of(PackageIdentifier.create(repository, packageNamePathFragment)));
        }
      }
    }
    if (firstNspe != null) {
      throw firstNspe;
    }
    if (env.valuesMissing()) {
      throw new MissingDepException();
    }
  }
}
