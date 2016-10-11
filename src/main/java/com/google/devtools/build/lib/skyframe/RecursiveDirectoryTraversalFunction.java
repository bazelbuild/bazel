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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.Map;

/**
 * RecursiveDirectoryTraversalFunction traverses the subdirectories of a directory, looking for
 * and loading packages, and builds up a value from these packages in a manner customized by
 * classes that derive from it.
 */
abstract class RecursiveDirectoryTraversalFunction
    <TVisitor extends RecursiveDirectoryTraversalFunction.Visitor, TReturn> {
  private static final String SENTINEL_FILE_NAME_FOR_NOT_TRAVERSING_SYMLINKS =
      "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN";

  private final ProcessPackageDirectory processPackageDirectory;

  protected RecursiveDirectoryTraversalFunction(BlazeDirectories directories) {
    this.processPackageDirectory =
        new ProcessPackageDirectory(
            directories,
            new ProcessPackageDirectory.SkyKeyTransformer() {
              @Override
              public SkyKey makeSkyKey(
                  RepositoryName repository,
                  RootedPath subdirectory,
                  ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
                return getSkyKeyForSubdirectory(
                    repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory);
              }
            });
  }

  /**
   * Called by {@link #visitDirectory}, which will next call {@link Visitor#visitPackageValue} if
   * the {@code recursivePkgKey} specifies a directory with a package, and which will lastly be
   * provided to {@link #aggregateWithSubdirectorySkyValues} to compute the {@code TReturn} value
   * returned by {@link #visitDirectory}.
   */
  protected abstract TVisitor getInitialVisitor();

  /**
   * Called by {@link #visitDirectory} to get the {@link SkyKey}s associated with recursive
   * computation in subdirectories of {@code subdirectory}, excluding directories in
   * {@code excludedSubdirectoriesBeneathSubdirectory}, all of which must be proper subdirectories
   * of {@code subdirectory}.
   */
  protected abstract SkyKey getSkyKeyForSubdirectory(
      RepositoryName repository, RootedPath subdirectory,
      ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory);

  /**
   * Called by {@link #visitDirectory} to compute the {@code TReturn} value it returns, as a
   * function of {@code visitor} and the {@link SkyValue}s computed for subdirectories
   * of the directory specified by {@code recursivePkgKey}, contained in
   * {@code subdirectorySkyValues}.
   */
  protected abstract TReturn aggregateWithSubdirectorySkyValues(
      TVisitor visitor, Map<SkyKey, SkyValue> subdirectorySkyValues);

  /**
   * A type of value used by {@link #visitDirectory} as it checks for a package in the directory
   * specified by {@code recursivePkgKey}; if such a package exists, {@link #visitPackageValue}
   * is called.
   *
   * <p>The value is then provided to {@link #aggregateWithSubdirectorySkyValues} to compute the
   * value returned by {@link #visitDirectory}.
   */
  interface Visitor {

    /**
     * Called iff the directory contains a package. Provides an {@link Environment} {@code env} so
     * that the visitor may do additional lookups. {@link Environment#valuesMissing} will be checked
     * afterwards.
     */
    void visitPackageValue(Package pkg, Environment env) throws InterruptedException;
  }

  /**
   * Looks in the directory specified by {@code recursivePkgKey} for a package, does some work as
   * specified by {@link Visitor} if such a package exists, then recursively does work in each
   * non-excluded subdirectory as specified by {@link #getSkyKeyForSubdirectory}, and finally
   * aggregates the {@link Visitor} value along with values from each subdirectory as specified by
   * {@link #aggregateWithSubdirectorySkyValues}, and returns that aggregation.
   *
   * <p>Returns null if {@code env.valuesMissing()} is true, checked after each call to one of
   * {@link RecursiveDirectoryTraversalFunction}'s abstract methods that were given {@code env}.
   * (And after each of {@code visitDirectory}'s own uses of {@code env}, of course.)
   */
  TReturn visitDirectory(RecursivePkgKey recursivePkgKey, Environment env)
      throws InterruptedException {
    RootedPath rootedPath = recursivePkgKey.getRootedPath();
    ProcessPackageDirectoryResult packageExistenceAndSubdirDeps =
        processPackageDirectory.getPackageExistenceAndSubdirDeps(
            rootedPath, recursivePkgKey.getRepository(), env, recursivePkgKey.getExcludedPaths());
    if (env.valuesMissing()) {
      return null;
    }

    Iterable<SkyKey> childDeps = packageExistenceAndSubdirDeps.getChildDeps();

    TVisitor visitor = getInitialVisitor();

    Map<SkyKey, SkyValue> subdirectorySkyValues;
    if (packageExistenceAndSubdirDeps.packageExists()) {
      PathFragment rootRelativePath = rootedPath.getRelativePath();
      SkyKey packageKey =
          PackageValue.key(
              PackageIdentifier.create(recursivePkgKey.getRepository(), rootRelativePath));
      Map<SkyKey, ValueOrException<NoSuchPackageException>> dependentSkyValues =
          env.getValuesOrThrow(
              Iterables.concat(childDeps, ImmutableList.of(packageKey)),
              NoSuchPackageException.class);
      if (env.valuesMissing()) {
        return null;
      }
      Package pkg = null;
      try {
        PackageValue pkgValue = (PackageValue) dependentSkyValues.get(packageKey).get();
        if (pkgValue == null) {
          return null;
        }
        pkg = pkgValue.getPackage();
        if (pkg.containsErrors()) {
          env.getListener()
              .handle(Event.error("package contains errors: " + rootRelativePath.getPathString()));
        }
      } catch (NoSuchPackageException e) {
        // The package had errors, but don't fail-fast as there might be subpackages below the
        // current directory.
        env.getListener()
            .handle(Event.error("package contains errors: " + rootRelativePath.getPathString()));
      }
      if (pkg != null) {
        visitor.visitPackageValue(pkg, env);
        if (env.valuesMissing()) {
          return null;
        }
      }
      ImmutableMap.Builder<SkyKey, SkyValue> subdirectoryBuilder = ImmutableMap.builder();
      for (Map.Entry<SkyKey, ValueOrException<NoSuchPackageException>> entry :
          Maps.filterKeys(dependentSkyValues, Predicates.not(Predicates.equalTo(packageKey)))
              .entrySet()) {
        try {
          subdirectoryBuilder.put(entry.getKey(), entry.getValue().get());
        } catch (NoSuchPackageException e) {
          // ignored.
        }
      }
      subdirectorySkyValues = subdirectoryBuilder.build();
    } else {
      subdirectorySkyValues = env.getValues(childDeps);
    }
    if (env.valuesMissing()) {
      return null;
    }
    return aggregateWithSubdirectorySkyValues(visitor, subdirectorySkyValues);
  }
}
