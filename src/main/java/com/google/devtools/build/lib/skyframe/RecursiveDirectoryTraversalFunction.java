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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.ProcessPackageDirectory.ProcessPackageDirectorySkyFunctionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.errorprone.annotations.ForOverride;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * RecursiveDirectoryTraversalFunction allows for a custom recursive traversal of the subdirectories
 * of a directory, building up a value based on package existence and results of the recursive
 * traversal.
 *
 * <p>It attempts to ignore package-definition-related errors, and even file symlink cycles, which
 * means that in keep-going mode, it will produce a result even if the traversed directory contains
 * such errors. In no-keep-going mode, such exceptions will shut down the build, so callers must be
 * prepared to handle {@link com.google.devtools.build.lib.packages.NoSuchPackageException} and
 * {@link com.google.devtools.build.lib.io.FileSymlinkException}.
 *
 * <p>It will always eagerly fail on exceptions indicating filesystem inconsistencies, since they
 * indicate bad disk that may make results unreliable.
 */
public abstract class RecursiveDirectoryTraversalFunction<
    ConsumerT extends RecursiveDirectoryTraversalFunction.PackageDirectoryConsumer, ReturnT> {
  private final BlazeDirectories directories;

  protected RecursiveDirectoryTraversalFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  /** Called by {@link #visitDirectory}, which will then recursive traverse the directory. */
  @ForOverride
  @Nullable
  protected ProcessPackageDirectoryResult getProcessPackageDirectoryResult(
      RecursivePkgKey recursivePkgKey, Environment env)
      throws InterruptedException, ProcessPackageDirectorySkyFunctionException {
    return new ProcessPackageDirectory(directories, this::getSkyKeyForSubdirectory)
        .getPackageExistenceAndSubdirDeps(
            recursivePkgKey.getRootedPath(),
            recursivePkgKey.getRepositoryName(),
            recursivePkgKey.getExcludedPaths(),
            env);
  }

  /**
   * Called by {@link #visitDirectory}, which will next call {@link
   * PackageDirectoryConsumer#notePackage} if the {@code recursivePkgKey} specifies a directory with
   * a package, and which will lastly be provided to {@link #aggregateWithSubdirectorySkyValues} to
   * compute the {@code TReturn} value returned by {@link #visitDirectory}.
   */
  protected abstract ConsumerT getInitialConsumer();

  /**
   * Called by {@link #visitDirectory} to get the {@link SkyKey}s associated with recursive
   * computation in subdirectories of {@code subdirectory}, excluding directories in {@code
   * excludedSubdirectoriesBeneathSubdirectory}, all of which must be proper subdirectories of
   * {@code subdirectory}.
   */
  protected abstract SkyKey getSkyKeyForSubdirectory(
      RepositoryName repository,
      RootedPath subdirectory,
      IgnoredSubdirectories excludedSubdirectoriesBeneathSubdirectory);

  /**
   * Called by {@link #visitDirectory} to compute the {@code TReturn} value it returns, as a
   * function of {@code consumer} and the {@link SkyValue}s computed for subdirectories of the
   * directory specified by {@code recursivePkgKey}, contained in {@code subdirectorySkyValues}.
   */
  protected abstract ReturnT aggregateWithSubdirectorySkyValues(
      ConsumerT consumer, Map<SkyKey, SkyValue> subdirectorySkyValues);

  /**
   * A type of consumer used by {@link #visitDirectory} as it checks for a package in the directory
   * specified by {@code recursivePkgKey}; if such a package exists, {@link #notePackage} is called.
   *
   * <p>The consumer is then provided to {@link #aggregateWithSubdirectorySkyValues} to compute the
   * value returned by {@link #visitDirectory}.
   */
  public interface PackageDirectoryConsumer {
    /** Called iff the directory contains a package. */
    void notePackage(PathFragment pkgPath) throws InterruptedException;

    /**
     * Called iff the directory contains a BUILD file but *not* a package, which can happen under
     * the following circumstances:
     *
     * <ol>
     *   <li>The BUILD file contains a Starlark load statement that is in error
     *   <li>TODO(mschaller), not yet implemented: The BUILD file is a symlink that points into a
     *       cycle
     * </ol>
     */
    void notePackageError(String noSuchPackageExceptionErrorMessage);
  }

  /**
   * Uses {@link #getProcessPackageDirectoryResult} to look for a package in the directory specified
   * by {@code recursivePkgKey}, does some work as specified by {@link PackageDirectoryConsumer} if
   * such a package exists, then recursively does work in each non-excluded subdirectory as
   * specified by {@link #getSkyKeyForSubdirectory}, and finally aggregates the {@link
   * PackageDirectoryConsumer} value along with values from each subdirectory as specified by {@link
   * #aggregateWithSubdirectorySkyValues}, and returns that aggregation.
   *
   * <p>Returns null if {@code env.valuesMissing()} is true, checked after each call to one of
   * {@link RecursiveDirectoryTraversalFunction}'s abstract methods that were given {@code env}.
   *
   * <p>Will propagate {@link com.google.devtools.build.lib.packages.NoSuchPackageException} during
   * a no-keep-going evaluation
   */
  @Nullable
  public final ReturnT visitDirectory(RecursivePkgKey recursivePkgKey, Environment env)
      throws InterruptedException, ProcessPackageDirectorySkyFunctionException {
    ProcessPackageDirectoryResult processPackageDirectoryResult =
        getProcessPackageDirectoryResult(recursivePkgKey, env);
    if (env.valuesMissing()) {
      return null;
    }

    Iterable<SkyKey> childDeps = processPackageDirectoryResult.getChildDeps();
    ConsumerT consumer = getInitialConsumer();

    SkyframeLookupResult dependentSkyValues;
    if (processPackageDirectoryResult.packageExists()) {
      PathFragment rootRelativePath = recursivePkgKey.getRootedPath().getRootRelativePath();
      SkyKey packageErrorMessageKey =
          PackageErrorMessageValue.key(
              PackageIdentifier.create(recursivePkgKey.getRepositoryName(), rootRelativePath));
      // In a no-keep-going build during error bubbling, PackageErrorMessageFunction may throw a
      // NoSuchPackageException. Since we don't catch such an exception here, this SkyFunction will
      // return immediately with a missing value, and the NoSuchPackageException will propagate up.
      dependentSkyValues =
          env.getValuesAndExceptions(
              Iterables.concat(ImmutableList.of(packageErrorMessageKey), childDeps));
      if (env.valuesMissing()) {
        return null;
      }
      PackageErrorMessageValue pkgErrorMessageValue =
          (PackageErrorMessageValue) dependentSkyValues.get(packageErrorMessageKey);
      if (pkgErrorMessageValue == null) {
        return null;
      }
      switch (pkgErrorMessageValue.getResult()) {
        case NO_ERROR:
          consumer.notePackage(rootRelativePath);
          break;
        case ERROR:
          env.getListener()
              .handle(Event.error("package contains errors: " + rootRelativePath.getPathString()));
          consumer.notePackage(rootRelativePath);
          break;
        case NO_SUCH_PACKAGE_EXCEPTION:
          // The package had errors, but don't fail-fast as there might be subpackages below the
          // current directory.
          String msg = pkgErrorMessageValue.getNoSuchPackageExceptionMessage();
          env.getListener().handle(Event.error(msg));
          consumer.notePackageError(msg);
          break;
        default:
          throw new IllegalStateException(pkgErrorMessageValue.getResult().toString());
      }
    } else {
      dependentSkyValues = env.getValuesAndExceptions(childDeps);
      if (env.valuesMissing()) {
        return null;
      }
    }
    ImmutableMap.Builder<SkyKey, SkyValue> subdirectorySkyValuesFromDeps =
        ImmutableMap.builderWithExpectedSize(Iterables.size(childDeps));
    for (SkyKey skyKey : childDeps) {
      SkyValue skyValue = dependentSkyValues.get(skyKey);
      if (skyValue == null) {
        return null;
      }
      subdirectorySkyValuesFromDeps.put(skyKey, skyValue);
    }

    subdirectorySkyValuesFromDeps.putAll(
        processPackageDirectoryResult.getAdditionalValuesToAggregate());
    return aggregateWithSubdirectorySkyValues(
        consumer, subdirectorySkyValuesFromDeps.buildOrThrow());
  }
}
