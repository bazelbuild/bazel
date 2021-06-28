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
package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Caches the results of glob expansion for a package.
 */
@ThreadSafety.ThreadCompatible
public class GlobCache {
  /**
   * A mapping from glob expressions (e.g. "*.java") to the list of files it
   * matched (in the order returned by VFS) at the time the package was
   * constructed.  Required for sound dependency analysis.
   *
   * We don't use a Multimap because it provides no way to distinguish "key not
   * present" from (key -> {}).
   */
  private final Map<Pair<String, Boolean>, Future<List<Path>>> globCache = new HashMap<>();

  /**
   * The directory in which our package's BUILD file resides.
   */
  private final Path packageDirectory;

  /**
   * The name of the package we belong to.
   */
  private final PackageIdentifier packageId;

  /**
   * The package locator-based directory traversal predicate.
   */
  private final Predicate<Path> childDirectoryPredicate;

  /** System call caching layer. */
  private final AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;

  private final int maxDirectoriesToEagerlyVisit;

  /** The thread pool for glob evaluation. */
  private final Executor globExecutor;

  private final AtomicBoolean globalStarted = new AtomicBoolean(false);

  /**
   * Create a glob expansion cache.
   *
   * @param packageDirectory globs will be expanded relatively to this directory.
   * @param packageId the name of the package this cache belongs to.
   * @param locator the package locator.
   * @param globExecutor thread pool for glob evaluation.
   * @param maxDirectoriesToEagerlyVisit the number of directories to eagerly traverse on the first
   *     glob for a given package, in order to warm the filesystem. -1 means do no eager traversal.
   *     See {@link
   *     com.google.devtools.build.lib.pkgcache.PackageOptions#maxDirectoriesToEagerlyVisitInGlobbing}.
   */
  public GlobCache(
      final Path packageDirectory,
      final PackageIdentifier packageId,
      final ImmutableSet<PathFragment> ignoredGlobPrefixes,
      final CachingPackageLocator locator,
      AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls,
      Executor globExecutor,
      int maxDirectoriesToEagerlyVisit) {
    this.packageDirectory = Preconditions.checkNotNull(packageDirectory);
    this.packageId = Preconditions.checkNotNull(packageId);
    this.globExecutor = Preconditions.checkNotNull(globExecutor);
    this.syscalls = syscalls == null ? new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS) : syscalls;
    this.maxDirectoriesToEagerlyVisit = maxDirectoriesToEagerlyVisit;

    Preconditions.checkNotNull(locator);
    childDirectoryPredicate =
        directory -> {
          if (directory.equals(packageDirectory)) {
            return true;
          }

          PathFragment subPackagePath =
              packageId.getPackageFragment().getRelative(directory.relativeTo(packageDirectory));

          for (PathFragment ignoredPrefix : ignoredGlobPrefixes) {
            if (subPackagePath.startsWith(ignoredPrefix)) {
              return false;
            }
          }

          PackageIdentifier subPackageId =
              PackageIdentifier.create(packageId.getRepository(), subPackagePath);
          return locator.getBuildFileForPackage(subPackageId) == null;
        };
  }

  /**
   * Returns the future result of evaluating glob "pattern" against this
   * package's directory, using the package's cache of previously-started
   * globs if possible.
   *
   * @return the list of paths matching the pattern, relative to the package's
   *   directory.
   * @throws BadGlobException if the glob was syntactically invalid, or
   *  contained uplevel references.
   */
  Future<List<Path>> getGlobUnsortedAsync(String pattern, boolean excludeDirs)
      throws BadGlobException {
    Future<List<Path>> cached = globCache.get(Pair.of(pattern, excludeDirs));
    if (cached == null) {
      if (maxDirectoriesToEagerlyVisit > -1
          && !globalStarted.getAndSet(true)) {
        packageDirectory.prefetchPackageAsync(maxDirectoriesToEagerlyVisit);
      }
      cached = safeGlobUnsorted(pattern, excludeDirs);
      setGlobPaths(pattern, excludeDirs, cached);
    }
    return cached;
  }

  @VisibleForTesting
  List<String> getGlobUnsorted(String pattern)
      throws IOException, BadGlobException, InterruptedException {
    return getGlobUnsorted(pattern, false);
  }

  @VisibleForTesting
  protected List<String> getGlobUnsorted(String pattern, boolean excludeDirs)
      throws IOException, BadGlobException, InterruptedException {
    Future<List<Path>> futureResult = getGlobUnsortedAsync(pattern, excludeDirs);
    List<Path> globPaths = fromFuture(futureResult);
    // Replace the UnixGlob.GlobFuture with a completed future object, to allow
    // garbage collection of the GlobFuture and GlobVisitor objects.
    if (!(futureResult instanceof SettableFuture<?>)) {
      SettableFuture<List<Path>> completedFuture = SettableFuture.create();
      completedFuture.set(globPaths);
      globCache.put(Pair.of(pattern, excludeDirs), completedFuture);
    }

    List<String> result = Lists.newArrayListWithCapacity(globPaths.size());
    for (Path path : globPaths) {
      String relative = path.relativeTo(packageDirectory).getPathString();
      // Don't permit "" (meaning ".") in the glob expansion, since it's
      // invalid as a label, plus users should say explicitly if they
      // really want to name the package directory.
      if (!relative.isEmpty()) {
        result.add(relative);
      }
    }
    return result;
  }

  /** Adds glob entries to the cache. */
  private void setGlobPaths(String pattern, boolean excludeDirectories, Future<List<Path>> result) {
    globCache.put(Pair.of(pattern, excludeDirectories), result);
  }

  /**
   * Actually execute a glob against the filesystem.  Otherwise similar to
   * getGlob().
   */
  @VisibleForTesting
  Future<List<Path>> safeGlobUnsorted(String pattern, boolean excludeDirs) throws BadGlobException {
    // Forbidden patterns:
    if (pattern.indexOf('?') != -1) {
      throw new BadGlobException("glob pattern '" + pattern + "' contains forbidden '?' wildcard");
    }
    // Patterns forbidden by UnixGlob library:
    String error = UnixGlob.checkPatternForError(pattern);
    if (error != null) {
      throw new BadGlobException(error + " (in glob pattern '" + pattern + "')");
    }
    try {
      return UnixGlob.forPath(packageDirectory)
          .addPattern(pattern)
          .setExcludeDirectories(excludeDirs)
          .setDirectoryFilter(childDirectoryPredicate)
          .setExecutor(globExecutor)
          .setFilesystemCalls(syscalls)
          .globAsync();
    } catch (UnixGlob.BadPattern ex) {
      throw new BadGlobException(ex.getMessage());
    }
  }

  /**
   * Sanitize the future exceptions - the only expected checked exception
   * is IOException.
   */
  private static List<Path> fromFuture(Future<List<Path>> future)
      throws IOException, InterruptedException {
    try {
      return future.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfPossible(cause,
          IOException.class, InterruptedException.class);
      throw new RuntimeException(e);
    }
  }

  /**
   * Helper for evaluating the build language expression "glob(includes, excludes)" in the context
   * of this package.
   *
   * <p>Called by PackageFactory via Package.
   */
  public List<String> globUnsorted(
      List<String> includes, List<String> excludes, boolean excludeDirs, boolean allowEmpty)
      throws IOException, BadGlobException, InterruptedException {
    // Start globbing all patterns in parallel. The getGlob() calls below will
    // block on an individual pattern's results, but the other globs can
    // continue in the background.
    for (String pattern : includes) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError = getGlobUnsortedAsync(pattern, excludeDirs);
    }

    HashSet<String> results = new HashSet<>();
    for (String pattern : includes) {
      List<String> items = getGlobUnsorted(pattern, excludeDirs);
      if (!allowEmpty && items.isEmpty()) {
        throw new BadGlobException(
            "glob pattern '"
                + pattern
                + "' didn't match anything, but allow_empty is set to False "
                + "(the default value of allow_empty can be set with "
                + "--incompatible_disallow_empty_glob).");
      }
      results.addAll(items);
    }
    try {
      UnixGlob.removeExcludes(results, excludes);
    } catch (UnixGlob.BadPattern ex) {
      throw new BadGlobException(ex.getMessage());
    }
    if (!allowEmpty && results.isEmpty()) {
      throw new BadGlobException(
          "all files in the glob have been excluded, but allow_empty is set to False "
              + "(the default value of allow_empty can be set with "
              + "--incompatible_disallow_empty_glob).");
    }
    return new ArrayList<>(results);
  }

  public Set<Pair<String, Boolean>> getKeySet() {
    return globCache.keySet();
  }

  /**
   * Block on the completion of all potentially-abandoned background tasks.
   */
  public void finishBackgroundTasks() {
    finishBackgroundTasks(globCache.values());
  }

  public void cancelBackgroundTasks() {
    cancelBackgroundTasks(globCache.values());
  }

  private static void finishBackgroundTasks(Collection<Future<List<Path>>> tasks) {
    for (Future<List<Path>> task : tasks) {
      try {
        fromFuture(task);
      } catch (CancellationException | IOException | InterruptedException e) {
        // Ignore: If this was still going on in the background, some other
        // failure already occurred.
      }
    }
  }

  private static void cancelBackgroundTasks(Collection<Future<List<Path>>> tasks) {
    for (Future<List<Path>> task : tasks) {
      task.cancel(true);
    }

    for (Future<List<Path>> task : tasks) {
      try {
        task.get();
      } catch (CancellationException | ExecutionException | InterruptedException e) {
        // We don't care. Point is, the task does not bother us anymore.
      }
    }
  }

  @Override
  public String toString() {
    return "GlobCache for " + packageId + " in " + packageDirectory;
  }
}
