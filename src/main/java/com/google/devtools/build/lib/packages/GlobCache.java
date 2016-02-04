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
import com.google.common.base.Predicate;
import com.google.common.base.Throwables;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Caches the results of glob expansion for a package.
 */
  // Used outside of Bazel!
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

  /**
   * System call caching layer.
   */
  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;

  /**
   * The thread pool for glob evaluation.
   */
  private final ThreadPoolExecutor globExecutor;

  /**
   * Create a glob expansion cache.
   * @param packageDirectory globs will be expanded relatively to this
   *                         directory.
   * @param packageId the name of the package this cache belongs to.
   * @param locator the package locator.
   * @param globExecutor thread pool for glob evaluation.
   */
  public GlobCache(final Path packageDirectory,
                   final PackageIdentifier packageId,
                   final CachingPackageLocator locator,
                   AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls,
                   ThreadPoolExecutor globExecutor) {
    this.packageDirectory = Preconditions.checkNotNull(packageDirectory);
    this.packageId = Preconditions.checkNotNull(packageId);
    this.globExecutor = Preconditions.checkNotNull(globExecutor);
    this.syscalls = syscalls == null ? new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS) : syscalls;

    Preconditions.checkNotNull(locator);
    childDirectoryPredicate = new Predicate<Path>() {
      @Override
      public boolean apply(Path directory) {
        if (directory.equals(packageDirectory)) {
          return true;
        }
        PackageIdentifier subPackageId = PackageIdentifier.create(
            packageId.getRepository(),
            packageId.getPackageFragment().getRelative(directory.relativeTo(packageDirectory)));
        return locator.getBuildFileForPackage(subPackageId) == null;
      }
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
  Future<List<Path>> getGlobAsync(String pattern, boolean excludeDirs)
      throws BadGlobException {
    Future<List<Path>> cached = globCache.get(Pair.of(pattern, excludeDirs));
    if (cached == null) {
      cached = safeGlob(pattern, excludeDirs);
      setGlobPaths(pattern, excludeDirs, cached);
    }
    return cached;
  }

  @VisibleForTesting
  List<String> getGlob(String pattern)
      throws IOException, BadGlobException, InterruptedException {
    return getGlob(pattern, false);
  }

  @VisibleForTesting
  protected List<String> getGlob(String pattern, boolean excludeDirs)
      throws IOException, BadGlobException, InterruptedException {
    Future<List<Path>> futureResult = getGlobAsync(pattern, excludeDirs);
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

  /**
   * Adds glob entries to the cache, making sure they are sorted first.
   */
  @VisibleForTesting
  void setGlobPaths(String pattern, boolean excludeDirectories, Future<List<Path>> result) {
    globCache.put(Pair.of(pattern, excludeDirectories), result);
  }

  /**
   * Actually execute a glob against the filesystem.  Otherwise similar to
   * getGlob().
   */
  @VisibleForTesting
  Future<List<Path>> safeGlob(String pattern, boolean excludeDirs) throws BadGlobException {
    // Forbidden patterns:
    if (pattern.indexOf('?') != -1) {
      throw new BadGlobException("glob pattern '" + pattern + "' contains forbidden '?' wildcard");
    }
    // Patterns forbidden by UnixGlob library:
    String error = UnixGlob.checkPatternForError(pattern);
    if (error != null) {
      throw new BadGlobException(error + " (in glob pattern '" + pattern + "')");
    }
    return UnixGlob.forPath(packageDirectory)
        .addPattern(pattern)
        .setExcludeDirectories(excludeDirs)
        .setDirectoryFilter(childDirectoryPredicate)
        .setThreadPool(globExecutor)
        .setFilesystemCalls(syscalls)
        .globAsync(true);
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
   * Returns true iff all this package's globs are up-to-date.  That is,
   * re-evaluating the package's BUILD file at this moment would yield an
   * equivalent Package instance.  (This call requires filesystem I/O to
   * re-evaluate the globs.)
   */
  public boolean globsUpToDate() throws InterruptedException {
    // Start all globs in parallel.
    Map<Pair<String, Boolean>, Future<List<Path>>> newGlobs = new HashMap<>();
    try {
      for (Map.Entry<Pair<String, Boolean>, Future<List<Path>>> entry : globCache.entrySet()) {
        Pair<String, Boolean> key = entry.getKey();
        try {
          newGlobs.put(key, safeGlob(key.first, key.second));
        } catch (BadGlobException e) {
          return false;
        }
      }

      for (Map.Entry<Pair<String, Boolean>, Future<List<Path>>> entry : globCache.entrySet()) {
        try {
          Pair<String, Boolean> key = entry.getKey();
          List<Path> newGlob = fromFuture(newGlobs.get(key));
          List<Path> oldGlob = fromFuture(entry.getValue());
          if (!oldGlob.equals(newGlob)) {
            return false;
          }
        } catch (IOException e) {
          return false;
        }
      }

      return true;
    } finally {
      finishBackgroundTasks(newGlobs.values());
    }
  }

  /**
   * Evaluate the build language expression "glob(includes, excludes)" in the
   * context of this package.
   *
   * <p>Called by PackageFactory via Package.
   */
  public List<String> glob(List<String> includes, List<String> excludes, boolean excludeDirs)
      throws IOException, BadGlobException, InterruptedException {
    // Start globbing all patterns in parallel. The getGlob() calls below will
    // block on an individual pattern's results, but the other globs can
    // continue in the background.
    for (String pattern : Iterables.concat(includes, excludes)) {
      getGlobAsync(pattern, excludeDirs);
    }

    LinkedHashSet<String> results = Sets.newLinkedHashSetWithExpectedSize(includes.size());
    for (String pattern : includes) {
      results.addAll(getGlob(pattern, excludeDirs));
    }
    for (String pattern : excludes) {
      results.removeAll(getGlob(pattern, excludeDirs));
    }

    Preconditions.checkState(!results.contains(null), "glob returned null");
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
