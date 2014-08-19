// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

/**
 * A mapping from the name of a package to the location of its BUILD file.
 * The implementation composes an ordered sequence of directories according to
 * the package-path rules.
 *
 * <p>All methods are thread-safe, and (assuming no change to the underlying
 * filesystem) idempotent.
 */
public class PathPackageLocator {

  public static final Set<String> DEFAULT_TOP_LEVEL_EXCLUDES =
      ImmutableSet.of("experimental", "obsolete");

  /**
   * An interface which accepts {@link PathFragment}s.
   */
  public interface AcceptsPathFragment {

    /**
     * Accept a {@link PathFragment}.
     *
     * @param fragment The path fragment.
     */
    void accept(PathFragment fragment);
  }

  private static final Logger LOG = Logger.getLogger(PathPackageLocator.class.getName());

  private final ImmutableList<Path> pathEntries;

  /**
   * Constructs a PathPackageLocator based on the specified list of package root directories.
   */
  public PathPackageLocator(List<Path> pathEntries) {
    this.pathEntries = ImmutableList.copyOf(pathEntries);
  }

  /**
   * Constructs a PathPackageLocator based on the specified array of package root directories.
   */
  public PathPackageLocator(Path... pathEntries) {
    this(Arrays.asList(pathEntries));
  }

  /**
   * Returns the path to the build file for this package.
   *
   * <p>The package's root directory may be computed by calling getParentFile()
   * on the result of this function.
   *
   * <p>Instances of this interface do not attempt to do any caching, nor
   * implement checks for package-boundary crossing logic; the PackageCache
   * does that.
   *
   * <p>If the same package exists beneath multiple package path entries, the
   * first path that matches always wins.
   */
  public Path getPackageBuildFile(String packageName) throws NoSuchPackageException {
    Path buildFile  = getPackageBuildFileNullable(packageName, UnixGlob.DEFAULT_SYSCALLS_REF);
    if (buildFile == null) {
      throw new BuildFileNotFoundException(packageName, "BUILD file not found on package path");
    }
    return buildFile;
  }

  /**
   * Like #getPackageBuildFile(), but returns null instead of throwing.
   *
   * @param packageName the name of the package.
   * @param cache a filesystem-level cache of stat() calls.
   */
  public Path getPackageBuildFileNullable(String packageName,
      AtomicReference<? extends UnixGlob.FilesystemCalls> cache)  {
    for (Path pathEntry : pathEntries) {
      Path buildFile = pathEntry.getRelative(packageName).getChild("BUILD");
      FileStatus stat = cache.get().statNullable(buildFile, Symlinks.FOLLOW);
      if (stat != null && stat.isFile()) {
        return buildFile;
      }
    }
    return null;
  }

  /**
   * <p>Visits the names of all packages beneath the given directory
   * recursively and concurrently.
   *
   * <p>Note: This operation needs to stat directories recursively.  It could
   * be very expensive when there is a big tree under the given directory.
   *
   * <p>Over a single iteration, package names are unique.
   *
   * <p>This method may spawn multiple threads and call the observer method
   * concurrently. When this method terminates, however, all such threads
   * will have completed.
   *
   * <p>To abort the traversal, call {@link Thread#interrupt()} on the calling
   * thread.
   *
   * @param directory The directory to search. It must be a relative
   *    path, free from up-level references.
   * @param eventHandler a eventHandler which should be used to log any errors that
   *    occur while scanning directories for BUILD files.
   * @param cache file system call cache to be used with the recursive
   *    visitation
   * @param topLevelExcludes top level directories to skip
   * @param packageVisitorPool the thread pool to use to visit packages in parallel. May be null.
   * @param observer is called for each path fragment found. May be called
   *    from multiple threads concurrently, and therefore must be thread-safe.
   * @throws InterruptedException if the calling thread was interrupted.
   */
  public void visitPackageNamesRecursively(PathFragment directory, EventHandler eventHandler,
      AtomicReference<? extends UnixGlob.FilesystemCalls> cache, Set<String> topLevelExcludes,
      ThreadPoolExecutor packageVisitorPool,
      final AcceptsPathFragment observer) throws InterruptedException {
    // <p>TODO(bazel-team): (2009) this method currently doesn't guarantee that all BUILD files
    // it returns correspond to valid package names, therefore the caller must call
    // Label.validatePackageName (or equivalent).  (Furthermore, the PackageCache may consider
    // some of these packages deleted.)
    Preconditions.checkNotNull(directory);
    Preconditions.checkNotNull(eventHandler);
    Preconditions.checkArgument(!directory.isAbsolute());
    Preconditions.checkArgument(directory.isNormalized());

    boolean shutdownOnCompletion = false;
    if (packageVisitorPool == null) {
      shutdownOnCompletion = true;
      packageVisitorPool =  new ThreadPoolExecutor(200, 200,
          0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(),
          new ThreadFactoryBuilder().setNameFormat("visit-packages-recursive-%d").build());
    }
    PackageNameVisitor visitor = new PackageNameVisitor(eventHandler, cache,
                                                        shutdownOnCompletion, packageVisitorPool) {
      @Override protected void visitPackageName(PathFragment pkgName) {
        observer.accept(pkgName);
      }
    };

    for (Path root : pathEntries) {
      visitor.enqueue(root, root.getRelative(directory), topLevelExcludes);
    }

    visitor.work(true);
  }

  /**
   * Same as {@link #visitPackageNamesRecursively(PathFragment, EventHandler,
   * UnixGlob.FilesystemCalls, Set, ThreadPoolExecutor, AcceptsPathFragment)}, with an empty set of
   * excludes and the {@link UnixGlob#DEFAULT_SYSCALLS}.
   */
  void visitPackageNamesRecursively(PathFragment directory, EventHandler eventHandler,
      final AcceptsPathFragment observer) throws InterruptedException {
    visitPackageNamesRecursively(directory, eventHandler,
        new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
        ImmutableSet.<String>of(), null, observer);
  }

  /**
   * A concurrent, recursive visitor over all package names
   * under a given directory.
   *
   * If an uncaught RuntimeException is encountered during the visitation,
   * subsequent traversal is stopped (fail-fast). The preferred method
   * of aborting the traversal is {@link Thread#interrupt()} on the
   * calling thread.
   */
  private abstract class PackageNameVisitor extends AbstractQueueVisitor {
    private final Set<Path> visitedDirs = Sets.newConcurrentHashSet();
    private final Set<PathFragment> visitedFrags = Sets.newConcurrentHashSet();
    private final EventHandler eventHandler;
    private final AtomicReference<? extends UnixGlob.FilesystemCalls> cache;

    private PackageNameVisitor(EventHandler eventHandler,
        AtomicReference<? extends UnixGlob.FilesystemCalls> cache,
        boolean shutdownOnCompletion, ThreadPoolExecutor packageVisitorPool) {
      super(packageVisitorPool, shutdownOnCompletion, /*failFast=*/true,
            /*failFastOnInterrupt=*/true);
      this.eventHandler = eventHandler;
      this.cache = cache;
    }

    /**
     * @param root Must be one of the pathEntries.
     * @param directory Must be beneath the given pathEntry.
     */
    public void enqueue(final Path root, final Path directory, final Set<String> topLevelExcludes) {
      if (visitedDirs.add(directory)) {
        super.enqueue(new Runnable() {
          @Override
          public void run() {
            try {
              // We only traverse directories that are not symlinks.
              if (!directory.isDirectory() || directory.isSymbolicLink()) {
                return;
              }

              Collection<Dirent> dirents = cache.get().readdir(directory, Symlinks.FOLLOW);
              for (Dirent dirent : dirents) {
                String basename = dirent.getName();
                if (topLevelExcludes.contains(basename)) {
                  continue;
                }
                if (dirent.getType() == Dirent.Type.FILE) {
                  if ("BUILD".equals(basename)) {
                    PathFragment pkgName = directory.relativeTo(root);
                    if (visitedFrags.add(pkgName)) {
                      visitPackageName(pkgName);
                    }
                  }
                } else {
                  enqueue(root, directory.getChild(basename), ImmutableSet.<String>of());
                }
              }
            } catch (IOException e) {
              // The specified directory can not be found, or there is some kind of
              // I/O error that occurs while trying to scan the directory. For example,
              // bug "Blaze crashes during TargetLabelParser.findTargetsBeneathDirectory" shows
              // a crash when a file in the package path cannot be
              // read due to a permissions error. Rather than an assertion error
              // that creates a stack dump, we should generate a valid error message.
              // To do this, we pass the error up to the package iterator, so that
              // it can (correctly) stop when it encounters an I/O error.
              eventHandler.handle(Event.error("I/O error searching '" + directory
                  + "' for BUILD files: " + e.getMessage()));
            }
          }
        });
      }
    }

    /**
     * Called exactly once for each package name found.
     * @param pkgName The package name PathFragment.
     */
    protected abstract void visitPackageName(PathFragment pkgName);

    @Override
    public void work(boolean interruptWorkers) throws InterruptedException {
      super.work(interruptWorkers);
    }
  }

  /**
   * Returns an immutable ordered list of the directories on the package path.
   */
  public ImmutableList<Path> getPathEntries() {
    return pathEntries;
  }

  @Override
  public String toString() {
    return "PathPackageLocator" + pathEntries;
  }

  /**
   * A factory of PathPackageLocators from a list of path elements.  Elements
   * may contain "%workspace%", indicating the workspace.
   *
   * @param pathElements Each element must be an absolute path, relative path,
   *                     or some string "%workspace%" + relative, where relative is itself a
   *                     relative path.  The special symbol "%workspace%" means to interpret
   *                     the path relative to the nearest enclosing workspace.  Relative
   *                     paths are interpreted relative to the client's working directory,
   *                     which may be below the workspace.
   * @param eventHandler The eventHandler.
   * @param workspace The nearest enclosing package root directory.
   * @param clientWorkingDirectory The client's working directory.
   * @return a list of {@link Path}s.
   */
  public static PathPackageLocator create(List<String> pathElements,
                                          EventHandler eventHandler,
                                          Path workspace,
                                          Path clientWorkingDirectory) {
    List<Path> resolvedPaths = new ArrayList<>();
    final String workspaceWildcard = "%workspace%";

    for (String pathElement : pathElements) {
      // Replace "%workspace%" with the path of the enclosing workspace directory.
      pathElement = pathElement.replace(workspaceWildcard, workspace.getPathString());

      PathFragment pathElementFragment = new PathFragment(pathElement);

      // If the path string started with "%workspace%" or "/", it is already absolute,
      // so the following line is a no-op.
      Path rootPath = clientWorkingDirectory.getRelative(pathElementFragment);

      if (!pathElementFragment.isAbsolute() && !clientWorkingDirectory.equals(workspace)) {
        eventHandler.handle(
            Event.warn("The package path element '" + pathElementFragment + "' will be "
                + "taken relative to your working directory. You may have intended "
                + "to have the path taken relative to your workspace directory. "
                + "If so, please use the '" + workspaceWildcard + "' wildcard."));
      }

      if (rootPath.exists()) {
        resolvedPaths.add(rootPath);
      } else {
        LOG.fine("package path element " + rootPath + " does not exist, ignoring");
      }
    }
    return new PathPackageLocator(resolvedPaths);
  }

}
