// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe;


import com.google.common.base.Preconditions;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.nio.file.ClosedWatchServiceException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.NotDirectoryException;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardWatchEventKinds;
import java.nio.file.WatchEvent;
import java.nio.file.WatchEvent.Kind;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * File system watcher for local filesystems. It's able to provide a list of changed files between
 * two consecutive calls. Uses the standard Java WatchService, which uses 'inotify' on Linux.
 */
public final class WatchServiceDiffAwareness extends LocalDiffAwareness {
  /**
   * Bijection from WatchKey to the (absolute) Path being watched. WatchKeys don't have this
   * functionality built-in so we do it ourselves.
   */
  private final HashBiMap<WatchKey, Path> watchKeyToDirBiMap = HashBiMap.create();

  private final boolean isWindows = OS.getCurrent() == OS.WINDOWS;

  /** Every directory is registered under this watch service. */
  private WatchService watchService;

  private final IgnoredSubdirectories ignoredPaths;

  WatchServiceDiffAwareness(Path watchRoot, IgnoredSubdirectories ignoredPaths) {
    super(watchRoot);
    this.ignoredPaths = ignoredPaths;
  }

  private void init() {
    Preconditions.checkState(watchService == null);
    try {
      watchService = watchRoot.getFileSystem().newWatchService();
    } catch (IOException ignored) {
      // According to the docs, this can never happen with the default file system provider.
    }
  }

  @Override
  public View getCurrentView(OptionsProvider options) throws BrokenDiffAwarenessException {
    // We need to consider 4 cases for watchFs:
    // previous view    current view
    //  disabled         disabled  -> EVERYTHING_MODIFIED
    //  disabled         enabled   -> valid View (1)
    //  enabled          disabled  -> throw BrokenDiffAwarenessException
    //  enabled          enabled   -> valid View
    //
    // (1) When watchFs gets enabled, we need to consider both the delta from the previous view
    //     to the current view (1a), and from the current view to the next view (1b).
    // (1a) If watchFs was previously disabled, then previous view was either EVERYTHING_MODIFIED,
    //      or we threw a BrokenDiffAwarenessException. The first is safe because comparing it to
    //      any view results in ModifiedFileSet.EVERYTHING_MODIFIED. The second is safe because
    //      the previous diff awareness gets closed and we're now in a new instance; comparisons
    //      between views with different owners always results in
    //      ModifiedFileSet.EVERYTHING_MODIFIED.
    // (1b) On the next run, we want to see the files that were modified between the current and the
    //      next run. For that, the view we return needs to be valid; however, it's ok for it to
    //      contain files that are modified between init() and poll() below, because those are
    //      already taken into account for the current build, as we ended up with
    //      ModifiedFileSet.EVERYTHING_MODIFIED in the current build.
    boolean watchFs =
        options.getOptions(Options.class).getWatchFS()
            &&
            // Guard WatchFs on Windows behind --experimental_windows_watchfs.
            (!isWindows || options.getOptions(Options.class).getWindowsWatchFS());
    if (watchFs && watchService == null) {
      init();
    } else if (!watchFs && (watchService != null)) {
      close();
      // The contract is that throwing BrokenDiffAwarenessException prevents reuse of the same
      // diff awareness object.
      // Consider this sequence of builds:
      // 1. build --watchfs    // startup the listener
      // 2. build --nowatchfs  // shutdown the listener
      // 3. build --watchfs    // startup the listener
      //
      // In the third build, we have to be careful not to reuse information from the first build,
      // since we don't know what changed between the second and third builds. One way to ensure
      // that is to carefully ensure that we increment the iteration numbers on every call;
      // LocalDiffAwareness will only return a Diff if the Views are in sequential order. The other
      // is to not reuse the DiffAwareness object, but create a new one; the DiffAwarenessManager
      // always assumes EVERYTHING_MODIFIED for different objects. That seems safer, so we're using
      // that here.
      throw new BrokenDiffAwarenessException("Switched off --watchfs again");
    }
    // If init() failed, then this if also applies.
    if (watchService == null) {
      return EVERYTHING_MODIFIED;
    }
    if (isFirstCall()) {
      try {
        registerSubDirectories(watchRoot);
      } catch (IOException e) {
        close();
        throw new BrokenDiffAwarenessException(
            "Error encountered with local file system watcher " + e);
      }
      return newView(ImmutableSet.of());
    }

    ChangesResult changesResult;
    try {
      changesResult = collectChanges();
    } catch (BrokenDiffAwarenessException e) {
      close();
      throw e;
    } catch (IOException e) {
      close();
      throw new BrokenDiffAwarenessException(
          "Error encountered with local file system watcher " + e);
    } catch (ClosedWatchServiceException e) {
      throw new BrokenDiffAwarenessException(
          "Internal error with the local file system watcher " + e);
    }
    if (changesResult.overflow) {
      return newOverflowView();
    }
    return newView(changesResult.changedPaths);
  }

  private static class ChangesResult {
    private final Set<Path> changedPaths;
    private final boolean overflow;

    private ChangesResult(Set<Path> changedPaths, boolean overflow) {
      this.changedPaths = changedPaths;
      this.overflow = overflow;
    }
  }

  @Override
  public void close() {
    if (watchService != null) {
      try {
        watchService.close();
      } catch (IOException ignored) {
        // Nothing we can do here.
      }
    }
  }

  /** Returns the changed files caught by the watch service. */
  private ChangesResult collectChanges() throws BrokenDiffAwarenessException, IOException {
    Set<Path> createdFilesAndDirectories = new HashSet<>();
    Set<Path> deletedOrModifiedFilesAndDirectories = new HashSet<>();
    Set<Path> deletedTrackedDirectories = new HashSet<>();
    boolean overflow = false;

    WatchKey watchKey;
    while ((watchKey = watchService.poll()) != null) {
      Path dir = watchKeyToDirBiMap.get(watchKey);
      if (dir == null) {
        for (WatchEvent<?> event : watchKey.pollEvents()) {
          if (event.kind().equals(StandardWatchEventKinds.OVERFLOW)) {
            overflow = true;
          }
        }
        watchKey.reset();
        continue;
      }

      // We replay all the events for this watched directory in chronological order and
      // construct the diff of this directory since the last #collectChanges call.
      for (WatchEvent<?> event : watchKey.pollEvents()) {
        Kind<?> kind = event.kind();
        if (kind.equals(StandardWatchEventKinds.OVERFLOW)) {
          overflow = true;
          continue;
        }
        if (event.context() == null) {
          // The WatchService documentation mentions that WatchEvent#context may return null, but
          // doesn't explain how/why it would do so. Looking at the implementation, it only
          // happens on an overflow event. But we make no assumptions about that implementation
          // detail here.
          throw new BrokenDiffAwarenessException(
              "Insufficient information from local file system watcher");
        }
        // For the events we've registered, the context given is a relative path.
        Path relativePath = (Path) event.context();
        Path path = dir.resolve(relativePath);
        Preconditions.checkState(path.isAbsolute(), path);
        if (kind == StandardWatchEventKinds.ENTRY_CREATE) {
          createdFilesAndDirectories.add(path);
          deletedOrModifiedFilesAndDirectories.remove(path);
        } else if (kind == StandardWatchEventKinds.ENTRY_DELETE) {
          createdFilesAndDirectories.remove(path);
          deletedOrModifiedFilesAndDirectories.add(path);
          WatchKey deletedDirectoryKey = watchKeyToDirBiMap.inverse().get(path);
          if (deletedDirectoryKey != null) {
            // If the deleted directory has children, then there will also be events for the
            // WatchKey of the directory itself. WatchService#poll doesn't specify the order in
            // which WatchKeys are returned, so the key for the directory itself may be processed
            // *after* the current key (the parent of the deleted directory), and so we don't want
            // to remove the deleted directory from our bimap just yet.
            //
            // For example, suppose we have the file '/root/a/foo.txt' and are watching the
            // directories '/root' and '/root/a'. If the directory '/root/a' gets deleted then the
            // following is a valid sequence of events by key.
            //
            // WatchKey '/root/'
            // WatchEvent EVENT_MODIFY 'a'
            // WatchEvent EVENT_DELETE 'a'
            // WatchKey '/root/a'
            // WatchEvent EVENT_DELETE 'foo.txt'
            deletedTrackedDirectories.add(path);
            // Since inotify uses inodes under the covers we cancel our registration on this key to
            // avoid getting WatchEvents from a new directory that happens to have the same inode.
            deletedDirectoryKey.cancel();
          }
        } else if (kind == StandardWatchEventKinds.ENTRY_MODIFY) {
          // If a file was created and then modified, then the net diff is that it was
          // created.
          if (!createdFilesAndDirectories.contains(path)) {
            deletedOrModifiedFilesAndDirectories.add(path);
          }
        }
      }

      if (!watchKey.reset()) {
        // Watcher got deleted, directory no longer valid.
        watchKeyToDirBiMap.remove(watchKey);
      }
    }

    for (Path path : deletedTrackedDirectories) {
      WatchKey staleKey = watchKeyToDirBiMap.inverse().get(path);
      watchKeyToDirBiMap.remove(staleKey);
    }
    if (watchKeyToDirBiMap.isEmpty()) {
      // No more directories to watch, something happened the root directory being watched.
      throw new IOException("Root directory " + watchRoot + " became inaccessible.");
    }

    if (overflow) {
      // Clean up any stale directory mappings that were deleted during overflow.
      Set<WatchKey> staleKeys = new HashSet<>();
      for (Map.Entry<WatchKey, Path> entry : watchKeyToDirBiMap.entrySet()) {
        if (!entry.getKey().isValid()
            || !Files.isDirectory(entry.getValue(), LinkOption.NOFOLLOW_LINKS)) {
          entry.getKey().cancel();
          staleKeys.add(entry.getKey());
        }
      }
      watchKeyToDirBiMap.keySet().removeAll(staleKeys);
      if (watchKeyToDirBiMap.isEmpty()) {
        throw new IOException("Root directory " + watchRoot + " became inaccessible.");
      }
      // Re-traverse the directory tree to discover and watch any subdirectories created during
      // the overflow.
      registerSubDirectories(watchRoot);
      return new ChangesResult(ImmutableSet.of(), /* overflow= */ true);
    }

    Set<Path> changedPaths = new HashSet<>();
    for (Path path : createdFilesAndDirectories) {
      if (Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
        // This is a new directory, so changes to it since its creation have not been watched.
        // We manually traverse the directory tree to register all the new subdirectories and find
        // all the new subdirectories and files.
        changedPaths.addAll(registerSubDirectoriesAndReturnContents(path));
      } else {
        changedPaths.add(path);
      }
    }
    changedPaths.addAll(deletedOrModifiedFilesAndDirectories);
    return new ChangesResult(changedPaths, /* overflow= */ false);
  }

  /** Traverses directory tree to register subdirectories. */
  private void registerSubDirectories(Path rootDir) throws IOException {
    // Note that this does not follow symlinks.
    WatcherFileVisitor watcherFileVisitor = new WatcherFileVisitor(ignoredPaths);
    Files.walkFileTree(rootDir, watcherFileVisitor);
    watcherFileVisitor.cancelDisplacedKeys();
  }

  /**
   * Traverses directory tree to register subdirectories. Returns all paths traversed (as absolute
   * paths).
   */
  private Set<Path> registerSubDirectoriesAndReturnContents(Path rootDir) throws IOException {
    Set<Path> visitedAbsolutePaths = new HashSet<>();
    // Note that this does not follow symlinks.
    WatcherFileVisitor watcherFileVisitor =
        new WatcherFileVisitor(visitedAbsolutePaths, ignoredPaths);
    Files.walkFileTree(rootDir, watcherFileVisitor);
    watcherFileVisitor.cancelDisplacedKeys();
    return visitedAbsolutePaths;
  }

  /** File visitor used by Files.walkFileTree() upon traversing subdirectories. */
  private class WatcherFileVisitor extends SimpleFileVisitor<Path> {

    private final Set<Path> visitedAbsolutePaths;
    private final IgnoredSubdirectories ignoredPaths;

    /** Keys that were evicted from {@link #watchKeyToDirBiMap} because their path was re-bound. */
    private final Set<WatchKey> displacedKeys = new HashSet<>();

    private WatcherFileVisitor(Set<Path> visitedPaths, IgnoredSubdirectories ignoredPaths) {
      this.visitedAbsolutePaths = visitedPaths;
      this.ignoredPaths = ignoredPaths;
    }

    private WatcherFileVisitor(IgnoredSubdirectories ignoredPaths) {
      this.visitedAbsolutePaths = new HashSet<>();
      this.ignoredPaths = ignoredPaths;
    }

    /**
     * Cancels the watches that were evicted during the traversal and not re-bound to some other
     * path, so that a directory that got replaced does not keep an inotify watch alive forever.
     *
     * <p>This has to happen after the traversal, because a directory that was merely moved is
     * re-registered under its new path and keeps the very same key.
     */
    private void cancelDisplacedKeys() {
      for (WatchKey displacedKey : displacedKeys) {
        if (!watchKeyToDirBiMap.containsKey(displacedKey)) {
          displacedKey.cancel();
        }
      }
    }

    private boolean isIgnored(Path path) {
      PathFragment pathFragment =
          PathFragment.create(path.toAbsolutePath().toString()).toRelative();
      return ignoredPaths.matchingEntry(pathFragment) != null;
    }

    @Override
    public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) {
      Preconditions.checkState(path.isAbsolute(), path);
      visitedAbsolutePaths.add(path);
      return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes attrs)
        throws IOException {
      if (isIgnored(path)) {
        return FileVisitResult.SKIP_SUBTREE;
      }

      // Do not traverse the bazel-* convenience symlinks. On windows these are created as
      // junctions.
      if (isWindows && attrs.isOther()) {
        return FileVisitResult.SKIP_SUBTREE;
      }

      // It's important that we register the directory before we visit its children. This way we
      // are guaranteed to see new files/directories either on this #getDiff or the next one.
      // Otherwise, e.g., an intra-build creation of a child directory will be forever missed if it
      // happens before the directory is listed as part of the visitation.
      Preconditions.checkState(path.isAbsolute(), path);
      // Always register, even if we already have a key for this path: a key being valid does not
      // mean that it still watches the inode that is currently at this path, e.g. when a directory
      // was moved aside and replaced while we were not watching.
      WatchKey key;
      try {
        key =
            path.register(
                watchService,
                StandardWatchEventKinds.ENTRY_CREATE,
                StandardWatchEventKinds.ENTRY_MODIFY,
                StandardWatchEventKinds.ENTRY_DELETE);
      } catch (NoSuchFileException | NotDirectoryException e) {
        if (path.equals(watchRoot)) {
          throw e;
        }
        // The directory vanished while we were traversing, which routinely happens when the tree
        // is re-registered after an overflow. The parent directory is watched, so its deletion is
        // reported to us. Any other failure (most notably the watch limit being reached) must not
        // be swallowed: leaving a directory unwatched makes subsequent builds miss changes.
        return FileVisitResult.SKIP_SUBTREE;
      }
      // Registering returns the same key for all paths resolving to the same inode, so both
      // directions of the mapping may be stale after directories are moved around.
      WatchKey displacedKey = watchKeyToDirBiMap.inverse().get(path);
      if (displacedKey != null && !displacedKey.equals(key)) {
        // The directory at this path was replaced; its old key is dealt with once we know whether
        // the directory turns up elsewhere in the tree.
        displacedKeys.add(displacedKey);
      }
      watchKeyToDirBiMap.forcePut(key, path);
      visitedAbsolutePaths.add(path);
      return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
      if (!file.equals(watchRoot) && (exc instanceof NoSuchFileException || isIgnored(file))) {
        // Either deleted while we were traversing, in which case the parent directory is watched
        // and reports the deletion, or a directory we were told to ignore -- its stream is opened
        // before #preVisitDirectory gets a chance to skip it. Anything else (e.g. an unreadable
        // directory) means that we may be unable to watch part of the tree, which must not be
        // silently ignored.
        return FileVisitResult.CONTINUE;
      }
      throw exc;
    }
  }
}
