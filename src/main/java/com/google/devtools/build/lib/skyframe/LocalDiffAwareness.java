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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.nio.file.ClosedWatchServiceException;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardWatchEventKinds;
import java.nio.file.WatchEvent;
import java.nio.file.WatchEvent.Kind;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * File system watcher for local filesystems. It's able to provide a list of changed
 * files between two consecutive calls. Uses the standard Java WatchService, which uses
 * 'inotify' on Linux.
 */
public class LocalDiffAwareness implements DiffAwareness {

  /** Factory for creating {@link LocalDiffAwareness} instances. */
  public static class Factory implements DiffAwareness.Factory {

    private static final PathFragment LOCALFS_PREFIX = new PathFragment("/usr/local/");

    @Override
    public DiffAwareness maybeCreate(com.google.devtools.build.lib.vfs.Path pathEntry,
        ImmutableList<com.google.devtools.build.lib.vfs.Path> pathEntries) {
      com.google.devtools.build.lib.vfs.Path resolvedPathEntry;
      try {
        resolvedPathEntry = pathEntry.resolveSymbolicLinks();
      } catch (IOException e) {
        return null;
      }
      PathFragment resolvedPathEntryFragment = resolvedPathEntry.asFragment();
      // TODO(bazel-team): rely on file system stats to check whether the path is a local file
      if (!resolvedPathEntryFragment.startsWith(LOCALFS_PREFIX)) {
        return null;
      }

      LocalDiffAwareness awareness = new LocalDiffAwareness(resolvedPathEntryFragment.toString());
      try {
        awareness.createWatchService();
      } catch (IOException e) {
        return null;
      }
      return awareness;
    }
  }

  @Override
  public ModifiedFileSet getDiff() throws BrokenDiffAwarenessException {
    if (firstGetDiff) {
      firstGetDiff = false;
      return ModifiedFileSet.EVERYTHING_MODIFIED;
    }
    Set<Path> modifiedPaths;
    try {
      modifiedPaths = collectChanges();
    } catch (IOException e) {
      close();
      throw new BrokenDiffAwarenessException(
          "Error encountered with local file system watcher " + e);
    } catch (ClosedWatchServiceException e) {
      throw new BrokenDiffAwarenessException(
          "Internal error with the local file system watcher " + e);
    }
    return ModifiedFileSet.builder()
        .modifyAll(Iterables.transform(modifiedPaths, nioPathToPathFragment))
        .build();
  }

  @Override
  public void close() {
    try {
      watchService.close();
    } catch (IOException ignored) {
      // Nothing we can do here.
    }
  }

  private boolean firstGetDiff = true;

  /** WatchKeys don't have a functionality to tell the path they're watching. Doing this by hand. */
  private final Map<WatchKey, Path> keys = new HashMap<WatchKey, Path>();

  /** Package root to watch. */
  private final Path watchRootPath;

  /** Every directory is registered under this watch service. */
  private WatchService watchService;

  private LocalDiffAwareness(String watchRoot) {
    this.watchRootPath = FileSystems.getDefault().getPath(watchRoot);
  }

  /** Initializes watch service, puts a watcher into every subdirectory. */
  private void createWatchService() throws IOException {
    watchService = FileSystems.getDefault().newWatchService();
    registerSubDirectories(watchRootPath);
  }

  /** Converts java.nio.file.Path objects to vfs.PathFragment */
  private final Function<Path, PathFragment> nioPathToPathFragment =
      new Function<Path, PathFragment>() {
        @Override
        public PathFragment apply(Path input) {
          Preconditions.checkArgument(input.startsWith(watchRootPath));
          return new PathFragment(watchRootPath.relativize(input).toString());
        }
      };

  /** Returns the changed files caught by the watch service */
  private Set<Path> collectChanges() throws IOException {
    Set<Path> changedPaths = new HashSet<Path>();
    WatchKey watchKey;
    while ((watchKey = watchService.poll()) != null) {
      Path dir = keys.get(watchKey);
      Preconditions.checkArgument(dir != null);

      for (WatchEvent<?> event : watchKey.pollEvents()) {
        handleWatchEvent(dir, event, changedPaths);
      }

      if (!watchKey.reset()) {
        // Watcher got deleted, directory no longer valid.
        keys.remove(watchKey);
        if (keys.isEmpty()) {
          // No more directories to watch, something happened to package root
          throw new IOException("Package root became inaccessible.");
        }
      }
    }
    return changedPaths;
  }

  /** Handles file system events sent by inotify. */
  private void handleWatchEvent(Path dir, WatchEvent<?> event,
      Set<Path> changedPaths) throws IOException {
    Path relativePath = (Path) event.context();
    Kind<?> kind = event.kind();
    Path path = dir.resolve(relativePath);

    if (kind == StandardWatchEventKinds.ENTRY_CREATE) {
      changedPaths.add(path);
      if (Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
        registerSubDirectories(path);
      }
    }
    if (kind == StandardWatchEventKinds.ENTRY_DELETE
        || kind == StandardWatchEventKinds.ENTRY_MODIFY) {
      changedPaths.add(path);
    }
    if (kind == StandardWatchEventKinds.OVERFLOW) {
      // TODO(bazel-team): find out when an overflow might happen, and maybe handle it more gently.
      throw new IOException("Event overflow.");
    }
  }

  /** Traverses directory tree to register subdirectories */
  private void registerSubDirectories(Path path) throws IOException {
    Files.walkFileTree(path, new WatcherFileVisitor());
  }

  /** File visitor used by Files.walkFileTree() upon traversing subdirectories */
  private class WatcherFileVisitor extends SimpleFileVisitor<Path> {
    @Override
    public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes attrs)
        throws IOException {
      WatchKey key = path.register(watchService,
          StandardWatchEventKinds.ENTRY_CREATE,
          StandardWatchEventKinds.ENTRY_MODIFY,
          StandardWatchEventKinds.ENTRY_DELETE);
      keys.put(key, path);
      return FileVisitResult.CONTINUE;
    }
  }
}
