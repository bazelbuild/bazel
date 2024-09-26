// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.Comparator.comparing;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.IORuntimeException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;

/**
 * A garbage collector for the disk cache.
 *
 * <p>Garbage collection works by enumerating the entire contents of the disk cache, identifying
 * candidates for deletion according to a {@link CollectionPolicy}, and deleting them. This process
 * may take a significant amount of time on large disk caches and slow filesystems, and may be
 * interrupted at any time.
 */
public final class DiskCacheGarbageCollector {
  private static final ImmutableSet<String> EXCLUDED_DIRS = ImmutableSet.of("tmp", "gc");

  /**
   * Describes a disk cache entry.
   *
   * @param path path relative to the root directory of the disk cache
   * @param size file size in bytes
   * @param mtime file modification time
   */
  private record Entry(String path, long size, long mtime) {}

  /**
   * Determines which entries should be collected.
   *
   * @param maxSizeBytes the maximum total size in bytes, or empty for no size limit
   * @param maxAge the maximum age of cache entries, or empty for no age limit
   */
  public record CollectionPolicy(Optional<Long> maxSizeBytes, Optional<Duration> maxAge) {

    /**
     * Returns the entries to be deleted.
     *
     * @param entries the full list of entries
     */
    List<Entry> getEntriesToDelete(List<Entry> entries) {
      entries.sort(comparing(Entry::mtime));

      long excessSizeBytes = getExcessSizeBytes(entries);
      long timeCutoff = getTimeCutoff();

      int i = 0;
      for (; i < entries.size(); i++) {
        if (excessSizeBytes <= 0 && entries.get(i).mtime() >= timeCutoff) {
          break;
        }
        excessSizeBytes -= entries.get(i).size();
      }

      return entries.subList(0, i);
    }

    private long getExcessSizeBytes(List<Entry> entries) {
      if (maxSizeBytes.isEmpty()) {
        return 0;
      }
      long currentSizeBytes = entries.stream().mapToLong(Entry::size).sum();
      return currentSizeBytes - maxSizeBytes.get();
    }

    private long getTimeCutoff() {
      if (maxAge.isEmpty()) {
        return 0;
      }
      return Instant.now().minus(maxAge.get()).toEpochMilli();
    }
  }

  private final Path root;
  private final CollectionPolicy policy;
  private final ExecutorService executorService;
  private final ImmutableSet<Path> excludedDirs;

  /**
   * Creates a new garbage collector.
   *
   * @param root the root directory of the disk cache
   * @param executorService the executor service to schedule I/O operations onto
   * @param policy the garbage collection policy to use
   */
  public DiskCacheGarbageCollector(
      Path root, ExecutorService executorService, CollectionPolicy policy) {
    this.root = root;
    this.policy = policy;
    this.executorService = executorService;
    this.excludedDirs = EXCLUDED_DIRS.stream().map(root::getChild).collect(toImmutableSet());
  }

  @VisibleForTesting
  public Path getRoot() {
    return root;
  }

  @VisibleForTesting
  public CollectionPolicy getPolicy() {
    return policy;
  }

  /**
   * Runs garbage collection.
   *
   * @throws IOException if an I/O error occurred
   * @throws InterruptedException if the thread was interrupted
   */
  public void run() throws IOException, InterruptedException {
    EntryScanner scanner = new EntryScanner();
    EntryDeleter deleter = new EntryDeleter();

    List<Entry> allEntries = scanner.scan();
    List<Entry> entriesToDelete = policy.getEntriesToDelete(allEntries);

    for (Entry entry : entriesToDelete) {
      deleter.delete(root.getRelative(entry.path()));
    }

    deleter.await();
  }

  /** Lists all disk cache entries, performing I/O in parallel. */
  private final class EntryScanner extends AbstractQueueVisitor {
    private final ArrayList<Entry> entries = new ArrayList<>();

    EntryScanner() {
      super(
          executorService,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
    }

    /** Lists all disk cache entries. */
    List<Entry> scan() throws IOException, InterruptedException {
      execute(() -> visitDirectory(root));
      try {
        awaitQuiescence(true);
      } catch (IORuntimeException e) {
        throw e.getCauseIOException();
      }
      return entries;
    }

    private void visitDirectory(Path path) {
      try {
        for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
          Path childPath = path.getChild(dirent.getName());
          if (dirent.getType().equals(Dirent.Type.FILE)) {
            // The file may be gone by the time we open it.
            FileStatus status = childPath.statIfFound();
            if (status != null) {
              Entry entry =
                  new Entry(
                      childPath.relativeTo(root).getPathString(),
                      status.getSize(),
                      status.getLastModifiedTime());
              synchronized (entries) {
                entries.add(entry);
              }
            }
          } else if (dirent.getType().equals(Dirent.Type.DIRECTORY)
              && !excludedDirs.contains(childPath)) {
            execute(() -> visitDirectory(childPath));
          }
          // Deliberately ignore other file types, which should never occur in a well-formed cache.
        }
      } catch (IOException e) {
        throw new IORuntimeException(e);
      }
    }
  }

  /** Deletes disk cache entries, performing I/O in parallel. */
  private final class EntryDeleter extends AbstractQueueVisitor {
    EntryDeleter() {
      super(
          executorService,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
    }

    /** Enqueues an entry to be deleted. */
    void delete(Path path) {
      execute(
          () -> {
            try {
              path.delete();
            } catch (IOException e) {
              throw new IORuntimeException(e);
            }
          });
    }

    /** Waits for all enqueued deletions to complete. */
    void await() throws IOException, InterruptedException {
      try {
        awaitQuiescence(true);
      } catch (IORuntimeException e) {
        throw e.getCauseIOException();
      }
    }
  }
}
