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
import static com.google.devtools.build.lib.remote.util.Utils.bytesCountToDisplayString;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.util.FileSystemLock;
import com.google.devtools.build.lib.util.FileSystemLock.LockMode;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;

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

    // Sort older entries before newer ones, tie breaking by path. This causes AC entries to be
    // sorted before CAS entries with the same age, making it less likely for garbage collection
    // to break referential integrity in the event that mtime resolution is insufficient.
    private static final Comparator<Entry> COMPARATOR =
        (x, y) ->
            ComparisonChain.start()
                .compare(x.mtime(), y.mtime())
                .compare(x.path(), y.path())
                .result();

    /**
     * Returns the entries to be deleted.
     *
     * @param entries the full list of entries
     */
    List<Entry> getEntriesToDelete(List<Entry> entries) {
      entries.sort(COMPARATOR);

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

  private record DeletionStats(long deletedEntries, long deletedBytes, boolean concurrentUpdate) {}

  /** Stats for a garbage collection run. */
  public record CollectionStats(
      long totalEntries,
      long totalBytes,
      long deletedEntries,
      long deletedBytes,
      boolean concurrentUpdate,
      Duration elapsedTime) {

    /** Returns a human-readable summary. */
    public String displayString() {
      double elapsedSeconds = elapsedTime.toSecondsPart() + elapsedTime.toMillisPart() / 1000.0;
      int filesPerSecond = (int) Math.round((double) deletedEntries / elapsedSeconds);
      int mbPerSecond = (int) Math.round((deletedBytes / (1024.0 * 1024.0)) / elapsedSeconds);

      return "Deleted %d of %d files, reclaimed %s of %s in %.2f seconds (%d files/s, %d MB/s)%s"
          .formatted(
              deletedEntries(),
              totalEntries(),
              bytesCountToDisplayString(deletedBytes()),
              bytesCountToDisplayString(totalBytes()),
              elapsedSeconds,
              filesPerSecond,
              mbPerSecond,
              concurrentUpdate() ? " (concurrent update detected)" : "");
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
  public CollectionStats run() throws IOException, InterruptedException {
    // Acquire an exclusive lock to prevent two Bazel processes from simultaneously running
    // garbage collection, which can waste resources and lead to incorrect results.
    try (var lock = FileSystemLock.tryGet(root.getRelative("gc/lock"), LockMode.EXCLUSIVE)) {
      return runUnderLock();
    }
  }

  private CollectionStats runUnderLock() throws IOException, InterruptedException {
    Instant startTime = Instant.now();
    EntryScanner scanner = new EntryScanner();
    EntryDeleter deleter = new EntryDeleter();

    List<Entry> allEntries = scanner.scan();
    List<Entry> entriesToDelete = policy.getEntriesToDelete(allEntries);

    for (Entry entry : entriesToDelete) {
      deleter.delete(entry);
    }

    DeletionStats deletionStats = deleter.await();
    Duration elapsedTime = Duration.between(startTime, Instant.now());

    return new CollectionStats(
        allEntries.size(),
        allEntries.stream().mapToLong(Entry::size).sum(),
        deletionStats.deletedEntries(),
        deletionStats.deletedBytes(),
        deletionStats.concurrentUpdate(),
        elapsedTime);
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
      } catch (UncheckedIOException e) {
        throw e.getCause();
      }
      return entries;
    }

    private void visitDirectory(Path path) {
      try {
        for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
          Path childPath = path.getChild(dirent.getName());
          if (dirent.getType().equals(Dirent.Type.FILE)) {
            // The file may be gone by the time we stat it.
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
        throw new UncheckedIOException(e);
      }
    }
  }

  /** Deletes disk cache entries, performing I/O in parallel. */
  private final class EntryDeleter extends AbstractQueueVisitor {
    private final LongAdder deletedEntries = new LongAdder();
    private final LongAdder deletedBytes = new LongAdder();
    private final AtomicBoolean concurrentUpdate = new AtomicBoolean(false);

    EntryDeleter() {
      super(
          executorService,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
    }

    /** Enqueues an entry to be deleted. */
    void delete(Entry entry) {
      execute(
          () -> {
            Path path = root.getRelative(entry.path());
            try {
              FileStatus status = path.statIfFound();
              if (status == null) {
                // The entry is already gone.
                concurrentUpdate.set(true);
                return;
              }
              if (status.getLastModifiedTime() != entry.mtime()) {
                // The entry was likely accessed by a build since we statted it.
                concurrentUpdate.set(true);
                return;
              }
              if (path.delete()) {
                deletedEntries.increment();
                deletedBytes.add(entry.size());
              } else {
                // The entry is already gone.
                concurrentUpdate.set(true);
              }
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          });
    }

    /** Waits for all enqueued deletions to complete. */
    DeletionStats await() throws IOException, InterruptedException {
      try {
        awaitQuiescence(true);
      } catch (UncheckedIOException e) {
        throw e.getCause();
      }
      return new DeletionStats(deletedEntries.sum(), deletedBytes.sum(), concurrentUpdate.get());
    }
  }
}
