// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.io.IOException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A {@link BulkDeleter} that moves each stale output tree into a trash directory on the same file
 * system and deletes it on a background thread, so an action's prepare step costs one rename per
 * tree regardless of the tree's size or shape. Once the build is over, the remaining trash is
 * deleted on as many threads as there are processors.
 *
 * <p>Every invocation gets its own trash directory under {@code trashRoot}. Trash left by an
 * earlier invocation that did not finish is deleted in the background first.
 */
public final class LocalBulkDeleter implements BulkDeleter {
  private final Path execRoot;
  private final Path trashRoot;
  private final Path trash;
  private final AtomicLong trashCount = new AtomicLong();
  private final ThreadPoolExecutor reaper =
      new ThreadPoolExecutor(
          1,
          1,
          10,
          TimeUnit.SECONDS,
          new LinkedBlockingQueue<>(),
          new ThreadFactoryBuilder()
              .setNameFormat("local-bulk-delete")
              .setDaemon(true)
              .setPriority(Thread.MIN_PRIORITY)
              .build());

  public LocalBulkDeleter(Path execRoot, Path trashRoot, String invocationId) {
    this.execRoot = checkNotNull(execRoot);
    this.trashRoot = checkNotNull(trashRoot);
    this.trash = trashRoot.getChild(invocationId);
    // An invocation that never reaches finalizeBuild must not leave an idle thread behind.
    reaper.allowCoreThreadTimeOut(true);
    reaper.execute(this::deleteEarlierTrash);
  }

  @Override
  public void bulkDelete(Iterable<PathFragment> paths) throws IOException {
    for (PathFragment fragment : paths) {
      Path path = execRoot.getRelative(fragment);
      try {
        // A file, a symlink, a missing path, and an empty directory are one remove(3).
        path.delete();
        continue;
      } catch (IOException ignored) {
        // ENOTEMPTY, or the parent directory is not writable.
      }
      Path parent = path.getParentDirectory();
      if (parent != null && parent.startsWith(execRoot)) {
        try {
          parent.setWritable(true);
        } catch (IOException ignored) {
          // The delete below reports the real failure.
        }
      }
      if (!path.isDirectory(Symlinks.NOFOLLOW) || !moveToTrash(path)) {
        path.deleteTree();
      }
    }
  }

  /** Waits until every tree moved to the trash so far is deleted. */
  public void awaitPendingDeletes() throws InterruptedException {
    int threads = Runtime.getRuntime().availableProcessors();
    reaper.setMaximumPoolSize(threads);
    reaper.setCorePoolSize(threads);
    reaper.shutdown();
    try (SilentCloseable c = Profiler.instance().profile("LocalBulkDeleter.awaitPendingDeletes")) {
      reaper.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
    }
  }

  private boolean moveToTrash(Path path) {
    Path target = trash.getChild(Long.toString(trashCount.getAndIncrement()));
    try {
      trash.createDirectoryAndParents();
      // Moving a directory to a new parent rewrites its ".." entry, which needs write access.
      path.setWritable(true);
      path.renameTo(target);
    } catch (IOException e) {
      return false;
    }
    try {
      reaper.execute(() -> deleteQuietly(target));
    } catch (RejectedExecutionException e) {
      deleteQuietly(target);
    }
    return true;
  }

  private void deleteEarlierTrash() {
    try {
      if (!trashRoot.isDirectory(Symlinks.NOFOLLOW)) {
        return;
      }
      for (Path earlier : trashRoot.getDirectoryEntries()) {
        if (!earlier.equals(trash)) {
          deleteQuietly(earlier);
        }
      }
    } catch (IOException ignored) {
      // The next invocation retries.
    }
  }

  private static void deleteQuietly(Path path) {
    try (SilentCloseable c = Profiler.instance().profile("LocalBulkDeleter.deleteTree")) {
      path.deleteTree();
    } catch (IOException ignored) {
      // The tree stays in the trash and the next invocation retries.
    }
  }
}
