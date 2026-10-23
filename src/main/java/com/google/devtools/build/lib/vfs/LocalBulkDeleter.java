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

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * A {@link BulkDeleter} that deletes paths under an exec root, removing the child directories of
 * each tree in parallel. Returns only after every path is gone.
 */
public final class LocalBulkDeleter implements BulkDeleter {
  private static final ExecutorService POOL =
      Executors.newFixedThreadPool(
          Runtime.getRuntime().availableProcessors(),
          new ThreadFactoryBuilder().setNameFormat("local-bulk-delete-%d").setDaemon(true).build());

  private final Path execRoot;

  public LocalBulkDeleter(Path execRoot) {
    this.execRoot = checkNotNull(execRoot);
  }

  @Override
  public void bulkDelete(Iterable<PathFragment> paths) throws IOException, InterruptedException {
    for (PathFragment fragment : paths) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path path = execRoot.getRelative(fragment);
      ImmutableList<Path> childDirectories = childDirectories(path);
      if (!childDirectories.isEmpty()) {
        path.setWritable(true);
        deleteInParallel(childDirectories);
      }
      path.deleteTree();
    }
  }

  private static ImmutableList<Path> childDirectories(Path path) {
    ImmutableList.Builder<Path> directories = ImmutableList.builder();
    try {
      if (!path.isDirectory(Symlinks.NOFOLLOW)) {
        return ImmutableList.of();
      }
      for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          directories.add(path.getChild(dirent.getName()));
        }
      }
    } catch (IOException e) {
      return ImmutableList.of();
    }
    return directories.build();
  }

  // Waits for every task even if interrupted, so no delete outlives the action's prepare.
  private static void deleteInParallel(ImmutableList<Path> directories) throws IOException {
    List<Future<?>> futures = new ArrayList<>(directories.size());
    for (Path directory : directories) {
      futures.add(
          POOL.submit(
              () -> {
                directory.deleteTree();
                return null;
              }));
    }
    IOException error = null;
    for (Future<?> future : futures) {
      try {
        Uninterruptibles.getUninterruptibly(future);
      } catch (ExecutionException e) {
        Throwables.throwIfUnchecked(e.getCause());
        IOException cause =
            e.getCause() instanceof IOException io ? io : new IOException(e.getCause());
        if (error == null) {
          error = cause;
        } else {
          error.addSuppressed(cause);
        }
      }
    }
    if (error != null) {
      throw error;
    }
  }
}
