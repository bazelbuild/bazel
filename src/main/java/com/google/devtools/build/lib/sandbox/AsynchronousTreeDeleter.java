// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * Executes file system tree deletions asynchronously.
 *
 * <p>The number of threads used to process the backlog of tree deletions can be configured at any
 * time via {@link #setThreads(int)}. While a build is running, this number should be low to not use
 * precious resources that could otherwise be used for the build itself. But when the build is
 * finished, this number should be raised to quickly go through any pending deletions.
 */
public class AsynchronousTreeDeleter implements TreeDeleter {

  public static final String MOVED_TRASH_DIR = "_moved_trash_dir";

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AtomicInteger trashCount = new AtomicInteger(0);

  /** Thread pool used to execute asynchronous tree deletions; null in synchronous mode. */
  @Nullable private ThreadPoolExecutor service;

  private final Path trashBase;

  private boolean trashBaseCreated = false;

  /** Constructs a new asynchronous tree deleter backed by just one thread. */
  public AsynchronousTreeDeleter(Path trashBase) {
    logger.atInfo().log("Starting async tree deletion pool with 1 thread");

    ThreadFactory threadFactory =
        new ThreadFactoryBuilder()
            .setNameFormat("tree-deleter")
            .setDaemon(true)
            .setPriority(Thread.MIN_PRIORITY)
            .build();

    service =
        new ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(), threadFactory);

    this.trashBase = trashBase;
  }

  /**
   * Resizes the thread pool to the given number of threads.
   *
   * <p>If the pool of active threads is larger than the requested number of threads, the resize
   * will progressively happen as those active threads become inactive. If the requested size is
   * zero, this will wait for all pending deletions to complete.
   *
   * @param threads desired number of threads, or 0 to go back to synchronous deletion
   */
  void setThreads(int threads) {
    checkState(threads > 0, "Use SynchronousTreeDeleter if no async behavior is desired");
    logger.atInfo().log("Resizing async tree deletion pool to %d threads", threads);
    checkNotNull(service, "Cannot call setThreads after shutdown").setMaximumPoolSize(threads);
  }

  @Override
  public void deleteTree(Path path) throws IOException {
    if (!trashBaseCreated) {
      trashBase.createDirectory();
      trashBaseCreated = true;
    }
    if (!path.exists()) {
      return;
    }
    Path trashPath = trashBase.getRelative(Integer.toString(trashCount.getAndIncrement()));
    try {
      path.renameTo(trashPath);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to rename %s -> %s for asynchronous removal. Removing synchronously.",
          path, trashPath);
      path.deleteTree();
      return;
    }
    checkNotNull(service, "Cannot call deleteTree after shutdown")
        .execute(
            () -> {
              try (SilentCloseable c = Profiler.instance().profile("trashPath.deleteTree")) {
                trashPath.deleteTree();
              } catch (IOException e) {
                logger.atWarning().withCause(e).log(
                    "Failed to delete tree %s asynchronously", path);
              }
            });
  }

  @Override
  public void shutdown() {
    if (service != null) {
      logger.atInfo().log("Finishing %d pending async tree deletions", service.getTaskCount());
      service.shutdown();
      service = null;
    }
  }

  public Path getTrashBase() {
    return trashBase;
  }
}
