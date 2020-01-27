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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Executes file system tree deletions asynchronously.
 *
 * <p>The number of threads used to process the backlog of tree deletions can be configured at any
 * time via {@link #setThreads(int)}. While a build is running, this number should be low to not use
 * precious resources that could otherwise be used for the build itself. But when the build is
 * finished, this number should be raised to quickly go through any pending deletions.
 */
class AsynchronousTreeDeleter implements TreeDeleter {

  private static final Logger logger = Logger.getLogger(TreeDeleter.class.getName());

  /** Thread pool used to execute asynchronous tree deletions; null in synchronous mode. */
  @Nullable private ThreadPoolExecutor service;

  /** Constructs a new asynchronous tree deleter backed by just one thread. */
  AsynchronousTreeDeleter() {
    logger.info("Starting async tree deletion pool with 1 thread");

    ThreadFactory threadFactory =
        new ThreadFactoryBuilder()
            .setNameFormat("tree-deleter")
            .setDaemon(true)
            .setPriority(Thread.MIN_PRIORITY)
            .build();

    service =
        new ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(), threadFactory);
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
    logger.info("Resizing async tree deletion pool to " + threads + " threads");
    checkNotNull(service, "Cannot call setThreads after shutdown").setMaximumPoolSize(threads);
  }

  @Override
  public void deleteTree(Path path) {
    checkNotNull(service, "Cannot call deleteTree after shutdown")
        .execute(
            () -> {
              try {
                path.deleteTree();
              } catch (IOException e) {
                logger.warning("Failed to delete tree " + path + " asynchronously: " + e);
              }
            });
  }

  @Override
  public void deleteTreesBelow(Path path) {
    checkNotNull(service, "Cannot call deleteTree after shutdown")
        .execute(
            () -> {
              try {
                path.deleteTreesBelow();
              } catch (IOException e) {
                logger.warning("Failed to delete contents of " + path + " asynchronously: " + e);
              }
            });
  }

  @Override
  public void shutdown() {
    if (service != null) {
      logger.info("Finishing " + service.getTaskCount() + " pending async tree deletions");
      service.shutdown();
      service = null;
    }
  }
}
