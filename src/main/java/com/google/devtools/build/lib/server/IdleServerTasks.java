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

package com.google.devtools.build.lib.server;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Run cleanup-related tasks during idle periods in the server.
 * idle() and busy() must be called in that order, and only once.
 */
class IdleServerTasks {

  private final Path workspaceDir;
  private final ScheduledThreadPoolExecutor executor;
  private static final Logger logger = Logger.getLogger(IdleServerTasks.class.getName());

  private static final long FIVE_MIN_MILLIS = 1000 * 60 * 5;

  /**
   * Must be called from the main thread.
   */
  public IdleServerTasks(@Nullable Path workspaceDir) {
    this.executor = new ScheduledThreadPoolExecutor(1);
    this.workspaceDir = workspaceDir;
  }

  /**
   * Called when the server becomes idle. Should not block, but may invoke
   * new threads.
   */
  public void idle() {
    Preconditions.checkState(!executor.isShutdown());

    // Do a GC cycle while the server is idle.
    @SuppressWarnings("unused")
    Future<?> possiblyIgnoredError =
        executor.schedule(
            () -> {
              try (AutoProfiler p = AutoProfiler.logged("Idle GC", logger)) {
                System.gc();
              }
            },
            10,
            TimeUnit.SECONDS);
  }

  /**
   * Called by the main thread when the server gets to work.
   * Should return quickly.
   */
  public void busy() {
    Preconditions.checkState(!executor.isShutdown());

    // Make sure tasks are finished after shutdown(), so they do not intefere
    // with subsequent server invocations.
    executor.shutdown();
    executor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
    executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);

    boolean interrupted = false;
    while (true) {
      try {
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
        break;
      } catch (InterruptedException e) {
        // It's unsafe to leak threads - just reset the interrupt bit later.
        interrupted = true;
      }
    }

    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  /**
   * Return true iff the server should continue processing requests.
   * Called from the main thread, so it should return quickly.
   */
  public boolean continueProcessing(long idleMillis) {
    if (!memoryHeuristic(idleMillis)) {
      return false;
    }
    if (workspaceDir == null) {
      return false;
    }

    FileStatus stat;
    try {
      stat = workspaceDir.statIfFound(Symlinks.FOLLOW);
    } catch (IOException e) {
      // Do not terminate the server if the workspace is temporarily inaccessible, for example,
      // if it is on a network filesystem and the connection is down.
      return true;
    }
    return stat != null && stat.isDirectory();
  }

  private boolean memoryHeuristic(long idleMillis) {
    if (idleMillis < FIVE_MIN_MILLIS) {
      // Don't check memory health until after five minutes.
      return true;
    }

    ProcMeminfoParser memInfo = null;
    try {
      memInfo = new ProcMeminfoParser();
    } catch (IOException e) {
      logger.info("Could not process /proc/meminfo: " + e);
      return true;
    }

    long totalPhysical;
    long totalFree;
    try {
      totalPhysical = memInfo.getTotalKb();
      totalFree = memInfo.getFreeRamKb(); // See method javadoc.
    } catch (ProcMeminfoParser.KeywordNotFoundException e) {
      LoggingUtil.logToRemote(Level.WARNING,
          "Could not read memInfo during idle query", e);
      return true;
    }
    double fractionFree = (double) totalFree / totalPhysical;

    // If the system as a whole is low on memory, let this server die.
    if (fractionFree < .1) {
      logger.info("Terminating due to memory constraints");
      logger.info(String.format("Total physical:%d\nTotal free: %d\n", totalPhysical, totalFree));
      return false;
    }

    return true;
  }
}
