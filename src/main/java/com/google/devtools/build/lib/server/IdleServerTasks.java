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

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.util.StringUtilities;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Run cleanup-related tasks during idle periods in the server.
 * idle() and busy() must be called in that order, and only once.
 */
class IdleServerTasks {
  private final ScheduledThreadPoolExecutor executor;
  private static final Logger logger = Logger.getLogger(IdleServerTasks.class.getName());

  /** Must be called from the main thread. */
  public IdleServerTasks() {
    this.executor = new ScheduledThreadPoolExecutor(
        1,
        new ThreadFactoryBuilder().setNameFormat("idle-server-tasks-%d").build());
  }

  /**
   * Called when the server becomes idle. Should not block, but may invoke
   * new threads.
   */
  public void idle() {
    Preconditions.checkState(!executor.isShutdown());

    @SuppressWarnings("unused")
    Future<?> possiblyIgnoredError =
        executor.schedule(
            () -> {
              MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
              MemoryUsage before = memBean.getHeapMemoryUsage();
              try (AutoProfiler p = AutoProfiler.logged("Idle GC", logger)) {
                System.gc();
              }
              MemoryUsage after = memBean.getHeapMemoryUsage();
              logger.info(
                  String.format(
                      "[Idle GC] used: %s -> %s, committed: %s -> %s",
                      StringUtilities.prettyPrintBytes(before.getUsed()),
                      StringUtilities.prettyPrintBytes(after.getUsed()),
                      StringUtilities.prettyPrintBytes(before.getCommitted()),
                      StringUtilities.prettyPrintBytes(after.getCommitted())));
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
}
