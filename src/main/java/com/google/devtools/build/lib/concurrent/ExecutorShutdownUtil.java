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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ThreadFactoryBuilder;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Utilities for safely shutting down executors.
 * TODO(bazel-team): Rename this class to something like "ExecutorUtil".
 */
public class ExecutorShutdownUtil {

  private ExecutorShutdownUtil() {
  }

  /**
   * Shutdown the executor. If an interrupt occurs, invoke shutdownNow(), but
   * still block on the eventual termination of the pool.
   *
   * @param executor the executor service.
   * @return true iff interrupted.
   */
  public static boolean interruptibleShutdown(ExecutorService executor) {
    return shutdownImpl(executor, /*interruptible=*/true);
  }

  /**
   * Shutdown the executor. If an interrupt occurs, ignore it and still block on the eventual
   * termination of the pool. This way, all tasks are guaranteed to have completed normally.
   *
   * @param executor the executor service.
   * @return true iff interrupted.
   */
  public static boolean uninterruptibleShutdown(ExecutorService executor) {
    return shutdownImpl(executor, /*interruptible=*/false);
  }

  private static boolean shutdownImpl(ExecutorService executor, boolean interruptible) {
    Preconditions.checkState(!executor.isShutdown());
    executor.shutdown();

    // Common pattern: check for interrupt, but don't return until all threads
    // are finished.
    boolean interrupted = false;
    while (true) {
      try {
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
        break;
      } catch (InterruptedException e) {
        if (interruptible) {
          executor.shutdownNow();
        }
        interrupted = true;
      }
    }
    return interrupted;
  }

  /**
   * Create a "slack" thread pool which has the following properties:
   * 1. the worker count shrinks as the threads go unused
   * 2. the rejection policy is caller-runs
   *
   * @param threads maximum number of threads in the pool
   * @param name name of the pool
   * @return the new ThreadPoolExecutor
   */
  public static ThreadPoolExecutor newSlackPool(int threads, String name) {
    // Using a synchronous queue with a bounded thread pool means we'll reject
    // tasks after the pool size. The CallerRunsPolicy, however, implies that
    // saturation is handled in the calling thread.
    ThreadPoolExecutor pool = new ThreadPoolExecutor(threads, threads, 3L, TimeUnit.SECONDS,
        new SynchronousQueue<Runnable>());
    // Do not consume threads when not in use.
    pool.allowCoreThreadTimeOut(true);
    pool.setThreadFactory(new ThreadFactoryBuilder().setNameFormat(name + " %d").build());
    pool.setRejectedExecutionHandler(new RejectedExecutionHandler() {
      @Override
      public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
        r.run();
      }
    });
    return pool;
  }
}
