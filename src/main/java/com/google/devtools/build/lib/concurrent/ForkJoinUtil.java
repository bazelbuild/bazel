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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Throwables;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

/**
 * Runs work in dedicated fork/join pools, so that it neither competes with nor depends on the
 * common pool.
 */
public final class ForkJoinUtil {

  /** Computes a result using a fork/join pool that is shut down afterwards. */
  public interface PoolFunction<T> {
    T apply(ForkJoinPool pool) throws InterruptedException;
  }

  private ForkJoinUtil() {}

  /**
   * Runs {@code function} with a new fork/join pool named {@code poolName} with one thread per
   * processor, shutting the pool down afterwards.
   */
  public static <T> T withPool(String poolName, PoolFunction<T> function)
      throws InterruptedException {
    ForkJoinPool pool =
        NamedForkJoinPool.newNamedPool(poolName, Runtime.getRuntime().availableProcessors());
    boolean completedNormally = false;
    try {
      T result = function.apply(pool);
      completedNormally = true;
      return result;
    } finally {
      if (completedNormally
          ? ExecutorUtil.interruptibleShutdown(pool)
          : ExecutorUtil.uninterruptibleShutdownNow(pool)) {
        Thread.currentThread().interrupt();
      }
    }
  }

  /** Runs {@code task} in {@code pool} and waits for it, propagating failures and interrupts. */
  public static void run(ForkJoinPool pool, ForkJoinTask<?> task) throws InterruptedException {
    try {
      pool.submit(task).get();
    } catch (ExecutionException e) {
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e.getCause());
    }
  }
}
