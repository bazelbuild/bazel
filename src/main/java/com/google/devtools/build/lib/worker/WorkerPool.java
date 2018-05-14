// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.concurrent.ThreadSafe;

/**
 * A worker pool that spawns multiple workers and delegates work to them.
 *
 * <p>This is useful when the worker cannot handle multiple parallel requests on its own and we need
 * to pre-fork a couple of them instead.
 */
@ThreadSafe
final class WorkerPool {
  private final AtomicInteger highPriorityWorkersInUse = new AtomicInteger(0);
  private final ImmutableSet<String> highPriorityWorkerMnemonics;
  private final ImmutableMap<String, Integer> config;
  private final ImmutableMap<Integer, SimpleWorkerPool> pools;

  /**
   * @param factory worker factory
   * @param config pool configuration; max number of workers per worker mnemonic; the empty string
   *     key specifies the default maximum
   * @param highPriorityWorkers mnemonics of high priority workers
   */
  public WorkerPool(
      WorkerFactory factory, Map<String, Integer> config, Iterable<String> highPriorityWorkers) {
    highPriorityWorkerMnemonics = ImmutableSet.copyOf(highPriorityWorkers);
    this.config = ImmutableMap.copyOf(config);
    ImmutableMap.Builder<Integer, SimpleWorkerPool> poolsBuilder = ImmutableMap.builder();
    for (Integer max : new HashSet<>(config.values())) {
      poolsBuilder.put(max, new SimpleWorkerPool(factory, makeConfig(max)));
    }
    pools = poolsBuilder.build();
  }

  private WorkerPoolConfig makeConfig(int max) {
    WorkerPoolConfig config = new WorkerPoolConfig();

    // It's better to re-use a worker as often as possible and keep it hot, in order to profit
    // from JIT optimizations as much as possible.
    config.setLifo(true);

    // Keep a fixed number of workers running per key.
    config.setMaxIdlePerKey(max);
    config.setMaxTotalPerKey(max);
    config.setMinIdlePerKey(max);

    // Don't limit the total number of worker processes, as otherwise the pool might be full of
    // e.g. Java workers and could never accommodate another request for a different kind of
    // worker.
    config.setMaxTotal(-1);

    // Wait for a worker to become ready when a thread needs one.
    config.setBlockWhenExhausted(true);

    // Always test the liveliness of worker processes.
    config.setTestOnBorrow(true);
    config.setTestOnCreate(true);
    config.setTestOnReturn(true);

    // No eviction of idle workers.
    config.setTimeBetweenEvictionRunsMillis(-1);

    return config;
  }

  private SimpleWorkerPool getPool(WorkerKey key) {
    Integer max = config.get(key.getMnemonic());
    if (max == null) {
      max = config.get("");
    }
    return pools.get(max);
  }

  /**
   * Gets a worker.
   *
   * @param key worker key
   * @return a worker
   */
  public Worker borrowObject(WorkerKey key) throws IOException, InterruptedException {
    Worker result;
    try {
      result = getPool(key).borrowObject(key);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, IOException.class, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }

    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      highPriorityWorkersInUse.incrementAndGet();
    } else {
      try {
        waitForHighPriorityWorkersToFinish();
      } catch (InterruptedException e) {
        returnObject(key, result);
        throw e;
      }
    }

    return result;
  }

  public void returnObject(WorkerKey key, Worker obj) {
    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      decrementHighPriorityWorkerCount();
    }
    getPool(key).returnObject(key, obj);
  }

  public void invalidateObject(WorkerKey key, Worker obj) throws IOException, InterruptedException {
    if (highPriorityWorkerMnemonics.contains(key.getMnemonic())) {
      decrementHighPriorityWorkerCount();
    }
    try {
      getPool(key).invalidateObject(key, obj);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, IOException.class, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }
  }

  // Decrements the high-priority workers counts and pings waiting threads if appropriate.
  private void decrementHighPriorityWorkerCount() {
    if (highPriorityWorkersInUse.decrementAndGet() <= 1) {
      synchronized (highPriorityWorkersInUse) {
        highPriorityWorkersInUse.notifyAll();
      }
    }
  }

  // Returns once less than two high-priority workers are running.
  private void waitForHighPriorityWorkersToFinish() throws InterruptedException {
    // Fast path for the case where the high-priority workers feature is not in use.
    if (highPriorityWorkerMnemonics.isEmpty()) {
      return;
    }

    while (highPriorityWorkersInUse.get() > 1) {
      synchronized (highPriorityWorkersInUse) {
        highPriorityWorkersInUse.wait();
      }
    }
  }

  public void close() {
    for (SimpleWorkerPool pool : pools.values()) {
      pool.close();
    }
  }
}
