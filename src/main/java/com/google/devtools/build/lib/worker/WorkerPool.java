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
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.concurrent.ThreadSafe;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;

/**
 * A worker pool that spawns multiple workers and delegates work to them. Allows separate
 * configuration for singleplex and multiplex workers. While the configuration is per mnemonic, the
 * actual pools need to be per WorkerKey, as different WorkerKeys may imply different process
 * startup options.
 *
 * <p>This is useful when the worker cannot handle multiple parallel requests on its own and we need
 * to pre-fork a couple of them instead. Multiplex workers <em>can</em> handle multiple parallel
 * requests, but do so through WorkerProxy instances.
 */
@ThreadSafe
final class WorkerPool {
  /** Unless otherwise specified, the max number of workers per WorkerKey. */
  private static final int DEFAULT_MAX_WORKERS = 4;
  /** Unless otherwise specified, the max number of multiplex workers per WorkerKey. */
  private static final int DEFAULT_MAX_MULTIPLEX_WORKERS = 8;
  /**
   * How many high-priority workers are currently borrowed. If greater than one, low-priority
   * workers cannot be borrowed until the high-priority ones are done.
   */
  private final AtomicInteger highPriorityWorkersInUse = new AtomicInteger(0);
  /** Which mnemonics create high-priority workers. */
  private final ImmutableSet<String> highPriorityWorkerMnemonics;
  /** Map of singleplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> workerPools;
  /** Map of multiplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> multiplexPools;

  /**
   * @param factory worker factory
   * @param config pool configuration; max number of workers per WorkerKey for each mnemonic; the
   *     empty string key specifies the default maximum
   * @param multiplexConfig like {@code config}, but for multiplex workers
   * @param highPriorityWorkers mnemonics of high priority workers
   */
  public WorkerPool(
      WorkerFactory factory,
      Map<String, Integer> config,
      Map<String, Integer> multiplexConfig,
      Iterable<String> highPriorityWorkers) {
    highPriorityWorkerMnemonics = ImmutableSet.copyOf(highPriorityWorkers);
    workerPools = createWorkerPools(factory, config, DEFAULT_MAX_WORKERS);
    multiplexPools = createWorkerPools(factory, multiplexConfig, DEFAULT_MAX_MULTIPLEX_WORKERS);
  }

  private static ImmutableMap<String, SimpleWorkerPool> createWorkerPools(
      WorkerFactory factory, Map<String, Integer> config, int defaultMaxWorkers) {
    ImmutableMap.Builder<String, SimpleWorkerPool> workerPoolsBuilder = ImmutableMap.builder();
    config.forEach(
        (key, value) ->
            workerPoolsBuilder.put(key, new SimpleWorkerPool(factory, makeConfig(value))));
    if (!config.containsKey("")) {
      workerPoolsBuilder.put("", new SimpleWorkerPool(factory, makeConfig(defaultMaxWorkers)));
    }
    return workerPoolsBuilder.build();
  }

  private static WorkerPoolConfig makeConfig(int max) {
    WorkerPoolConfig config = new WorkerPoolConfig();

    // It's better to re-use a worker as often as possible and keep it hot, in order to profit
    // from JIT optimizations as much as possible.
    config.setLifo(true);

    // Keep a fixed number of workers running per key.
    config.setMaxIdlePerKey(max);
    config.setMaxTotalPerKey(max);
    config.setMinIdlePerKey(max);

    // Don't limit the total number of worker processes, as otherwise the pool might be full of
    // workers for one WorkerKey and can't accommodate a worker for another WorkerKey.
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
    if (key.getProxied()) {
      return multiplexPools.getOrDefault(key.getMnemonic(), multiplexPools.get(""));
    } else {
      return workerPools.getOrDefault(key.getMnemonic(), workerPools.get(""));
    }
  }

  /**
   * Gets a worker. May block indefinitely if too many high-priority workers are in use and the
   * requested worker is not high priority.
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
    workerPools.values().forEach(GenericKeyedObjectPool::close);
    multiplexPools.values().forEach(GenericKeyedObjectPool::close);
  }

  /** Stops any ongoing work in the worker pools. This may entail killing the worker processes. */
  public void stopWork() {
    workerPools
        .values()
        .forEach(
            pool -> {
              if (pool.getNumActive() > 0) {
                pool.clear();
              }
            });
    multiplexPools
        .values()
        .forEach(
            pool -> {
              if (pool.getNumActive() > 0) {
                pool.clear();
              }
            });
  }
}
