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
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.ThreadSafe;
import org.apache.commons.pool2.impl.EvictionPolicy;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;

/** Implementation of WorkerPool. */
@ThreadSafe
public class WorkerPoolImpl implements WorkerPool {
  /** Unless otherwise specified, the max number of workers per WorkerKey. */
  private static final int DEFAULT_MAX_WORKERS = 4;
  /** Unless otherwise specified, the max number of multiplex workers per WorkerKey. */
  private static final int DEFAULT_MAX_MULTIPLEX_WORKERS = 8;

  private final WorkerPoolConfig workerPoolConfig;
  /** Map of singleplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> workerPools;
  /** Map of multiplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> multiplexPools;

  /** Set of worker ids which are going to be destroyed after they are returned to the pool */
  private ImmutableSet<Integer> doomedWorkers = ImmutableSet.of();

  public WorkerPoolImpl(WorkerPoolConfig workerPoolConfig) {
    this.workerPoolConfig = workerPoolConfig;

    ImmutableMap<String, Integer> config =
        createConfigFromOptions(workerPoolConfig.getWorkerMaxInstances(), DEFAULT_MAX_WORKERS);
    ImmutableMap<String, Integer> multiplexConfig =
        createConfigFromOptions(
            workerPoolConfig.getWorkerMaxMultiplexInstances(), DEFAULT_MAX_MULTIPLEX_WORKERS);

    workerPools = createWorkerPools(workerPoolConfig.getWorkerFactory(), config);
    multiplexPools = createWorkerPools(workerPoolConfig.getWorkerFactory(), multiplexConfig);
  }

  public WorkerPoolConfig getWorkerPoolConfig() {
    return workerPoolConfig;
  }

  /**
   * Creates a configuration for a worker pool from the options given. If the same mnemonic occurs
   * more than once in the options, the last value passed wins.
   */
  @Nonnull
  private static ImmutableMap<String, Integer> createConfigFromOptions(
      List<Entry<String, Integer>> options, int defaultMaxWorkers) {
    LinkedHashMap<String, Integer> newConfigBuilder = new LinkedHashMap<>();
    for (Map.Entry<String, Integer> entry : options) {
      if (entry.getValue() != null) {
        newConfigBuilder.put(entry.getKey(), entry.getValue());
      } else if (entry.getKey() != null) {
        newConfigBuilder.put(entry.getKey(), defaultMaxWorkers);
      }
    }
    if (!newConfigBuilder.containsKey("")) {
      // Empty string gives the number of workers for any type of worker not explicitly specified.
      // If no value is given, use the default.
      newConfigBuilder.put("", defaultMaxWorkers);
    }
    return ImmutableMap.copyOf(newConfigBuilder);
  }

  private static ImmutableMap<String, SimpleWorkerPool> createWorkerPools(
      WorkerFactory factory, Map<String, Integer> config) {
    ImmutableMap.Builder<String, SimpleWorkerPool> workerPoolsBuilder = ImmutableMap.builder();
    config.forEach(
        (key, value) -> workerPoolsBuilder.put(key, new SimpleWorkerPool(factory, value)));
    return workerPoolsBuilder.build();
  }

  private SimpleWorkerPool getPool(WorkerKey key) {
    if (key.isMultiplex()) {
      return multiplexPools.getOrDefault(key.getMnemonic(), multiplexPools.get(""));
    } else {
      return workerPools.getOrDefault(key.getMnemonic(), workerPools.get(""));
    }
  }

  @Override
  public int getMaxTotalPerKey(WorkerKey key) {
    return getPool(key).getMaxTotalPerKey(key);
  }

  public int getNumIdlePerKey(WorkerKey key) {
    return getPool(key).getNumIdle(key);
  }

  @Override
  public int getNumActive(WorkerKey key) {
    return getPool(key).getNumActive(key);
  }

  // TODO (b/242835648) filter throwed exceptions better
  @Override
  public void evictWithPolicy(EvictionPolicy<Worker> evictionPolicy) throws InterruptedException {
    for (SimpleWorkerPool pool : workerPools.values()) {
      evictWithPolicy(evictionPolicy, pool);
    }
    for (SimpleWorkerPool pool : multiplexPools.values()) {
      evictWithPolicy(evictionPolicy, pool);
    }
  }

  private void evictWithPolicy(EvictionPolicy<Worker> evictionPolicy, SimpleWorkerPool pool)
      throws InterruptedException {
    try {
        pool.setEvictionPolicy(evictionPolicy);
        pool.evict();
      } catch (Throwable t) {
        Throwables.propagateIfPossible(t, InterruptedException.class);
      throw new VerifyException("unexpected", t);
    }
  }

  /**
   * Gets a worker from worker pool. Could wait if no idle workers are available.
   *
   * @param key worker key
   * @return a worker
   */
  @Override
  public Worker borrowObject(WorkerKey key) throws IOException, InterruptedException {
    Worker result;
    try {
      result = getPool(key).borrowObject(key);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, IOException.class, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }
    return result;
  }

  @Override
  public void returnObject(WorkerKey key, Worker obj) {
    if (doomedWorkers.contains(obj.getWorkerId())) {
      obj.setDoomed(true);
    }
    getPool(key).returnObject(key, obj);
  }

  @Override
  public void invalidateObject(WorkerKey key, Worker obj) throws InterruptedException {
    if (doomedWorkers.contains(obj.getWorkerId())) {
      obj.setDoomed(true);
    }
    try {
      getPool(key).invalidateObject(key, obj);
    } catch (Throwable t) {
      Throwables.propagateIfPossible(t, InterruptedException.class);
      throw new RuntimeException("unexpected", t);
    }
  }

  @Override
  public synchronized void setDoomedWorkers(ImmutableSet<Integer> workerIds) {
    this.doomedWorkers = workerIds;
  }

  /** Clear set of doomed workers. Also reset all shrunk subtrahend of all worker pools. */
  @Override
  public synchronized void clearDoomedWorkers() {
    this.doomedWorkers = ImmutableSet.of();
    for (SimpleWorkerPool pool : workerPools.values()) {
      pool.clearShrunkBy();
    }
    for (SimpleWorkerPool pool : multiplexPools.values()) {
      pool.clearShrunkBy();
    }
  }

  ImmutableSet<Integer> getDoomedWorkers() {
    return doomedWorkers;
  }

  @Override
  public void setEventBus(EventBus eventBus) {
    for (SimpleWorkerPool pool : workerPools.values()) {
      pool.setEventBus(eventBus);
    }
    for (SimpleWorkerPool pool : multiplexPools.values()) {
      pool.setEventBus(eventBus);
    }
  }

  /**
   * Closes all the worker pools, destroying the workers in the process. This waits for any
   * currently-ongoing work to finish.
   */
  @Override
  public void close() {
    workerPools.values().forEach(GenericKeyedObjectPool::close);
    multiplexPools.values().forEach(GenericKeyedObjectPool::close);
  }

  /**
   * Describes the configuration of worker pool, e.g. number of maximal instances and priority of
   * the workers.
   */
  public static class WorkerPoolConfig {
    private final WorkerFactory workerFactory;
    private final List<Entry<String, Integer>> workerMaxInstances;
    private final List<Entry<String, Integer>> workerMaxMultiplexInstances;

    public WorkerPoolConfig(
        WorkerFactory workerFactory,
        List<Entry<String, Integer>> workerMaxInstances,
        List<Entry<String, Integer>> workerMaxMultiplexInstances) {
      this.workerFactory = workerFactory;
      this.workerMaxInstances = workerMaxInstances;
      this.workerMaxMultiplexInstances = workerMaxMultiplexInstances;
    }

    public WorkerFactory getWorkerFactory() {
      return workerFactory;
    }

    public List<Entry<String, Integer>> getWorkerMaxInstances() {
      return workerMaxInstances;
    }

    public List<Entry<String, Integer>> getWorkerMaxMultiplexInstances() {
      return workerMaxMultiplexInstances;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof WorkerPoolConfig)) {
        return false;
      }
      WorkerPoolConfig that = (WorkerPoolConfig) o;
      return workerFactory.equals(that.workerFactory)
          && workerMaxInstances.equals(that.workerMaxInstances)
          && workerMaxMultiplexInstances.equals(that.workerMaxMultiplexInstances);
    }

    @Override
    public int hashCode() {
      return Objects.hash(workerFactory, workerMaxInstances, workerMaxMultiplexInstances);
    }
  }
}
