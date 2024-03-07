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

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.ThreadSafe;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.EvictionConfig;
import org.apache.commons.pool2.impl.EvictionPolicy;
import org.apache.commons.pool2.impl.GenericKeyedObjectPool;

/**
 * TODO(b/323880131): Legacy implementation of WorkerPool to be removed eventually to cut dependency
 * on the apache.commons.pool2 library.
 */
@ThreadSafe
public class WorkerPoolImplLegacy implements WorkerPool {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Unless otherwise specified, the max number of workers per WorkerKey. */
  private static final int DEFAULT_MAX_WORKERS = 4;

  /** Unless otherwise specified, the max number of multiplex workers per WorkerKey. */
  private static final int DEFAULT_MAX_MULTIPLEX_WORKERS = 8;

  private final WorkerPoolConfig workerPoolConfig;

  /** Map of singleplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> workerPools;

  /** Map of multiplex worker pools, one per mnemonic. */
  private final ImmutableMap<String, SimpleWorkerPool> multiplexPools;

  public WorkerPoolImplLegacy(WorkerPoolConfig workerPoolConfig) {
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

  public void evictWithPolicy(EvictionPolicy<Worker> evictionPolicy) throws InterruptedException {
    for (SimpleWorkerPool pool : workerPools.values()) {
      evictWithPolicy(evictionPolicy, pool);
    }
    for (SimpleWorkerPool pool : multiplexPools.values()) {
      evictWithPolicy(evictionPolicy, pool);
    }
  }

  @Override
  public ImmutableSet<Integer> evictWorkers(ImmutableSet<Integer> workerIdsToEvict)
      throws InterruptedException {
    CandidateEvictionPolicy policy = new CandidateEvictionPolicy(workerIdsToEvict);
    evictWithPolicy(policy);
    return policy.getEvictedWorkers();
  }

  @Override
  public ImmutableSet<Integer> getIdleWorkers() throws InterruptedException {
    InfoEvictionPolicy infoEvictionPolicy = new InfoEvictionPolicy();
    evictWithPolicy(infoEvictionPolicy);
    return ImmutableSet.copyOf(infoEvictionPolicy.getWorkerIds());
  }

  /**
   * Eviction policy for WorkerPool. Only collects ids of idle workers, doesn't evict any of them.
   */
  private static class InfoEvictionPolicy implements EvictionPolicy<Worker> {
    private final Set<Integer> workerIds = new HashSet<>();

    public InfoEvictionPolicy() {}

    @Override
    public boolean evict(EvictionConfig config, PooledObject<Worker> underTest, int idleCount) {
      workerIds.add(underTest.getObject().getWorkerId());
      return false;
    }

    public Set<Integer> getWorkerIds() {
      return workerIds;
    }
  }

  private void evictWithPolicy(EvictionPolicy<Worker> evictionPolicy, SimpleWorkerPool pool)
      throws InterruptedException {
    try {
      pool.setEvictionPolicy(evictionPolicy);
      pool.evict();
    } catch (Throwable t) {
      throwIfInstanceOf(t, InterruptedException.class);
      throwIfUnchecked(t);
      throw new VerifyException("unexpected", t);
    }
  }

  /** Eviction policy for WorkerPool. Evict all idle workers, which were passed in constructor. */
  private static class CandidateEvictionPolicy implements EvictionPolicy<Worker> {
    private final ImmutableSet<Integer> workerIdsToEvict;
    private final Set<Integer> evictedWorkers;

    public CandidateEvictionPolicy(ImmutableSet<Integer> workerIdsToEvict) {
      this.workerIdsToEvict = workerIdsToEvict;
      this.evictedWorkers = new HashSet<>();
    }

    @Override
    public boolean evict(EvictionConfig config, PooledObject<Worker> underTest, int idleCount) {
      int workerId = underTest.getObject().getWorkerId();
      if (workerIdsToEvict.contains(workerId)) {
        evictedWorkers.add(workerId);
        // Eviction through an EvictionPolicy doesn't go through the #returnObject and
        // #invalidateObject code paths and directly calls #destroy, so we'll need to specify that
        // explicitly here.
        underTest
            .getObject()
            .getStatus()
            .maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);
        logger.atInfo().log(
            "Evicting worker %d with mnemonic %s",
            workerId, underTest.getObject().getWorkerKey().getMnemonic());
        return true;
      }
      return false;
    }

    public ImmutableSet<Integer> getEvictedWorkers() {
      return ImmutableSet.copyOf(evictedWorkers);
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
      throwIfInstanceOf(t, IOException.class);
      throwIfInstanceOf(t, InterruptedException.class);
      throwIfUnchecked(t);
      throw new RuntimeException("unexpected", t);
    }
    return result;
  }

  @Override
  public void returnObject(WorkerKey key, Worker obj) {
    getPool(key).returnObject(key, obj);
  }

  @Override
  public void invalidateObject(WorkerKey key, Worker obj) throws InterruptedException {
    try {
      getPool(key).invalidateObject(key, obj);
    } catch (Throwable t) {
      throwIfInstanceOf(t, InterruptedException.class);
      throwIfUnchecked(t);
      throw new RuntimeException("unexpected", t);
    }
  }

  /** Reset all shrunk subtrahend of all worker pools. */
  @Override
  public synchronized void reset() {
    for (Entry<String, SimpleWorkerPool> entry : workerPools.entrySet()) {
      logger.atInfo().log(
          "clearing shrunk by values for %s worker pool",
          entry.getKey().isEmpty() ? "shared" : entry.getKey());
      entry.getValue().clearShrunkBy();
    }
    for (Entry<String, SimpleWorkerPool> entry : multiplexPools.entrySet()) {
      logger.atInfo().log(
          "clearing shrunk by values for %s multiplex worker pool",
          entry.getKey().isEmpty() ? "shared" : entry.getKey());
      entry.getValue().clearShrunkBy();
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
}
