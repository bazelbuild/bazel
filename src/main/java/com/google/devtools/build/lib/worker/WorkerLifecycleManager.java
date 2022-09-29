// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import java.time.Duration;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.EvictionConfig;
import org.apache.commons.pool2.impl.EvictionPolicy;

/**
 * This class kills idle persistent workers at intervals, if the total worker resource usage is
 * above a specified limit. Must be used as singleton.
 */
final class WorkerLifecycleManager extends Thread {

  private static final Duration SLEEP_INTERVAL = Duration.ofSeconds(5);
  // Collects metric not older than METRICS_MINIMAL_INTERVAL, to reduce calls of MetricsCollector.
  private static final Duration METRICS_MINIMAL_INTERVAL = Duration.ofSeconds(1);
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private boolean isWorking = false;
  private final WorkerPool workerPool;
  private final WorkerOptions options;

  public WorkerLifecycleManager(WorkerPool workerPool, WorkerOptions options) {
    this.workerPool = workerPool;
    this.options = options;
  }

  @Override
  public void run() {
    if (options.totalWorkerMemoryLimitMb == 0) {
      return;
    }

    isWorking = true;

    // This loop works until method stopProcessing() called by WorkerModule.
    while (isWorking) {
      try {
        Thread.sleep(SLEEP_INTERVAL.toMillis());
      } catch (InterruptedException e) {
        break;
      }

      ImmutableList<WorkerMetric> workerMetrics =
          WorkerMetricsCollector.instance().collectMetrics(METRICS_MINIMAL_INTERVAL);

      try {
        evictWorkers(workerMetrics);
      } catch (InterruptedException e) {
        break;
      }
    }

    isWorking = false;
  }

  void stopProcessing() {
    isWorking = false;
  }

  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  void evictWorkers(ImmutableList<WorkerMetric> workerMetrics) throws InterruptedException {

    if (options.totalWorkerMemoryLimitMb == 0) {
      return;
    }

    int workerMemeoryUsage =
        workerMetrics.stream()
            .filter(metric -> metric.getWorkerStat() != null)
            .mapToInt(metric -> metric.getWorkerStat().getUsedMemoryInKB() / 1000)
            .sum();

    if (workerMemeoryUsage <= options.totalWorkerMemoryLimitMb) {
      return;
    }

    ImmutableSet<Integer> candidates =
        collectEvictionCandidates(
            workerMetrics, options.totalWorkerMemoryLimitMb, workerMemeoryUsage);

    if (candidates.isEmpty()) {
      return;
    }

    logger.atInfo().log("Going to evict %d workers with ids: %s", candidates.size(), candidates);

    evictCandidates(workerPool, candidates);
  }

  private void evictCandidates(WorkerPool pool, ImmutableSet<Integer> candidates)
      throws InterruptedException {
    pool.evictWithPolicy(new CandidateEvictionPolicy(candidates));
  }

  /** Collects worker candidates to evict. Choses workers with the largest memory consumption. */
  @SuppressWarnings("JdkCollectors")
  ImmutableSet<Integer> collectEvictionCandidates(
      ImmutableList<WorkerMetric> workerMetrics, int memoryLimitMb, int workerMemeoryUsageMb)
      throws InterruptedException {
    Set<Integer> idleWorkers = getIdleWorkers();

    List<WorkerMetric> idleWorkerMetrics =
        workerMetrics.stream()
            .filter(
                metric ->
                    metric.getWorkerStat() != null
                        && idleWorkers.contains(metric.getWorkerProperties().getWorkerId()))
            .collect(Collectors.toList());

    if (idleWorkerMetrics.size() != idleWorkers.size()) {
      logger.atInfo().log(
          "Difference between idle workers and idle worker metrics is %d",
          idleWorkers.size() - idleWorkerMetrics.size());
    }

    idleWorkerMetrics.sort(new MemoryComparator());

    ImmutableSet.Builder<Integer> candidates = ImmutableSet.builder();
    int freeMemoryMb = 0;
    for (WorkerMetric metric : idleWorkerMetrics) {
      candidates.add(metric.getWorkerProperties().getWorkerId());
      freeMemoryMb += metric.getWorkerStat().getUsedMemoryInKB() / 1000;

      if (workerMemeoryUsageMb - freeMemoryMb <= memoryLimitMb) {
        break;
      }
    }

    return candidates.build();
  }

  /**
   * Calls workerPool.evict() to collect information, but doesn't kill any workers during this
   * process.
   */
  private Set<Integer> getIdleWorkers() throws InterruptedException {
    InfoEvictionPolicy infoEvictionPolicy = new InfoEvictionPolicy();
    workerPool.evictWithPolicy(infoEvictionPolicy);
    return infoEvictionPolicy.getWorkerIds();
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

  /** Eviction policy for WorkerPool. Evict all idle workers, which were passed in constructor. */
  private static class CandidateEvictionPolicy implements EvictionPolicy<Worker> {
    private final ImmutableSet<Integer> workerIds;

    public CandidateEvictionPolicy(ImmutableSet<Integer> workerIds) {
      this.workerIds = workerIds;
    }

    @Override
    public boolean evict(EvictionConfig config, PooledObject<Worker> underTest, int idleCount) {
      int workerId = underTest.getObject().getWorkerId();
      if (workerIds.contains(workerId)) {
        logger.atInfo().log(
            "Evicting worker %d with mnemonic %s",
            workerId, underTest.getObject().getWorkerKey().getMnemonic());
        return true;
      }

      return false;
    }
  }

  /** Compare workers memory in descending order. */
  private static class MemoryComparator implements Comparator<WorkerMetric> {
    @Override
    public int compare(WorkerMetric m1, WorkerMetric m2) {
      return m2.getWorkerStat().getUsedMemoryInKB() - m1.getWorkerStat().getUsedMemoryInKB();
    }
  }
}
