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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.EvictionConfig;
import org.apache.commons.pool2.impl.EvictionPolicy;

/**
 * This class kills idle persistent workers at intervals, if the total worker resource usage is
 * above a specified limit. Must be used as singleton.
 */
final class WorkerLifecycleManager extends Thread {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private boolean isWorking = false;
  private boolean emptyEvictionWasLogged = false;
  private final WorkerPool workerPool;
  private final WorkerOptions options;

  private Reporter reporter;
  private EventBus eventBus;

  public WorkerLifecycleManager(WorkerPool workerPool, WorkerOptions options) {
    this.workerPool = workerPool;
    this.options = options;
  }

  public void setReporter(Reporter reporter) {
    this.reporter = reporter;
  }

  public void setEventBus(EventBus eventBus) {
    this.eventBus = eventBus;
  }

  @Override
  public void run() {
    if (options.totalWorkerMemoryLimitMb == 0 && options.workerMemoryLimitMb == 0) {
      return;
    }

    String msg =
        String.format(
            "Worker Lifecycle Manager starts work with (total limit: %d MB, limit: %d MB,"
                + " shrinking: %s)",
            options.totalWorkerMemoryLimitMb,
            options.workerMemoryLimitMb,
            options.shrinkWorkerPool ? "enabled" : "disabled");
    logger.atInfo().log("%s", msg);
    if (reporter != null) {
      reporter.handle(Event.info(msg));
    }

    isWorking = true;

    // This loop works until method stopProcessing() called by WorkerModule.
    while (isWorking) {
      try {
        Thread.sleep(options.workerMetricsPollInterval.toMillis());
      } catch (InterruptedException e) {
        logger.atInfo().withCause(e).log("received interrupt in worker life cycle manager");
        break;
      }

      ImmutableList<WorkerMetric> workerMetrics =
          WorkerMetricsCollector.instance().collectMetrics();

      if (options.totalWorkerMemoryLimitMb > 0) {
        try {
          evictWorkers(workerMetrics);
        } catch (InterruptedException e) {
          logger.atInfo().withCause(e).log("received interrupt in worker life cycle manager");
          break;
        }
      }

      if (options.workerMemoryLimitMb > 0) {
        killLargeWorkers(workerMetrics, options.workerMemoryLimitMb);
      }
    }

    isWorking = false;
  }

  void stopProcessing() {
    isWorking = false;
  }

  /** Kills any worker that uses more than {@code limitMb} MB of memory. */
  void killLargeWorkers(ImmutableList<WorkerMetric> workerMetrics, int limitMb) {
    ImmutableList<WorkerMetric> large =
        workerMetrics.stream()
            .filter(m -> m.getWorkerStat().getUsedMemoryInKB() / 1000 > limitMb)
            .collect(toImmutableList());

    for (WorkerMetric l : large) {
      String msg;

      ImmutableList<Integer> workerIds = l.getWorkerProperties().getWorkerIds();
      Optional<ProcessHandle> ph = ProcessHandle.of(l.getWorkerProperties().getProcessId());
      if (ph.isPresent()) {
        msg =
            String.format(
                "Killing %s worker %s (pid %d) taking %dMB",
                l.getWorkerProperties().getMnemonic(),
                workerIds.size() == 1 ? workerIds.get(0) : workerIds,
                l.getWorkerProperties().getProcessId(),
                l.getWorkerStat().getUsedMemoryInKB() / 1000);
        ph.get().destroyForcibly();
        logger.atInfo().log("%s", msg);
        if (reporter != null) {
          reporter.handle(Event.info(msg));
        }
        if (eventBus != null) {
          eventBus.post(
              new WorkerEvictedEvent(
                  l.getWorkerProperties().getWorkerKeyHash(),
                  l.getWorkerProperties().getMnemonic()));
        }
      }
    }
  }

  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  void evictWorkers(ImmutableList<WorkerMetric> workerMetrics) throws InterruptedException {

    if (options.totalWorkerMemoryLimitMb == 0) {
      return;
    }

    int workerMemoryUsage =
        workerMetrics.stream()
            .mapToInt(metric -> metric.getWorkerStat().getUsedMemoryInKB() / 1000)
            .sum();

    // TODO: Remove after b/274608075 is fixed.
    if (!workerMetrics.isEmpty()) {
      logger.atInfo().atMostEvery(1, TimeUnit.MINUTES).log(
          "total worker memory %dMB while limit is %dMB - details: %s",
          workerMemoryUsage,
          options.totalWorkerMemoryLimitMb,
          workerMetrics.stream()
              .map(
                  metric ->
                      metric.getWorkerProperties().getWorkerIds()
                          + " "
                          + metric.getWorkerProperties().getMnemonic()
                          + " "
                          + metric.getWorkerStat().getUsedMemoryInKB()
                          + "kB")
              .collect(Collectors.joining(", ")));
    }

    if (workerMemoryUsage <= options.totalWorkerMemoryLimitMb) {
      return;
    }

    ImmutableSet<Integer> candidates =
        collectEvictionCandidates(
            workerMetrics, options.totalWorkerMemoryLimitMb, workerMemoryUsage);

    if (!candidates.isEmpty() || !emptyEvictionWasLogged) {
      String msg;
      if (candidates.isEmpty()) {
        msg =
            String.format(
                "Could not find any worker eviction candidates. Worker memory usage: %d MB, Memory"
                    + " limit: %d MB",
                workerMemoryUsage, options.totalWorkerMemoryLimitMb);
      } else {
        msg =
            String.format("Going to evict %d workers with ids: %s", candidates.size(), candidates);
      }

      logger.atInfo().log("%s", msg);
      if (reporter != null) {
        reporter.handle(Event.info(msg));
      }
    }

    ImmutableSet<Integer> evictedWorkers = evictCandidates(workerPool, candidates);

    if (!evictedWorkers.isEmpty() || !emptyEvictionWasLogged) {
      String msg =
          String.format(
              "Total evicted idle workers %d. With ids: %s", evictedWorkers.size(), evictedWorkers);
      logger.atInfo().log("%s", msg);
      if (reporter != null) {
        reporter.handle(Event.info(msg));
      }

      emptyEvictionWasLogged = candidates.isEmpty();
    }

    if (eventBus != null) {
      for (WorkerMetric metric : workerMetrics) {
        WorkerMetric.WorkerProperties properties = metric.getWorkerProperties();

        for (Integer workerId : properties.getWorkerIds()) {
          if (evictedWorkers.contains(workerId)) {
            eventBus.post(
                new WorkerEvictedEvent(properties.getWorkerKeyHash(), properties.getMnemonic()));
          }
        }
      }
    }

    if (options.shrinkWorkerPool) {
      List<WorkerMetric> notEvictedWorkerMetrics =
          workerMetrics.stream()
              .filter(
                  metric ->
                      !evictedWorkers.containsAll(metric.getWorkerProperties().getWorkerIds()))
              .collect(Collectors.toList());

      int notEvictedWorkerMemoryUsage =
          notEvictedWorkerMetrics.stream()
              .mapToInt(metric -> metric.getWorkerStat().getUsedMemoryInKB() / 1000)
              .sum();

      if (notEvictedWorkerMemoryUsage <= options.totalWorkerMemoryLimitMb) {
        return;
      }

      postponeInvalidation(notEvictedWorkerMetrics, notEvictedWorkerMemoryUsage);
    }
  }

  private void postponeInvalidation(
      List<WorkerMetric> workerMetrics, int notEvictedWorkerMemoryUsage) {
    ImmutableSet<Integer> potentialCandidates =
        getCandidates(workerMetrics, options.totalWorkerMemoryLimitMb, notEvictedWorkerMemoryUsage);

    if (!potentialCandidates.isEmpty()) {
      String msg = String.format("New doomed workers candidates %s", potentialCandidates);
      logger.atInfo().log("%s", msg);
      if (reporter != null) {
        reporter.handle(Event.info(msg));
      }
      workerPool.setDoomedWorkers(potentialCandidates);
    }
  }

  /**
   * Applies eviction police for candidates. Returns the worker ids of evicted workers. We don't
   * guarantee that every candidate is going to be evicted. Returns worker ids of evicted workers.
   */
  private static ImmutableSet<Integer> evictCandidates(
      WorkerPool pool, ImmutableSet<Integer> candidates) throws InterruptedException {
    CandidateEvictionPolicy policy = new CandidateEvictionPolicy(candidates);
    pool.evictWithPolicy(policy);
    return policy.getEvictedWorkers();
  }

  /** Collects worker candidates to evict. Chooses workers with the largest memory consumption. */
  @SuppressWarnings("JdkCollectors")
  ImmutableSet<Integer> collectEvictionCandidates(
      ImmutableList<WorkerMetric> workerMetrics, int memoryLimitMb, int workerMemoryUsageMb)
      throws InterruptedException {
    Set<Integer> idleWorkers = getIdleWorkers();

    List<WorkerMetric> idleWorkerMetrics =
        workerMetrics.stream()
            .filter(metric -> idleWorkers.containsAll(metric.getWorkerProperties().getWorkerIds()))
            .collect(Collectors.toList());

    return getCandidates(idleWorkerMetrics, memoryLimitMb, workerMemoryUsageMb);
  }

  /**
   * Chooses the worker ids of workers with the most usage of memory. Selects workers until total
   * memory usage is less than memoryLimitMb.
   */
  private static ImmutableSet<Integer> getCandidates(
      List<WorkerMetric> workerMetrics, int memoryLimitMb, int usedMemoryMb) {

    workerMetrics.sort(new MemoryComparator());
    ImmutableSet.Builder<Integer> candidates = ImmutableSet.builder();
    int freeMemoryMb = 0;
    for (WorkerMetric metric : workerMetrics) {
      candidates.addAll(metric.getWorkerProperties().getWorkerIds());
      freeMemoryMb += metric.getWorkerStat().getUsedMemoryInKB() / 1000;

      if (usedMemoryMb - freeMemoryMb <= memoryLimitMb) {
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
    private final ImmutableSet<Integer> workerCandidates;
    private final Set<Integer> evictedWorkers;

    public CandidateEvictionPolicy(ImmutableSet<Integer> workerCandidates) {
      this.workerCandidates = workerCandidates;
      this.evictedWorkers = new HashSet<>();
    }

    @Override
    public boolean evict(EvictionConfig config, PooledObject<Worker> underTest, int idleCount) {
      int workerId = underTest.getObject().getWorkerId();
      if (workerCandidates.contains(workerId)) {
        evictedWorkers.add(workerId);
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

  /** Compare worker metrics by memory consumption in descending order. */
  private static class MemoryComparator implements Comparator<WorkerMetric> {
    @Override
    public int compare(WorkerMetric m1, WorkerMetric m2) {
      return m2.getWorkerStat().getUsedMemoryInKB() - m1.getWorkerStat().getUsedMemoryInKB();
    }
  }
}
