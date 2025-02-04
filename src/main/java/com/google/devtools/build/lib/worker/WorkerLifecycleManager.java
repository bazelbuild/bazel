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
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.worker.WorkerProcessStatus.Status;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

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

  public WorkerLifecycleManager(WorkerPool workerPool, WorkerOptions options) {
    this.workerPool = workerPool;
    this.options = options;
  }

  public void setReporter(Reporter reporter) {
    this.reporter = reporter;
  }

  @Override
  public void run() {
    if (options.totalWorkerMemoryLimitMb == 0 && options.workerMemoryLimitMb == 0) {
      return;
    }

    String msg =
        String.format(
            "Worker Lifecycle Manager starts work with (total limit: %d MB, individual limit: %d"
                + " MB, shrinking: %s)",
            options.totalWorkerMemoryLimitMb,
            options.workerMemoryLimitMb,
            options.shrinkWorkerPool ? "enabled" : "disabled");
    logger.atInfo().log("%s", msg);
    if (options.workerVerbose && this.reporter != null) {
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

      ImmutableList<WorkerProcessMetrics> workerProcessMetrics =
          WorkerProcessMetricsCollector.instance().getLiveWorkerProcessMetrics();

      if (options.totalWorkerMemoryLimitMb > 0) {
        try {
          evictWorkers(workerProcessMetrics);
        } catch (InterruptedException e) {
          logger.atInfo().withCause(e).log("received interrupt in worker life cycle manager");
          break;
        }
      }

      if (options.workerMemoryLimitMb > 0) {
        killLargeWorkers(workerProcessMetrics, options.workerMemoryLimitMb);
      }
    }

    isWorking = false;
  }

  void stopProcessing() {
    isWorking = false;
  }

  /** Kills any worker that uses more than {@code limitMb} MB of memory. */
  void killLargeWorkers(ImmutableList<WorkerProcessMetrics> workerProcessMetrics, int limitMb) {
    ImmutableList<WorkerProcessMetrics> large =
        workerProcessMetrics.stream()
            .filter(m -> m.getUsedMemoryInKb() > limitMb * 1000)
            .collect(toImmutableList());

    for (WorkerProcessMetrics l : large) {
      String msg;

      ImmutableList<Integer> workerIds = l.getWorkerIds();
      Optional<ProcessHandle> ph = ProcessHandle.of(l.getProcessId());
      if (ph.isPresent()) {
        msg =
            String.format(
                "Killing %s worker %s (pid %d) because it is using more memory than the limit (%d"
                    + " KB > %d MB)",
                l.getMnemonic(),
                workerIds.size() == 1 ? workerIds.get(0) : workerIds,
                l.getProcessId(),
                l.getUsedMemoryInKb(),
                limitMb);
        logger.atInfo().log("%s", msg);
        // TODO(b/310640400): Converge APIs in killing workers, rather than killing via the process
        //  handle here (resulting in errors in execution), perhaps we want to wait till the worker
        //  is returned before killing it.
        ph.get().destroyForcibly();
        l.getStatus().maybeUpdateStatus(WorkerProcessStatus.Status.KILLED_DUE_TO_MEMORY_PRESSURE);
        // We want to always report this as this is a potential source of build failure.
        if (this.reporter != null) {
          reporter.handle(Event.warn(msg));
        }
      }
    }
  }

  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  void evictWorkers(ImmutableList<WorkerProcessMetrics> workerProcessMetrics)
      throws InterruptedException {

    if (options.totalWorkerMemoryLimitMb == 0) {
      return;
    }

    int workerMemoryUsageKb =
        workerProcessMetrics.stream().mapToInt(WorkerProcessMetrics::getUsedMemoryInKb).sum();

    // TODO: Remove after b/274608075 is fixed.
    if (!workerProcessMetrics.isEmpty()) {
      logger.atInfo().atMostEvery(1, TimeUnit.MINUTES).log(
          "total worker memory %d KB while limit is %d MB - details: %s",
          workerMemoryUsageKb,
          options.totalWorkerMemoryLimitMb,
          workerProcessMetrics.stream()
              .map(
                  metric ->
                      metric.getWorkerIds()
                          + " "
                          + metric.getMnemonic()
                          + " "
                          + metric.getUsedMemoryInKb()
                          + "KB")
              .collect(Collectors.joining(", ")));
    }

    if (workerMemoryUsageKb <= options.totalWorkerMemoryLimitMb * 1000) {
      return;
    }

    ImmutableSet<WorkerProcessMetrics> candidates =
        collectEvictionCandidates(
            workerProcessMetrics, options.totalWorkerMemoryLimitMb, workerMemoryUsageKb);

    if (!candidates.isEmpty() || !emptyEvictionWasLogged) {
      String msg;
      if (candidates.isEmpty()) {
        msg =
            String.format(
                "Could not find any worker eviction candidates. Worker memory usage: %d KB, Memory"
                    + " limit: %d MB",
                workerMemoryUsageKb, options.totalWorkerMemoryLimitMb);
      } else {
        ImmutableSet<Integer> workerIdsToEvict =
            candidates.stream().flatMap(m -> m.getWorkerIds().stream()).collect(toImmutableSet());
        msg =
            String.format(
                "Attempting eviction of %d workers with ids: %s",
                workerIdsToEvict.size(), workerIdsToEvict);
      }

      logger.atInfo().log("%s", msg);
      if (options.workerVerbose && this.reporter != null) {
        reporter.handle(Event.info(msg));
      }
    }

    ImmutableSet<Integer> evictedWorkers = evictCandidates(workerPool, candidates);

    if (!evictedWorkers.isEmpty() || !emptyEvictionWasLogged) {
      String msg =
          String.format(
              "Total evicted idle workers %d. With ids: %s", evictedWorkers.size(), evictedWorkers);
      logger.atInfo().log("%s", msg);
      if (options.workerVerbose && this.reporter != null) {
        reporter.handle(Event.info(msg));
      }

      emptyEvictionWasLogged = candidates.isEmpty();
    }

    // TODO(b/300067854): Shrinking of the worker pool happens on worker keys that are active at the
    //  time of polling, but doesn't shrink the pools of idle workers. We might be wrongly
    //  penalizing lower memory usage workers (but more active) by shrinking their pool sizes
    //  instead of higher memory usage workers (but less active) and are killed directly with
    //  {@code #evictCandidates()} (where shrinking doesn't happen).
    if (options.shrinkWorkerPool) {
      List<WorkerProcessMetrics> notEvictedWorkerProcessMetrics =
          workerProcessMetrics.stream()
              .filter(metric -> !evictedWorkers.containsAll(metric.getWorkerIds()))
              .collect(Collectors.toList());

      int notEvictedWorkerMemoryUsageKb =
          notEvictedWorkerProcessMetrics.stream()
              .mapToInt(WorkerProcessMetrics::getUsedMemoryInKb)
              .sum();

      if (notEvictedWorkerMemoryUsageKb <= options.totalWorkerMemoryLimitMb * 1000) {
        return;
      }

      postponeInvalidation(notEvictedWorkerProcessMetrics, notEvictedWorkerMemoryUsageKb);
    }
  }

  private void postponeInvalidation(
      List<WorkerProcessMetrics> workerProcessMetrics, int notEvictedWorkerMemoryUsageKb) {
    ImmutableSet<WorkerProcessMetrics> potentialCandidates =
        getCandidates(
            workerProcessMetrics, options.totalWorkerMemoryLimitMb, notEvictedWorkerMemoryUsageKb);

    if (!potentialCandidates.isEmpty()) {
      String msg =
          String.format(
              "Postponing eviction of worker ids: %s",
              potentialCandidates.stream()
                  .flatMap(m -> m.getWorkerIds().stream())
                  .collect(toImmutableList()));
      logger.atInfo().log("%s", msg);
      if (options.workerVerbose && this.reporter != null) {
        reporter.handle(Event.info(msg));
      }
      potentialCandidates.forEach(
          m -> m.getStatus().maybeUpdateStatus(Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE));
    }
  }

  /**
   * Applies eviction police for candidates. Returns the worker ids of evicted workers. We don't
   * guarantee that every candidate is going to be evicted. Returns worker ids of evicted workers.
   */
  private static ImmutableSet<Integer> evictCandidates(
      WorkerPool pool, ImmutableSet<WorkerProcessMetrics> candidates) throws InterruptedException {
    return pool.evictWorkers(
        candidates.stream().flatMap(w -> w.getWorkerIds().stream()).collect(toImmutableSet()));
  }

  /** Collects worker candidates to evict. Chooses workers with the largest memory consumption. */
  @SuppressWarnings("JdkCollectors")
  ImmutableSet<WorkerProcessMetrics> collectEvictionCandidates(
      ImmutableList<WorkerProcessMetrics> workerProcessMetrics,
      int memoryLimitMb,
      int workerMemoryUsageKb)
      throws InterruptedException {
    // TODO(b/300067854): Consider rethinking the strategy here. The current logic kills idle
    //  workers that have lower memory usage if the other higher memory usage workers are active
    //  (where killing them would have brought the memory usage under the limit). This means we
    //  could be killing memory compliant and performant workers unnecessarily; i.e. this strategy
    //  maximizes responsiveness towards being compliant to the memory limit with no guarantees of
    //  making it immediately compliant. Since we can't guarantee immediate compliance, tradeoff
    //  some of this responsiveness by just killing or marking workers as killed in descending
    //  memory usage and waiting for the active workers to be returned later (where they are then
    //  killed).
    ImmutableSet<Integer> idleWorkers = workerPool.getIdleWorkers();

    List<WorkerProcessMetrics> idleWorkerProcessMetrics =
        workerProcessMetrics.stream()
            .filter(metric -> metric.getWorkerIds().stream().anyMatch(idleWorkers::contains))
            .collect(Collectors.toList());

    return getCandidates(idleWorkerProcessMetrics, memoryLimitMb, workerMemoryUsageKb);
  }

  /**
   * Chooses the WorkerProcessMetrics of workers with the most usage of memory. Selects workers
   * until total memory usage is less than memoryLimitMb.
   */
  private static ImmutableSet<WorkerProcessMetrics> getCandidates(
      List<WorkerProcessMetrics> workerProcessMetrics, int memoryLimitMb, int usedMemoryKb) {

    workerProcessMetrics.sort(new MemoryComparator());
    ImmutableSet.Builder<WorkerProcessMetrics> candidates = ImmutableSet.builder();
    int freeMemoryKb = 0;
    for (WorkerProcessMetrics metric : workerProcessMetrics) {
      candidates.add(metric);
      freeMemoryKb += metric.getUsedMemoryInKb();

      if (usedMemoryKb - freeMemoryKb <= memoryLimitMb * 1000) {
        break;
      }
    }

    return candidates.build();
  }

  /** Compare worker metrics by memory consumption in descending order. */
  private static class MemoryComparator implements Comparator<WorkerProcessMetrics> {
    @Override
    public int compare(WorkerProcessMetrics m1, WorkerProcessMetrics m2) {
      return m2.getUsedMemoryInKb() - m1.getUsedMemoryInKb();
    }
  }
}
