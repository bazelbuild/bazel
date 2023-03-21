// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.VerifyException;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.PsInfoCollector;
import com.google.devtools.build.lib.worker.WorkerMetric.WorkerStat;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Collects and populates system metrics about persistent workers. */
public class WorkerMetricsCollector {

  /** The metrics collector (a static singleton instance). Inactive by default. */
  private static final WorkerMetricsCollector instance = new WorkerMetricsCollector();

  private Clock clock;

  /**
   * Mapping of worker process ids to their properties. One process could be mapped to multiple
   * workers because of multiplex workers.
   */
  private final Map<Long, WorkerMetric.WorkerProperties> processIdToWorkerProperties =
      new ConcurrentHashMap<>();

  private final Map<Long, Instant> workerLastCallTime = new ConcurrentHashMap<>();

  private MetricsCache metricsCache;

  private WorkerMetricsCollector() {}

  public static WorkerMetricsCollector instance() {
    return instance;
  }

  public void setClock(Clock clock) {
    this.clock = clock;
  }

  /**
   * Collects memory usage of all ancestors of processes by pid. If a pid does not allow collecting
   * memory usage, it is silently ignored.
   */
  MemoryCollectionResult collectMemoryUsageByPid(OS os, ImmutableSet<Long> processIds) {
    // TODO(b/181317827): Support Windows.
    if (processIds.isEmpty() || (os != OS.LINUX && os != OS.DARWIN)) {
      return new MemoryCollectionResult(
          ImmutableMap.of(), Instant.ofEpochMilli(clock.currentTimeMillis()));
    }

    ImmutableMap<Long, PsInfoCollector.PsInfo> psInfos;
    try {
      psInfos = PsInfoCollector.collectDataFromPs();
    } catch (RuntimeException e) {
      throw new VerifyException(
          String.format("Could not collect data for pids: %s", processIds), e);
    }

    ImmutableMap<Long, Integer> pidToMemoryInKb = summarizeDescendantsMemory(psInfos, processIds);
    return new MemoryCollectionResult(
        pidToMemoryInKb, Instant.ofEpochMilli(clock.currentTimeMillis()));
  }

  /** Calculates summary memory usage of all descendata of processes */
  ImmutableMap<Long, Integer> summarizeDescendantsMemory(
      ImmutableMap<Long, PsInfoCollector.PsInfo> pidToPsInfo, ImmutableSet<Long> processIds) {

    HashMultimap<Long, PsInfoCollector.PsInfo> parentPidToPsInfo = HashMultimap.create();
    for (PsInfoCollector.PsInfo psInfo : pidToPsInfo.values()) {
      parentPidToPsInfo.put(psInfo.getParentPid(), psInfo);
    }

    ImmutableMap.Builder<Long, Integer> pidToTotalMemoryInKb = ImmutableMap.builder();
    for (Long pid : processIds) {
      if (!pidToPsInfo.containsKey(pid)) {
        continue;
      }
      PsInfoCollector.PsInfo psInfo = pidToPsInfo.get(pid);
      pidToTotalMemoryInKb.put(pid, collectMemoryUsageOfDescendants(psInfo, parentPidToPsInfo));
    }

    return pidToTotalMemoryInKb.buildOrThrow();
  }

  /** Recurseviely collects total memory usage of all descendants of process. */
  private int collectMemoryUsageOfDescendants(
      PsInfoCollector.PsInfo psInfo, HashMultimap<Long, PsInfoCollector.PsInfo> parentPidToPsInfo) {
    int currentMemoryInKb = psInfo.getMemoryInKb();
    for (PsInfoCollector.PsInfo childrenPsInfo : parentPidToPsInfo.get(psInfo.getPid())) {
      currentMemoryInKb += collectMemoryUsageOfDescendants(childrenPsInfo, parentPidToPsInfo);
    }

    return currentMemoryInKb;
  }

  /**
   * Collect worker metrics. If last collected metrics weren't more than interval time ago, then
   * returns previously collected metrics;
   */
  public ImmutableList<WorkerMetric> collectMetrics(Duration interval) {
    Instant now = Instant.ofEpochMilli(clock.currentTimeMillis());
    if (metricsCache != null
        && Duration.between(metricsCache.cachedTime, now).compareTo(interval) < 0) {
      return metricsCache.metrics;
    }

    return collectMetrics();
  }

  // TODO(wilwell): add exception if we couldn't collect the metrics.
  public ImmutableList<WorkerMetric> collectMetrics() {
    MemoryCollectionResult memoryCollectionResult =
        collectMemoryUsageByPid(
            OS.getCurrent(), ImmutableSet.copyOf(processIdToWorkerProperties.keySet()));

    ImmutableMap<Long, Integer> pidToMemoryInKb = memoryCollectionResult.pidToMemoryInKb;
    Instant collectionTime = memoryCollectionResult.collectionTime;

    ImmutableList.Builder<WorkerMetric> workerMetrics = new ImmutableList.Builder<>();
    List<Long> nonMeasurableProcessIds = new ArrayList<>();
    for (WorkerMetric.WorkerProperties workerProperties : processIdToWorkerProperties.values()) {
      Long pid = workerProperties.getProcessId();

      WorkerStat workerStats =
          WorkerStat.create(
              pidToMemoryInKb.getOrDefault(pid, 0), workerLastCallTime.get(pid), collectionTime);

      workerMetrics.add(
          WorkerMetric.create(
              workerProperties, workerStats, /* isMeasurable= */ pidToMemoryInKb.containsKey(pid)));

      if (!pidToMemoryInKb.containsKey(pid)) {
        nonMeasurableProcessIds.add(pid);
      }
    }

    processIdToWorkerProperties.keySet().removeAll(nonMeasurableProcessIds);

    return updateMetricsCache(workerMetrics.build(), collectionTime).metrics;
  }

  public void clear() {
    processIdToWorkerProperties.clear();
    workerLastCallTime.clear();
    metricsCache = null;
  }

  @VisibleForTesting
  Map<Long, WorkerMetric.WorkerProperties> getProcessIdToWorkerProperties() {
    return processIdToWorkerProperties;
  }

  @VisibleForTesting
  Map<Long, Instant> getWorkerLastCallTime() {
    return workerLastCallTime;
  }

  /**
   * Initializes workerIdToWorkerProperties for workers. If worker metrics already exists for this
   * worker, only updates workerLastCallTime.
   */
  public synchronized void registerWorker(
      int workerId, long processId, String mnemonic, boolean isMultiplex, boolean isSandboxed) {
    WorkerMetric.WorkerProperties existingWorkerProperties =
        processIdToWorkerProperties.get(processId);

    workerLastCallTime.put(processId, Instant.ofEpochMilli(clock.currentTimeMillis()));

    if (existingWorkerProperties == null) {
      processIdToWorkerProperties.put(
          processId,
          WorkerMetric.WorkerProperties.create(
              ImmutableList.of(workerId), processId, mnemonic, isMultiplex, isSandboxed));
      return;
    }

    if (existingWorkerProperties.getWorkerIds().contains(workerId)) {
      return;
    }

    ImmutableList<Integer> updatedWorkerIds =
        ImmutableList.<Integer>builder()
            .addAll(existingWorkerProperties.getWorkerIds())
            .add(workerId)
            .build();

    WorkerMetric.WorkerProperties updatedWorkerProperties =
        WorkerMetric.WorkerProperties.create(
            updatedWorkerIds, processId, mnemonic, isMultiplex, isSandboxed);
    processIdToWorkerProperties.put(processId, updatedWorkerProperties);
  }

  private synchronized MetricsCache updateMetricsCache(
      ImmutableList<WorkerMetric> metrics, Instant time) {
    metricsCache = new MetricsCache(metrics, time);
    return metricsCache;
  }

  private static class MetricsCache {
    public final ImmutableList<WorkerMetric> metrics;
    public final Instant cachedTime;

    public MetricsCache(ImmutableList<WorkerMetric> metrics, Instant cachedTime) {
      this.metrics = metrics;
      this.cachedTime = cachedTime;
    }
  }

  static class MemoryCollectionResult {
    public final ImmutableMap<Long, Integer> pidToMemoryInKb;
    public final Instant collectionTime;

    public MemoryCollectionResult(
        ImmutableMap<Long, Integer> pidToMemoryInKb, Instant collectionTime) {
      this.pidToMemoryInKb = pidToMemoryInKb;
      this.collectionTime = collectionTime;
    }
  }

  // TODO(b/238416583) Add deregister function
}
