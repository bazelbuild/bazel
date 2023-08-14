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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.metrics.PsInfoCollector;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerMetric.WorkerStat;
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
  PsInfoCollector.ResourceSnapshot collectMemoryUsageByPid(OS os, ImmutableSet<Long> processIds) {
    // TODO(b/181317827): Support Windows.
    if (processIds.isEmpty() || (os != OS.LINUX && os != OS.DARWIN)) {
      return PsInfoCollector.ResourceSnapshot.create(
          /* pidToMemoryInKb= */ ImmutableMap.of(), /* collectionTime= */ Instant.now());
    }

    return PsInfoCollector.instance().collectResourceUsage(processIds);
  }

  public ImmutableList<WorkerMetric> collectMetrics() {
    PsInfoCollector.ResourceSnapshot resourceSnapshot =
        collectMemoryUsageByPid(
            OS.getCurrent(), ImmutableSet.copyOf(processIdToWorkerProperties.keySet()));

    return buildWorkerMetrics(resourceSnapshot);
  }

  private ImmutableList<WorkerMetric> buildWorkerMetrics(
      PsInfoCollector.ResourceSnapshot resourceSnapshot) {
    ImmutableMap<Long, Integer> pidToMemoryInKb = resourceSnapshot.getPidToMemoryInKb();
    Instant collectionTime = resourceSnapshot.getCollectionTime();

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

    return workerMetrics.build();
  }

  public ImmutableList<WorkerMetrics> createWorkerMetricsProto() {
    return collectMetrics().stream().map(WorkerMetric::toProto).collect(toImmutableList());
  }

  public void clear() {
    processIdToWorkerProperties.clear();
    workerLastCallTime.clear();
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
      int workerId,
      long processId,
      String mnemonic,
      boolean isMultiplex,
      boolean isSandboxed,
      int workerKeyHash) {
    WorkerMetric.WorkerProperties existingWorkerProperties =
        processIdToWorkerProperties.get(processId);

    workerLastCallTime.put(processId, Instant.ofEpochMilli(clock.currentTimeMillis()));

    if (existingWorkerProperties == null) {
      processIdToWorkerProperties.put(
          processId,
          WorkerMetric.WorkerProperties.create(
              ImmutableList.of(workerId),
              processId,
              mnemonic,
              isMultiplex,
              isSandboxed,
              workerKeyHash));
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
            updatedWorkerIds, processId, mnemonic, isMultiplex, isSandboxed, workerKeyHash);
    processIdToWorkerProperties.put(processId, updatedWorkerProperties);
  }

  // TODO(b/238416583) Add deregister function
}
