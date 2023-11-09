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
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Collects and populates system metrics about persistent workers. */
public class WorkerProcessMetricsCollector {

  /** The metrics collector (a static singleton instance). Inactive by default. */
  private static final WorkerProcessMetricsCollector instance = new WorkerProcessMetricsCollector();

  private Clock clock;

  /**
   * Mapping of worker process ids to their properties. One process could be mapped to multiple
   * workers because of multiplex workers.
   */
  private final Map<Long, WorkerProcessMetrics> processIdToWorkerProcessMetrics =
      new ConcurrentHashMap<>();

  private WorkerProcessMetricsCollector() {}

  public static WorkerProcessMetricsCollector instance() {
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

  public ImmutableList<WorkerProcessMetrics> collectMetrics() {
    PsInfoCollector.ResourceSnapshot resourceSnapshot =
        collectMemoryUsageByPid(
            OS.getCurrent(), ImmutableSet.copyOf(processIdToWorkerProcessMetrics.keySet()));

    ImmutableMap<Long, Integer> pidToMemoryInKb = resourceSnapshot.getPidToMemoryInKb();
    Instant collectionTime = resourceSnapshot.getCollectionTime();

    List<Long> nonMeasurableProcessIds = new ArrayList<>();
    ImmutableList.Builder<WorkerProcessMetrics> workerMetrics = new ImmutableList.Builder<>();
    for (Map.Entry<Long, WorkerProcessMetrics> entry : processIdToWorkerProcessMetrics.entrySet()) {
      WorkerProcessMetrics workerMetric = entry.getValue();
      Long pid = workerMetric.getProcessId();
      workerMetric.addCollectedMetrics(
          /* memoryInKb= */ pidToMemoryInKb.getOrDefault(pid, 0),
          /* isMeasurable= */ pidToMemoryInKb.containsKey(pid),
          /* collectionTime= */ collectionTime);

      workerMetrics.add(workerMetric);

      if (!pidToMemoryInKb.containsKey(pid)) {
        nonMeasurableProcessIds.add(pid);
      }
    }

    processIdToWorkerProcessMetrics.keySet().removeAll(nonMeasurableProcessIds);

    return workerMetrics.build();
  }

  public ImmutableList<WorkerMetrics> createWorkerMetricsProto() {
    return collectMetrics().stream().map(WorkerProcessMetrics::toProto).collect(toImmutableList());
  }

  public void clear() {
    processIdToWorkerProcessMetrics.clear();
  }

  @VisibleForTesting
  Map<Long, WorkerProcessMetrics> getProcessIdToWorkerProcessMetrics() {
    return processIdToWorkerProcessMetrics;
  }

  /**
   * Initializes workerIdToWorkerProperties for workers. If worker metrics already exists for this
   * worker, only updates the last call time and maybe adds the multiplex worker id.
   */
  public synchronized void registerWorker(
      int workerId,
      long processId,
      String mnemonic,
      boolean isMultiplex,
      boolean isSandboxed,
      int workerKeyHash) {
    WorkerProcessMetrics workerMetric =
        processIdToWorkerProcessMetrics.computeIfAbsent(
            processId,
            (pid) ->
                new WorkerProcessMetrics(
                    workerId, processId, mnemonic, isMultiplex, isSandboxed, workerKeyHash));

    workerMetric.setLastCallTime(Instant.ofEpochMilli(clock.currentTimeMillis()));
    workerMetric.maybeAddWorkerId(workerId);
  }

  // TODO(b/238416583) Add deregister function
}
