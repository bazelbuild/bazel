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
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerMetric.WorkerStat;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** Collects and populates system metrics about persistent workers. */
public class WorkerMetricsCollector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The metrics collector (a static singleton instance). Inactive by default. */
  private static final WorkerMetricsCollector instance = new WorkerMetricsCollector();

  private Clock clock;

  /**
   * Mapping of worker ids to their metrics. Contains worker ids, which memory usage could be
   * measured.
   */
  private final Map<Integer, WorkerMetric.WorkerProperties> workerIdToWorkerProperties =
      new ConcurrentHashMap<>();

  private final Map<Integer, Instant> workerLastCallTime = new ConcurrentHashMap<>();

  private MetricsWithTime lastMetrics = new MetricsWithTime(ImmutableList.of(), Instant.EPOCH);

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
    if (os != OS.LINUX && os != OS.DARWIN) {
      return new MemoryCollectionResult(
          ImmutableMap.of(), Instant.ofEpochMilli(clock.currentTimeMillis()));
    }

    ImmutableMap<Long, Long> pidsToWorkerPid = getSubprocesses(processIds);
    ImmutableMap<Long, Integer> psMemory = collectDataFromPs(pidsToWorkerPid.keySet());

    Map<Long, Integer> sumMemory = new HashMap<>();
    psMemory.forEach(
        (pid, memory) -> {
          long parent = pidsToWorkerPid.get(pid);
          int parentMemory = 0;
          if (sumMemory.containsKey(parent)) {
            parentMemory = sumMemory.get(parent);
          }
          sumMemory.put(parent, parentMemory + memory);
        });

    return new MemoryCollectionResult(
        ImmutableMap.copyOf(sumMemory), Instant.ofEpochMilli(clock.currentTimeMillis()));
  }

  /**
   * For each parent process collects pids of all descendants. Stores them into the map, where key
   * is the descendant pid and the value is parent pid.
   */
  ImmutableMap<Long, Long> getSubprocesses(ImmutableSet<Long> parents) {
    ImmutableMap.Builder<Long, Long> subprocessesToProcess = ImmutableMap.builder();
    for (Long pid : parents) {
      Optional<ProcessHandle> processHandle = ProcessHandle.of(pid);

      if (processHandle.isPresent()) {
        processHandle
            .get()
            .descendants()
            .map(p -> p.pid())
            .forEach(p -> subprocessesToProcess.put(p, pid));
        subprocessesToProcess.put(pid, pid);
      }
    }

    return subprocessesToProcess.buildKeepingLast();
  }

  // Collects memory usage for every process
  private ImmutableMap<Long, Integer> collectDataFromPs(Collection<Long> pids) {
    BufferedReader psOutput;
    try {
      psOutput =
          new BufferedReader(new InputStreamReader(buildPsProcess(pids).getInputStream(), UTF_8));
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error while executing command for pids: %s", pids);
      return ImmutableMap.of();
    }

    ImmutableMap.Builder<Long, Integer> processMemory = ImmutableMap.builder();

    try {
      // The output of the above ps command looks similar to this:
      // PID RSS
      // 211706 222972
      // 2612333 6180
      // We skip over the first line (the header) and then parse the PID and the resident memory
      // size in kilobytes.
      String output = null;
      boolean isFirst = true;
      while ((output = psOutput.readLine()) != null) {
        if (isFirst) {
          isFirst = false;
          continue;
        }
        List<String> line = Splitter.on(" ").trimResults().omitEmptyStrings().splitToList(output);
        if (line.size() != 2) {
          logger.atWarning().log("Unexpected length of split line %s %d", output, line.size());
          continue;
        }

        long pid = Long.parseLong(line.get(0));
        int memoryInKb = Integer.parseInt(line.get(1));

        processMemory.put(pid, memoryInKb);
      }
    } catch (IllegalArgumentException | IOException e) {
      logger.atWarning().withCause(e).log("Error while parsing psOutput: %s", psOutput);
    }

    return processMemory.buildOrThrow();
  }

  @VisibleForTesting
  public Process buildPsProcess(Collection<Long> processIds) throws IOException {
    ImmutableList<Long> filteredProcessIds =
        processIds.stream().filter(p -> p > 0).collect(toImmutableList());
    String pids = Joiner.on(",").join(filteredProcessIds);
    return new ProcessBuilder("ps", "-o", "pid,rss", "-p", pids).start();
  }

  /**
   * Collect worker metrics. If last collected metrics weren't more than interval time ago, then
   * returns previously collected metrics;
   */
  public ImmutableList<WorkerMetric> collectMetrics(Duration interval) {
    Instant now = Instant.ofEpochMilli(clock.currentTimeMillis());
    if (Duration.between(lastMetrics.time, now).compareTo(interval) < 0) {
      return lastMetrics.metrics;
    }

    return collectMetrics();
  }

  // TODO(wilwell): add exception if we couldn't collect the metrics.
  public ImmutableList<WorkerMetric> collectMetrics() {
    MemoryCollectionResult memoryCollectionResult =
        collectMemoryUsageByPid(
            OS.getCurrent(),
            workerIdToWorkerProperties.values().stream()
                .map(WorkerMetric.WorkerProperties::getProcessId)
                .collect(toImmutableSet()));

    ImmutableMap<Long, Integer> pidToMemoryInKb = memoryCollectionResult.pidToMemoryInKb;
    Instant collectionTime = memoryCollectionResult.collectionTime;

    ImmutableList.Builder<WorkerMetric> workerMetrics = new ImmutableList.Builder<>();
    List<Integer> nonMeasurableWorkerIds = new ArrayList<>();
    for (WorkerMetric.WorkerProperties workerProperties : workerIdToWorkerProperties.values()) {
      Long pid = workerProperties.getProcessId();
      Integer workerId = workerProperties.getWorkerId();

      WorkerStat workerStats =
          WorkerStat.create(
              pidToMemoryInKb.getOrDefault(pid, 0),
              workerLastCallTime.get(workerId),
              collectionTime);

      workerMetrics.add(
          WorkerMetric.create(
              workerProperties, workerStats, /* isMeasurable= */ pidToMemoryInKb.containsKey(pid)));

      if (!pidToMemoryInKb.containsKey(pid)) {
        nonMeasurableWorkerIds.add(workerId);
      }
    }

    workerIdToWorkerProperties.keySet().removeAll(nonMeasurableWorkerIds);

    return updateLastCollectMetrics(workerMetrics.build(), collectionTime).metrics;
  }

  public void clear() {
    workerIdToWorkerProperties.clear();
  }

  @VisibleForTesting
  public Map<Integer, WorkerMetric.WorkerProperties> getWorkerIdToWorkerProperties() {
    return workerIdToWorkerProperties;
  }

  @VisibleForTesting
  public Map<Integer, Instant> getWorkerLastCallTime() {
    return workerLastCallTime;
  }

  /**
   * Initializes workerIdToWorkerProperties for workers. If worker metrics already exists for this
   * worker, only updates workerLastCallTime.
   */
  public void registerWorker(WorkerMetric.WorkerProperties properties) {
    int workerId = properties.getWorkerId();

    workerIdToWorkerProperties.putIfAbsent(workerId, properties);
    workerLastCallTime.put(workerId, Instant.ofEpochMilli(clock.currentTimeMillis()));
  }

  private synchronized MetricsWithTime updateLastCollectMetrics(
      ImmutableList<WorkerMetric> metrics, Instant time) {
    lastMetrics = new MetricsWithTime(metrics, time);
    return lastMetrics;
  }

  private static class MetricsWithTime {
    public final ImmutableList<WorkerMetric> metrics;
    public final Instant time;

    public MetricsWithTime(ImmutableList<WorkerMetric> metrics, Instant time) {
      this.metrics = metrics;
      this.time = time;
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
