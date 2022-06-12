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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.buildtool.CollectMetricsEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.OS;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Collects and populates system metrics about the workers. */
class WorkerMetricsCollector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ExtendedEventHandler reporter;
  /** Mapping of worker ids to their metrics. */
  private Map<Integer, WorkerMetric> workerIdToWorkerMetric = new ConcurrentHashMap<>();

  public WorkerMetricsCollector(ExtendedEventHandler reporter, EventBus eventBus) {
    this.reporter = reporter;
    eventBus.register(this);
  }

  // Collects process stats for each worker
  @VisibleForTesting
  public Map<Long, WorkerMetric.WorkerStat> collectStats(OS os, List<Long> processIds) {
    Map<Long, WorkerMetric.WorkerStat> pidResults = new HashMap<>();

    if (os != OS.LINUX && os != OS.DARWIN) {
      return pidResults;
    }

    BufferedReader psOutput;
    try {
      psOutput =
          new BufferedReader(
              new InputStreamReader(this.buildPsProcess(processIds).getInputStream(), UTF_8));
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error while executing command for pids: %s", processIds);
      return pidResults;
    }

    try {
      // The output of the above ps command looks similar to this:
      // PID RSS
      // 211706 222972
      // 2612333 6180
      // We skip over the first line (the header) and then parse the PID and the resident memory
      // size in kilobytes.
      Instant now = Instant.now();
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
        int memoryInKb = Integer.parseInt(line.get(1)) / 1000;

        pidResults.put(pid, new WorkerMetric.WorkerStat(memoryInKb, now));
      }
    } catch (IllegalArgumentException | IOException e) {
      logger.atWarning().withCause(e).log("Error while parsing psOutput: %s", psOutput);
    }
    return pidResults;
  }

  @VisibleForTesting
  public Process buildPsProcess(List<Long> processIds) throws IOException {
    ImmutableList<Long> filteredProcessIds =
        processIds.stream().filter(p -> p > 0).collect(toImmutableList());
    String pids = Joiner.on(",").join(filteredProcessIds);
    return new ProcessBuilder("ps", "-o", "pid,rss", "-p", pids).start();
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onCollectMetricsEvent(CollectMetricsEvent event) {
    Map<Long, WorkerMetric.WorkerStat> workerStats =
        collectStats(
            OS.getCurrent(),
            this.workerIdToWorkerMetric.values().stream()
                .map(WorkerMetric::getProcessId)
                .collect(toImmutableList()));

    for (WorkerMetric workerMetric : this.workerIdToWorkerMetric.values()) {
      WorkerMetric.WorkerStat workerStat = workerStats.get(workerMetric.getProcessId());
      if (workerStat == null) {
        workerMetric.setIsMeasurable(false);
        continue;
      }
      workerMetric.addWorkerStat(workerStat);
    }

    this.reporter.post(
        new WorkerMetricsEvent(new ArrayList<>(this.workerIdToWorkerMetric.values())));
    this.workerIdToWorkerMetric.clear();

    // remove dead workers from metrics list
    Map<Integer, WorkerMetric> measurableWorkerMetrics = new HashMap<>();
    for (WorkerMetric workerMetric : workerIdToWorkerMetric.values()) {
      if (workerMetric.getIsMeasurable()) {
        measurableWorkerMetrics.put(workerMetric.getWorkerId(), workerMetric);
      }
    }

    this.workerIdToWorkerMetric = measurableWorkerMetrics;
  }

  /**
   * Initializes metricsSet for workers. If worker metrics already exists for this worker, does
   * nothing
   */
  public void initializeMetricsSet(WorkerKey workerKey, Worker worker) {

    if (workerIdToWorkerMetric.containsKey(worker.getWorkerId())) {
      return;
    }
    long processId = worker.getProcessId();

    WorkerMetric workerMetric =
        new WorkerMetric(
            worker.getWorkerId(),
            processId,
            workerKey.getMnemonic(),
            workerKey.isMultiplex(),
            workerKey.isSandboxed());

    workerIdToWorkerMetric.put(worker.getWorkerId(), workerMetric);
  }
}
