// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ResourceEstimator;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.profiler.NetworkMetricsCollector.SystemNetworkUsages;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.worker.WorkerMetric;
import com.google.devtools.build.lib.worker.WorkerMetricsCollector;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** Thread to collect local resource usage data and log into JSON profile. */
public class CollectLocalResourceUsage extends Thread {

  // TODO(twerth): Make these configurable.
  private static final Duration BUCKET_DURATION = Duration.ofSeconds(1);
  private static final Duration LOCAL_RESOURCES_COLLECT_SLEEP_INTERVAL = Duration.ofMillis(200);

  private final BugReporter bugReporter;
  private final boolean collectWorkerDataInProfiler;
  private final boolean collectLoadAverage;
  private final boolean collectSystemNetworkUsage;
  private final boolean collectResourceManagerEstimation;

  private volatile boolean stopLocalUsageCollection;
  private volatile boolean profilingStarted;

  @GuardedBy("this")
  @Nullable
  private Map<ProfilerTask, TimeSeries> timeSeries;

  private Stopwatch stopwatch;

  private final WorkerMetricsCollector workerMetricsCollector;

  private final ResourceEstimator resourceEstimator;
  private final boolean collectPressureStallIndicators;

  CollectLocalResourceUsage(
      BugReporter bugReporter,
      WorkerMetricsCollector workerMetricsCollector,
      ResourceEstimator resourceEstimator,
      boolean collectWorkerDataInProfiler,
      boolean collectLoadAverage,
      boolean collectSystemNetworkUsage,
      boolean collectResourceManagerEstimation,
      boolean collectPressureStallIndicators) {
    super("collect-local-resources");
    this.bugReporter = checkNotNull(bugReporter);
    this.collectWorkerDataInProfiler = collectWorkerDataInProfiler;
    this.workerMetricsCollector = workerMetricsCollector;
    this.collectLoadAverage = collectLoadAverage;
    this.collectSystemNetworkUsage = collectSystemNetworkUsage;
    this.collectResourceManagerEstimation = collectResourceManagerEstimation;
    this.resourceEstimator = resourceEstimator;
    this.collectPressureStallIndicators = collectPressureStallIndicators;
  }

  @Override
  public void run() {
    int numProcessors = Runtime.getRuntime().availableProcessors();
    stopwatch = Stopwatch.createStarted();
    synchronized (this) {
      timeSeries = new HashMap<>();
      Duration startTime = stopwatch.elapsed();
      List<ProfilerTask> enabledCounters = new ArrayList<>();
      enabledCounters.addAll(
          ImmutableList.of(
              ProfilerTask.LOCAL_CPU_USAGE,
              ProfilerTask.LOCAL_MEMORY_USAGE,
              ProfilerTask.SYSTEM_CPU_USAGE,
              ProfilerTask.SYSTEM_MEMORY_USAGE));

      if (collectWorkerDataInProfiler) {
        enabledCounters.add(ProfilerTask.WORKERS_MEMORY_USAGE);
      }
      if (collectLoadAverage) {
        enabledCounters.add(ProfilerTask.SYSTEM_LOAD_AVERAGE);
      }
      if (collectSystemNetworkUsage) {
        enabledCounters.add(ProfilerTask.SYSTEM_NETWORK_UP_USAGE);
        enabledCounters.add(ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE);
      }
      if (collectResourceManagerEstimation) {
        enabledCounters.add(ProfilerTask.MEMORY_USAGE_ESTIMATION);
        enabledCounters.add(ProfilerTask.CPU_USAGE_ESTIMATION);
      }
      if (collectPressureStallIndicators) {
        enabledCounters.add(ProfilerTask.PRESSURE_STALL_IO);
        enabledCounters.add(ProfilerTask.PRESSURE_STALL_MEMORY);
      }

      for (ProfilerTask counter : enabledCounters) {
        timeSeries.put(counter, new TimeSeries(startTime, BUCKET_DURATION));
      }
    }
    OperatingSystemMXBean osBean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    Duration previousElapsed = stopwatch.elapsed();
    long previousCpuTimeNanos = osBean.getProcessCpuTime();
    profilingStarted = true;
    while (!stopLocalUsageCollection) {
      try {
        Thread.sleep(LOCAL_RESOURCES_COLLECT_SLEEP_INTERVAL.toMillis());
      } catch (InterruptedException e) {
        return;
      }
      Duration nextElapsed = stopwatch.elapsed();
      long nextCpuTimeNanos = osBean.getProcessCpuTime();

      double systemCpuLoad = osBean.getSystemCpuLoad();
      if (Double.isNaN(systemCpuLoad)) {
        // Unlike advertised, on Mac the system CPU load is NaN sometimes.
        // There is no good way to handle this, so to avoid any downstream method crashing on this,
        // we reset the CPU value here.
        systemCpuLoad = 0;
      }
      double systemUsage = systemCpuLoad * numProcessors;

      long systemMemoryUsageMb = -1;
      if (OS.getCurrent() == OS.LINUX) {
        // On Linux we get a better estimate by using /proc/meminfo. See
        // https://www.linuxatemyram.com/ for more info on buffer caches.
        try {
          ProcMeminfoParser procMeminfoParser = new ProcMeminfoParser("/proc/meminfo");
          systemMemoryUsageMb =
              (procMeminfoParser.getTotalKb() - procMeminfoParser.getFreeRamKb()) / 1024;
        } catch (IOException e) {
          // Silently ignore and fallback.
        }
      }
      if (systemMemoryUsageMb <= 0) {
        // In case we aren't running on Linux or /proc/meminfo parsing went wrong, fall back to the
        // OS bean.
        systemMemoryUsageMb =
            (osBean.getTotalPhysicalMemorySize() - osBean.getFreePhysicalMemorySize())
                / (1024 * 1024);
      }

      long memoryUsage;
      try {
        memoryUsage =
            memoryBean.getHeapMemoryUsage().getUsed()
                + memoryBean.getNonHeapMemoryUsage().getUsed();
      } catch (IllegalArgumentException e) {
        // The JVM may report committed > max. See b/180619163.
        bugReporter.sendBugReport(e);
        memoryUsage = -1;
      }

      int workerMemoryUsageMb = 0;
      if (collectWorkerDataInProfiler) {
        try (SilentCloseable c = Profiler.instance().profile("Worker metrics collection")) {
          workerMemoryUsageMb =
              this.workerMetricsCollector.collectMetrics().stream()
                      .map(WorkerMetric::getWorkerStat)
                      .filter(Objects::nonNull)
                      .mapToInt(WorkerMetric.WorkerStat::getUsedMemoryInKB)
                      .sum()
                  / 1024;
        }
      }
      double loadAverage = 0;
      if (collectLoadAverage) {
        loadAverage = osBean.getSystemLoadAverage();
      }
      double pressureStallIo = 0;
      double pressureStallMemory = 0;
      if (collectPressureStallIndicators) {
        pressureStallIo = ResourceUsage.readPressureStallIndicator("io");
        pressureStallMemory = ResourceUsage.readPressureStallIndicator("memory");
      }

      double deltaNanos = nextElapsed.minus(previousElapsed).toNanos();
      double cpuLevel = (nextCpuTimeNanos - previousCpuTimeNanos) / deltaNanos;

      SystemNetworkUsages systemNetworkUsages = null;
      if (collectSystemNetworkUsage) {
        systemNetworkUsages =
            NetworkMetricsCollector.instance().collectSystemNetworkUsages(deltaNanos);
      }

      double estimatedCpuUsage = 0;
      double estimatedMemoryUsageInMb = 0;
      if (collectResourceManagerEstimation) {
        estimatedCpuUsage = resourceEstimator.getUsedCPU();
        estimatedMemoryUsageInMb = resourceEstimator.getUsedMemoryInMb();
      }

      synchronized (this) {
        addRange(ProfilerTask.LOCAL_CPU_USAGE, previousElapsed, nextElapsed, cpuLevel);
        if (memoryUsage != -1) {
          double memoryUsageMb = (double) memoryUsage / (1024 * 1024);
          addRange(ProfilerTask.LOCAL_MEMORY_USAGE, previousElapsed, nextElapsed, memoryUsageMb);
        }
        addRange(ProfilerTask.SYSTEM_CPU_USAGE, previousElapsed, nextElapsed, systemUsage);
        addRange(
            ProfilerTask.SYSTEM_MEMORY_USAGE,
            previousElapsed,
            nextElapsed,
            (double) systemMemoryUsageMb);
        addRange(
            ProfilerTask.WORKERS_MEMORY_USAGE, previousElapsed, nextElapsed, workerMemoryUsageMb);
        if (loadAverage > 0) {
          addRange(ProfilerTask.SYSTEM_LOAD_AVERAGE, previousElapsed, nextElapsed, loadAverage);
        }
        if (pressureStallIo >= 0) {
          addRange(ProfilerTask.PRESSURE_STALL_IO, previousElapsed, nextElapsed, pressureStallIo);
        }
        if (pressureStallMemory >= 0) {
          addRange(
              ProfilerTask.PRESSURE_STALL_IO, previousElapsed, nextElapsed, pressureStallMemory);
        }
        if (systemNetworkUsages != null) {
          addRange(
              ProfilerTask.SYSTEM_NETWORK_UP_USAGE,
              previousElapsed,
              nextElapsed,
              systemNetworkUsages.megabitsSentPerSec());
          addRange(
              ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE,
              previousElapsed,
              nextElapsed,
              systemNetworkUsages.megabitsRecvPerSec());
        }

        addRange(
            ProfilerTask.MEMORY_USAGE_ESTIMATION,
            previousElapsed,
            nextElapsed,
            estimatedMemoryUsageInMb);
        addRange(
            ProfilerTask.CPU_USAGE_ESTIMATION, previousElapsed, nextElapsed, estimatedCpuUsage);
      }
      previousElapsed = nextElapsed;
      previousCpuTimeNanos = nextCpuTimeNanos;
    }
  }

  public void stopCollecting() {
    Preconditions.checkArgument(!stopLocalUsageCollection);
    stopLocalUsageCollection = true;
    interrupt();
  }

  synchronized void logCollectedData() {
    if (!profilingStarted) {
      return;
    }
    Preconditions.checkArgument(stopLocalUsageCollection);
    long endTimeNanos = System.nanoTime();
    long elapsedNanos = stopwatch.elapsed(TimeUnit.NANOSECONDS);
    long startTimeNanos = endTimeNanos - elapsedNanos;
    Duration profileStart = Duration.ofNanos(startTimeNanos);
    int len = (int) (elapsedNanos / BUCKET_DURATION.toNanos()) + 1;
    Profiler profiler = Profiler.instance();

    for (ProfilerTask task : timeSeries.keySet()) {
      profiler.logCounters(
          ImmutableMap.ofEntries(Map.entry(task, timeSeries.get(task).toDoubleArray(len))),
          profileStart,
          BUCKET_DURATION);
    }
    timeSeries = null;
  }

  private void addRange(
      ProfilerTask type, Duration previousElapsed, Duration nextElapsed, double value) {
    synchronized (this) {
      if (timeSeries == null) {
        return;
      }
      TimeSeries series = timeSeries.get(type);
      if (series != null) {
        series.addRange(previousElapsed, nextElapsed, value);
      }
    }
  }
}
