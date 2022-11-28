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
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.profiler.NetworkMetricsCollector.SystemNetworkUsages;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerMetric;
import com.google.devtools.build.lib.worker.WorkerMetricsCollector;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/** Thread to collect local resource usage data and log into JSON profile. */
public class CollectLocalResourceUsage extends Thread {

  // TODO(twerth): Make these configurable.
  private static final Duration BUCKET_DURATION = Duration.ofSeconds(1);
  private static final Duration LOCAL_RESOURCES_COLLECT_SLEEP_INTERVAL = Duration.ofMillis(200);

  private final BugReporter bugReporter;
  private final boolean collectWorkerDataInProfiler;
  private final boolean collectLoadAverage;
  private final boolean collectSystemNetworkUsage;

  private volatile boolean stopLocalUsageCollection;
  private volatile boolean profilingStarted;

  @GuardedBy("this")
  private TimeSeries localCpuUsage;

  @GuardedBy("this")
  private TimeSeries systemCpuUsage;

  @GuardedBy("this")
  private TimeSeries localMemoryUsage;

  @GuardedBy("this")
  private TimeSeries systemMemoryUsage;

  @GuardedBy("this")
  private TimeSeries workersMemoryUsage;

  @GuardedBy("this")
  private TimeSeries systemLoadAverage;

  @GuardedBy("this")
  private TimeSeries systemNetworkUpUsage;

  @GuardedBy("this")
  private TimeSeries systemNetworkDownUsage;

  private Stopwatch stopwatch;

  private final WorkerMetricsCollector workerMetricsCollector;

  CollectLocalResourceUsage(
      BugReporter bugReporter,
      WorkerMetricsCollector workerMetricsCollector,
      boolean collectWorkerDataInProfiler,
      boolean collectLoadAverage,
      boolean collectSystemNetworkUsage) {
    this.bugReporter = checkNotNull(bugReporter);
    this.collectWorkerDataInProfiler = collectWorkerDataInProfiler;
    this.workerMetricsCollector = workerMetricsCollector;
    this.collectLoadAverage = collectLoadAverage;
    this.collectSystemNetworkUsage = collectSystemNetworkUsage;
  }

  @Override
  public void run() {
    int numProcessors = Runtime.getRuntime().availableProcessors();
    stopwatch = Stopwatch.createStarted();
    synchronized (this) {
      Duration startTime = stopwatch.elapsed();
      localCpuUsage = new TimeSeries(startTime, BUCKET_DURATION);
      localMemoryUsage = new TimeSeries(startTime, BUCKET_DURATION);
      systemCpuUsage = new TimeSeries(startTime, BUCKET_DURATION);
      systemMemoryUsage = new TimeSeries(startTime, BUCKET_DURATION);
      if (collectWorkerDataInProfiler) {
        workersMemoryUsage = new TimeSeries(startTime, BUCKET_DURATION);
      }
      if (collectLoadAverage) {
        systemLoadAverage = new TimeSeries(startTime, BUCKET_DURATION);
      }
      if (collectSystemNetworkUsage) {
        systemNetworkUpUsage = new TimeSeries(startTime, BUCKET_DURATION);
        systemNetworkDownUsage = new TimeSeries(startTime, BUCKET_DURATION);
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
        workerMemoryUsageMb =
            this.workerMetricsCollector.collectMetrics().stream()
                    .map(WorkerMetric::getWorkerStat)
                    .filter(Objects::nonNull)
                    .mapToInt(WorkerMetric.WorkerStat::getUsedMemoryInKB)
                    .sum()
                / 1024;
      }
      double loadAverage = 0;
      if (collectLoadAverage) {
        loadAverage = osBean.getSystemLoadAverage();
      }

      double deltaNanos = nextElapsed.minus(previousElapsed).toNanos();
      double cpuLevel = (nextCpuTimeNanos - previousCpuTimeNanos) / deltaNanos;

      SystemNetworkUsages systemNetworkUsages = null;
      if (collectSystemNetworkUsage) {
        systemNetworkUsages =
            NetworkMetricsCollector.instance().collectSystemNetworkUsages(deltaNanos);
      }

      synchronized (this) {
        if (localCpuUsage != null) {
          localCpuUsage.addRange(previousElapsed, nextElapsed, cpuLevel);
        }
        if (localMemoryUsage != null && memoryUsage != -1) {
          long memoryUsageMb = memoryUsage / (1024 * 1024);
          localMemoryUsage.addRange(previousElapsed, nextElapsed, (double) memoryUsageMb);
        }
        if (systemCpuUsage != null) {
          systemCpuUsage.addRange(previousElapsed, nextElapsed, systemUsage);
        }
        if (systemMemoryUsage != null) {
          systemMemoryUsage.addRange(previousElapsed, nextElapsed, (double) systemMemoryUsageMb);
        }
        if (collectWorkerDataInProfiler && (workersMemoryUsage != null)) {
          workersMemoryUsage.addRange(previousElapsed, nextElapsed, workerMemoryUsageMb);
        }
        if (collectLoadAverage && (systemLoadAverage != null) && loadAverage > 0) {
          systemLoadAverage.addRange(previousElapsed, nextElapsed, loadAverage);
        }
        if (systemNetworkUsages != null) {
          systemNetworkUpUsage.addRange(
              previousElapsed, nextElapsed, systemNetworkUsages.megabitsSentPerSec());
          systemNetworkDownUsage.addRange(
              previousElapsed, nextElapsed, systemNetworkUsages.megabitsRecvPerSec());
        }
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

    profiler.logCounters(
        ProfilerTask.LOCAL_CPU_USAGE,
        localCpuUsage.toDoubleArray(len),
        profileStart,
        BUCKET_DURATION);
    localCpuUsage = null;

    profiler.logCounters(
        ProfilerTask.LOCAL_MEMORY_USAGE,
        localMemoryUsage.toDoubleArray(len),
        profileStart,
        BUCKET_DURATION);
    localMemoryUsage = null;

    profiler.logCounters(
        ProfilerTask.SYSTEM_CPU_USAGE,
        systemCpuUsage.toDoubleArray(len),
        profileStart,
        BUCKET_DURATION);
    systemCpuUsage = null;

    profiler.logCounters(
        ProfilerTask.SYSTEM_MEMORY_USAGE,
        systemMemoryUsage.toDoubleArray(len),
        profileStart,
        BUCKET_DURATION);
    systemMemoryUsage = null;

    if (collectWorkerDataInProfiler) {
      profiler.logCounters(
          ProfilerTask.WORKERS_MEMORY_USAGE,
          workersMemoryUsage.toDoubleArray(len),
          profileStart,
          BUCKET_DURATION);
    }
    workersMemoryUsage = null;

    if (collectLoadAverage) {
      profiler.logCounters(
          ProfilerTask.SYSTEM_LOAD_AVERAGE,
          systemLoadAverage.toDoubleArray(len),
          profileStart,
          BUCKET_DURATION);
    }
    systemLoadAverage = null;

    if (collectSystemNetworkUsage) {
      profiler.logCounters(
          ProfilerTask.SYSTEM_NETWORK_UP_USAGE,
          systemNetworkUpUsage.toDoubleArray(len),
          profileStart,
          BUCKET_DURATION);
      profiler.logCounters(
          ProfilerTask.SYSTEM_NETWORK_DOWN_USAGE,
          systemNetworkDownUsage.toDoubleArray(len),
          profileStart,
          BUCKET_DURATION);
    }
    systemNetworkUpUsage = null;
    systemNetworkDownUsage = null;
  }
}
