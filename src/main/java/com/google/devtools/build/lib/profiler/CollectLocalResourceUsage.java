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
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorMetric.FULL;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorMetric.SOME;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.CPU;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.IO;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.MEMORY;

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
import com.google.devtools.build.lib.worker.WorkerProcessMetrics;
import com.google.devtools.build.lib.worker.WorkerProcessMetricsCollector;
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
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** Thread to collect local resource usage data and log into JSON profile. */
public class CollectLocalResourceUsage implements LocalResourceCollector {

  // TODO(twerth): Make these configurable.
  private static final Duration BUCKET_DURATION = Duration.ofSeconds(1);
  private static final Duration LOCAL_RESOURCES_COLLECT_SLEEP_INTERVAL = Duration.ofMillis(200);

  private final BugReporter bugReporter;
  private final boolean collectLoadAverage;
  private final boolean collectSystemNetworkUsage;
  private final boolean collectResourceManagerEstimation;

  private volatile boolean stopLocalUsageCollection;
  private volatile boolean profilingStarted;

  @GuardedBy("this")
  @Nullable
  private Map<ProfilerTask, TimeSeries> timeSeries;

  @GuardedBy("this")
  @Nullable
  private List<List<ProfilerTask>> stackedTaskGroups;

  private Stopwatch stopwatch;

  private final WorkerProcessMetricsCollector workerProcessMetricsCollector;

  private final ResourceEstimator resourceEstimator;
  private final boolean collectPressureStallIndicators;

  private Collector collector;

  public CollectLocalResourceUsage(
      BugReporter bugReporter,
      WorkerProcessMetricsCollector workerProcessMetricsCollector,
      ResourceEstimator resourceEstimator,
      boolean collectLoadAverage,
      boolean collectSystemNetworkUsage,
      boolean collectResourceManagerEstimation,
      boolean collectPressureStallIndicators) {
    this.bugReporter = checkNotNull(bugReporter);
    this.workerProcessMetricsCollector = workerProcessMetricsCollector;
    this.collectLoadAverage = collectLoadAverage;
    this.collectSystemNetworkUsage = collectSystemNetworkUsage;
    this.collectResourceManagerEstimation = collectResourceManagerEstimation;
    this.resourceEstimator = resourceEstimator;
    this.collectPressureStallIndicators = collectPressureStallIndicators;
    this.collector = new Collector();
  }

  @Override
  public void start() {
    collector.setDaemon(true);
    collector.start();
  }

  /** Thread that does the collection. */
  private class Collector extends Thread {

    Collector() {
      super("collect-local-resources");
    }

    @Override
    public void run() {
      int numProcessors = Runtime.getRuntime().availableProcessors();
      stopwatch = Stopwatch.createStarted();
      synchronized (CollectLocalResourceUsage.this) {
        timeSeries = new HashMap<>();
        stackedTaskGroups = new ArrayList<>();
        Duration startTime = stopwatch.elapsed();
        List<ProfilerTask> enabledCounters = new ArrayList<>();
        enabledCounters.addAll(
            ImmutableList.of(
                ProfilerTask.LOCAL_CPU_USAGE,
                ProfilerTask.LOCAL_MEMORY_USAGE,
                ProfilerTask.SYSTEM_CPU_USAGE,
                ProfilerTask.SYSTEM_MEMORY_USAGE,
                ProfilerTask.WORKERS_MEMORY_USAGE));

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
          enabledCounters.add(ProfilerTask.PRESSURE_STALL_FULL_IO);
          enabledCounters.add(ProfilerTask.PRESSURE_STALL_FULL_MEMORY);
          enabledCounters.add(ProfilerTask.PRESSURE_STALL_SOME_IO);
          enabledCounters.add(ProfilerTask.PRESSURE_STALL_SOME_MEMORY);
          enabledCounters.add(ProfilerTask.PRESSURE_STALL_SOME_CPU);

          // There is no PRESSURE_STALL_FULL_CPU metric, so it's not a stacked counter.
          stackedTaskGroups.add(
              ImmutableList.of(
                  ProfilerTask.PRESSURE_STALL_FULL_IO, ProfilerTask.PRESSURE_STALL_SOME_IO));
          stackedTaskGroups.add(
              ImmutableList.of(
                  ProfilerTask.PRESSURE_STALL_FULL_MEMORY,
                  ProfilerTask.PRESSURE_STALL_SOME_MEMORY));
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
          // There is no good way to handle this, so to avoid any downstream method crashing on
          // this,
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
          // In case we aren't running on Linux or /proc/meminfo parsing went wrong, fall back to
          // the
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
        try (SilentCloseable c = Profiler.instance().profile("Worker metrics collection")) {
          if (OS.getCurrent() == OS.LINUX || OS.getCurrent() == OS.DARWIN) {
            workerMemoryUsageMb =
                workerProcessMetricsCollector.getLiveWorkerProcessMetrics().stream()
                        .mapToInt(WorkerProcessMetrics::getUsedMemoryInKb)
                        .sum()
                    / 1024;
          }
        }
        double loadAverage = 0;
        if (collectLoadAverage) {
          loadAverage = osBean.getSystemLoadAverage();
        }

        // The pressure stall indicator for full CPU metric is not defined.
        double pressureStallFullIo = 0;
        double pressureStallFullMemory = 0;
        double pressureStallSomeIo = 0;
        double pressureStallSomeMemory = 0;
        double pressureStallSomeCpu = 0;
        // The pressure stall indicators are only available on Linux.
        if (collectPressureStallIndicators && OS.getCurrent() == OS.LINUX) {
          pressureStallFullIo = ResourceUsage.readPressureStallIndicator(IO, FULL);
          pressureStallFullMemory = ResourceUsage.readPressureStallIndicator(MEMORY, FULL);

          pressureStallSomeIo = ResourceUsage.readPressureStallIndicator(IO, SOME);
          pressureStallSomeMemory = ResourceUsage.readPressureStallIndicator(MEMORY, SOME);
          pressureStallSomeCpu = ResourceUsage.readPressureStallIndicator(CPU, SOME);
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

        synchronized (CollectLocalResourceUsage.this) {
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
          if (workerMemoryUsageMb > 0) {
            addRange(
                ProfilerTask.WORKERS_MEMORY_USAGE,
                previousElapsed,
                nextElapsed,
                workerMemoryUsageMb);
          }
          if (loadAverage > 0) {
            addRange(ProfilerTask.SYSTEM_LOAD_AVERAGE, previousElapsed, nextElapsed, loadAverage);
          }
          addRange(
              ProfilerTask.PRESSURE_STALL_SOME_CPU,
              previousElapsed,
              nextElapsed,
              pressureStallSomeCpu);
          addRange(
              ProfilerTask.PRESSURE_STALL_SOME_IO,
              previousElapsed,
              nextElapsed,
              pressureStallSomeIo);
          addRange(
              ProfilerTask.PRESSURE_STALL_FULL_IO,
              previousElapsed,
              nextElapsed,
              pressureStallFullIo);
          addRange(
              ProfilerTask.PRESSURE_STALL_SOME_MEMORY,
              previousElapsed,
              nextElapsed,
              pressureStallSomeMemory);
          addRange(
              ProfilerTask.PRESSURE_STALL_FULL_MEMORY,
              previousElapsed,
              nextElapsed,
              pressureStallFullMemory);
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
  }

  @Override
  public void stop() {
    if (collector != null) {
      Preconditions.checkArgument(!stopLocalUsageCollection);
      stopLocalUsageCollection = true;
      collector.interrupt();
      try {
        collector.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      logCollectedData();
      collector = null;
    }
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
      if (isStacked(task)) {
        continue;
      }
      profiler.logCounters(
          ImmutableMap.ofEntries(Map.entry(task, timeSeries.get(task).toDoubleArray(len))),
          profileStart,
          BUCKET_DURATION);
    }

    for (List<ProfilerTask> taskGroup : stackedTaskGroups) {
      ImmutableMap.Builder<ProfilerTask, double[]> stackedCounters = ImmutableMap.builder();
      for (ProfilerTask task : taskGroup) {
        stackedCounters.put(task, timeSeries.get(task).toDoubleArray(len));
      }
      profiler.logCounters(stackedCounters.buildOrThrow(), profileStart, BUCKET_DURATION);
    }

    timeSeries = null;
    stackedTaskGroups = null;
  }

  private synchronized boolean isStacked(ProfilerTask type) {
    for (List<ProfilerTask> tasks : stackedTaskGroups) {
      if (tasks.contains(type)) {
        return true;
      }
    }
    return false;
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
