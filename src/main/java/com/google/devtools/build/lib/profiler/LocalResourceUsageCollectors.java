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

import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorMetric.FULL;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorMetric.SOME;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.CPU;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.IO;
import static com.google.devtools.build.lib.util.ResourceUsage.PressureStallIndicatorResource.MEMORY;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.ResourceEstimator;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.worker.WorkerProcessMetrics;
import com.google.devtools.build.lib.worker.WorkerProcessMetricsCollector;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.HashMap;
import java.util.function.BiConsumer;

/** An assortment of classes that collects various interesting metrics about the local system. */
public class LocalResourceUsageCollectors {
  private final BugReporter bugReporter;

  private final InMemoryGraph graph;
  private final WorkerProcessMetricsCollector workerProcessMetricsCollector;

  private final ResourceEstimator resourceEstimator;

  private final SystemNetworkStatsService systemNetworkStatsService;

  public LocalResourceUsageCollectors(
      BugReporter bugReporter,
      InMemoryGraph graph,
      WorkerProcessMetricsCollector workerProcessMetricsCollector,
      ResourceEstimator resourceEstimator,
      SystemNetworkStatsService systemNetworkStatsService) {
    this.bugReporter = bugReporter;
    this.graph = graph;
    this.workerProcessMetricsCollector = workerProcessMetricsCollector;
    this.resourceEstimator = resourceEstimator;
    this.systemNetworkStatsService = systemNetworkStatsService;
  }

  public void addCollectors(
      boolean collectWorkerDataInProfiler,
      boolean collectLoadAverage,
      boolean collectSystemNetworkUsage,
      boolean collectResourceManagerEstimation,
      boolean collectPressureStallIndicators,
      boolean collectSkyframeCounts) {
    Preconditions.checkState(
        !collectSkyframeCounts || graph != null,
        "--experimental_collect_skyframe_counts_in_profiler requires the Skyframe graph.");

    OperatingSystemMXBean osBean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    Profiler.instance().registerCounterSeriesCollector(new LocalCpuUsageCollector(osBean));
    Profiler.instance()
        .registerCounterSeriesCollector(new LocalMemoryUsageCollector(memoryBean, bugReporter));
    Profiler.instance().registerCounterSeriesCollector(new SystemCpuUsageCollector(osBean));
    Profiler.instance().registerCounterSeriesCollector(new SystemMemoryUsageCollector(osBean));

    if (collectWorkerDataInProfiler
        && (OS.getCurrent() == OS.LINUX || OS.getCurrent() == OS.DARWIN)) {
      // Enabling the WorkerMemoryUsageCollector will cause hangs on Windows. We should only enable
      // it on Linux and Darwin.
      Profiler.instance()
          .registerCounterSeriesCollector(
              new TotalWorkerMemoryUsageCollector(workerProcessMetricsCollector));
      Profiler.instance()
          .registerCounterSeriesCollector(
              new PerMnemonicWorkerMemoryUsageCollector(workerProcessMetricsCollector));
    }
    if (collectLoadAverage) {
      Profiler.instance().registerCounterSeriesCollector(new SystemLoadAverageCollector(osBean));
    }
    if (collectSystemNetworkUsage) {
      Profiler.instance()
          .registerCounterSeriesCollector(
              new SystemNetworkUsageCollector(systemNetworkStatsService));
    }
    if (collectResourceManagerEstimation) {
      Profiler.instance()
          .registerCounterSeriesCollector(
              new ResourceManagerEstimationCollector(resourceEstimator));
    }
    // The pressure stall indicators are only available on Linux.
    if (collectPressureStallIndicators && OS.getCurrent() == OS.LINUX) {
      Profiler.instance().registerCounterSeriesCollector(new PressureStallIndicatorCollector());
    }

    if (collectSkyframeCounts) {
      Profiler.instance().registerCounterSeriesCollector(new SkyframeCountsCollector(graph));
    }
  }

  static class LocalCpuUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask LOCAL_CPU_USAGE =
        new CounterSeriesTask("CPU usage (Bazel)", "cpu", CounterSeriesTask.Color.GOOD);

    private final OperatingSystemMXBean osBean;
    private long previousCpuTimeNanos;

    private LocalCpuUsageCollector(OperatingSystemMXBean osBean) {
      this.osBean = osBean;
      this.previousCpuTimeNanos = osBean.getProcessCpuTime();
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      long nextCpuTimeNanos = osBean.getProcessCpuTime();
      double cpuLevel = (nextCpuTimeNanos - previousCpuTimeNanos) / deltaNanos;
      previousCpuTimeNanos = nextCpuTimeNanos;
      consumer.accept(LOCAL_CPU_USAGE, cpuLevel);
    }
  }

  static class LocalMemoryUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask LOCAL_MEMORY_USAGE =
        new CounterSeriesTask("Memory usage (Bazel)", "memory", CounterSeriesTask.Color.OLIVE);
    private final MemoryMXBean memoryBean;
    private final BugReporter bugReporter;

    private LocalMemoryUsageCollector(MemoryMXBean memoryBean, BugReporter bugReporter) {
      this.memoryBean = memoryBean;
      this.bugReporter = bugReporter;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
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
      if (memoryUsage != -1) {
        memoryUsage = memoryUsage / (1024 * 1024);
        consumer.accept(LOCAL_MEMORY_USAGE, (double) memoryUsage);
      }
    }
  }

  static class SystemCpuUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask SYSTEM_CPU_USAGE =
        new CounterSeriesTask("CPU usage (total)", "system cpu", CounterSeriesTask.Color.RAIL_LOAD);
    private final OperatingSystemMXBean osBean;
    private final int numProcessors;

    private SystemCpuUsageCollector(OperatingSystemMXBean osBean) {
      this.osBean = osBean;
      this.numProcessors = Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      double systemCpuLoad;
      try {
        systemCpuLoad = osBean.getCpuLoad();
      } catch (NullPointerException unused) {
        // OperatingSystemMXBean.getCpuLoad() suffers from a TOCTOU issue on Linux that can
        // cause a NullPointerException. See https://github.com/bazelbuild/bazel/issues/24519 for
        // details.
        systemCpuLoad = 0;
      }
      if (Double.isNaN(systemCpuLoad)) {
        // Unlike advertised, on Mac the system CPU load is NaN sometimes.
        // There is no good way to handle this, so to avoid any downstream method crashing on
        // this,
        // we reset the CPU value here.
        systemCpuLoad = 0;
      }
      consumer.accept(SYSTEM_CPU_USAGE, systemCpuLoad * numProcessors);
    }
  }

  static class SystemMemoryUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask SYSTEM_MEMORY_USAGE =
        new CounterSeriesTask("Memory usage (total)", "system memory", CounterSeriesTask.Color.BAD);
    private final OperatingSystemMXBean osBean;

    private SystemMemoryUsageCollector(OperatingSystemMXBean osBean) {
      this.osBean = osBean;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
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
        // the OS bean.
        systemMemoryUsageMb =
            (osBean.getTotalPhysicalMemorySize() - osBean.getFreePhysicalMemorySize())
                / (1024 * 1024);
      }
      consumer.accept(SYSTEM_MEMORY_USAGE, (double) systemMemoryUsageMb);
    }
  }

  static class TotalWorkerMemoryUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask WORKERS_MEMORY_USAGE =
        new CounterSeriesTask(
            "Total worker memory usage", "workers memory", CounterSeriesTask.Color.RAIL_ANIMATION);
    private final WorkerProcessMetricsCollector workerProcessMetricsCollector;

    private TotalWorkerMemoryUsageCollector(
        WorkerProcessMetricsCollector workerProcessMetricsCollector) {
      this.workerProcessMetricsCollector = workerProcessMetricsCollector;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      int workerMemoryUsageMb = 0;
      try (SilentCloseable c = Profiler.instance().profile("Worker metrics collection")) {
        workerMemoryUsageMb =
            workerProcessMetricsCollector.getLiveWorkerProcessMetrics().stream()
                    .mapToInt(WorkerProcessMetrics::getUsedMemoryInKb)
                    .sum()
                / 1024;
      }
      consumer.accept(WORKERS_MEMORY_USAGE, (double) workerMemoryUsageMb);
    }
  }

  static class PerMnemonicWorkerMemoryUsageCollector implements CounterSeriesCollector {
    private static final HashMap<String, CounterSeriesTask> workerMemoryUsageSeries =
        new HashMap<>();
    private static final String SERIES_LANE_NAME = "Per-mnemonic worker memory usage";
    private final WorkerProcessMetricsCollector workerProcessMetricsCollector;

    private PerMnemonicWorkerMemoryUsageCollector(
        WorkerProcessMetricsCollector workerProcessMetricsCollector) {
      this.workerProcessMetricsCollector = workerProcessMetricsCollector;
    }

    private static CounterSeriesTask getWorkerMemoryUsageSeries(String mnemonic) {
      return workerMemoryUsageSeries.computeIfAbsent(
          mnemonic,
          key ->
              new CounterSeriesTask(
                  SERIES_LANE_NAME, mnemonic, CounterSeriesTask.Color.RAIL_ANIMATION));
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {

      workerProcessMetricsCollector
          .getLiveWorkerProcessMetrics()
          .forEach(
              collector -> {
                try (SilentCloseable c = Profiler.instance().profile(collector.getMnemonic())) {
                  consumer.accept(
                      getWorkerMemoryUsageSeries(collector.getMnemonic()),
                      (double) collector.getUsedMemoryInKb() / 1024);
                }
              });
    }
  }

  static class SystemLoadAverageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask SYSTEM_LOAD_AVERAGE =
        new CounterSeriesTask("System load average", "load", CounterSeriesTask.Color.GENERIC_WORK);
    private final OperatingSystemMXBean osBean;

    private SystemLoadAverageCollector(OperatingSystemMXBean osBean) {
      this.osBean = osBean;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      double loadAverage = osBean.getSystemLoadAverage();
      if (loadAverage > 0) {
        consumer.accept(SYSTEM_LOAD_AVERAGE, loadAverage);
      }
    }
  }

  static class SystemNetworkUsageCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask SYSTEM_NETWORK_UP_USAGE =
        new CounterSeriesTask(
            "Network Up usage (total)",
            "system network up (Mbps)",
            CounterSeriesTask.Color.RAIL_RESPONSE);
    private static final CounterSeriesTask SYSTEM_NETWORK_DOWN_USAGE =
        new CounterSeriesTask(
            "Network Down usage (total)",
            "system network down (Mbps)",
            CounterSeriesTask.Color.RAIL_RESPONSE);
    private final SystemNetworkStatsService systemNetworkStatsService;

    private SystemNetworkUsageCollector(SystemNetworkStatsService systemNetworkStatsService) {
      this.systemNetworkStatsService = systemNetworkStatsService;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      var systemNetworkUsages =
          NetworkMetricsCollector.instance()
              .collectSystemNetworkUsages(deltaNanos, systemNetworkStatsService);
      if (systemNetworkUsages != null) {
        consumer.accept(SYSTEM_NETWORK_UP_USAGE, systemNetworkUsages.megabitsSentPerSec());
        consumer.accept(SYSTEM_NETWORK_DOWN_USAGE, systemNetworkUsages.megabitsRecvPerSec());
      }
    }
  }

  static class ResourceManagerEstimationCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask MEMORY_USAGE_ESTIMATION =
        new CounterSeriesTask(
            "Memory usage estimation", "estimated memory", CounterSeriesTask.Color.RAIL_IDLE);
    private static final CounterSeriesTask CPU_USAGE_ESTIMATION =
        new CounterSeriesTask(
            "CPU usage estimation",
            "estimated cpu",
            CounterSeriesTask.Color.CQ_BUILD_ATTEMPT_PASSED);

    private final ResourceEstimator resourceEstimator;

    private ResourceManagerEstimationCollector(ResourceEstimator resourceEstimator) {
      this.resourceEstimator = resourceEstimator;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      double estimatedCpuUsage = resourceEstimator.getUsedCPU();
      double estimatedMemoryUsageInMb = resourceEstimator.getUsedMemoryInMb();
      consumer.accept(CPU_USAGE_ESTIMATION, estimatedCpuUsage);
      consumer.accept(MEMORY_USAGE_ESTIMATION, estimatedMemoryUsageInMb);
    }
  }

  static class PressureStallIndicatorCollector implements CounterSeriesCollector {
    private static final CounterSeriesTask PRESSURE_STALL_FULL_IO =
        new CounterSeriesTask(
            "I/O pressure stall level",
            "i/o pressure (full)",
            CounterSeriesTask.Color.RAIL_ANIMATION);
    private static final CounterSeriesTask PRESSURE_STALL_SOME_IO =
        new CounterSeriesTask(
            "I/O pressure stall level",
            "i/o pressure (some)",
            CounterSeriesTask.Color.CQ_BUILD_ATTEMPT_FAILED);
    private static final CounterSeriesTask PRESSURE_STALL_FULL_MEMORY =
        new CounterSeriesTask(
            "Memory pressure stall level",
            "memory pressure (full)",
            CounterSeriesTask.Color.THREAD_STATE_UNKNOWN);
    private static final CounterSeriesTask PRESSURE_STALL_SOME_MEMORY =
        new CounterSeriesTask(
            "Memory pressure stall level",
            "memory pressure (some)",
            CounterSeriesTask.Color.RAIL_IDLE);
    private static final CounterSeriesTask PRESSURE_STALL_SOME_CPU =
        new CounterSeriesTask(
            "CPU pressure stall level",
            "cpu pressure (some)",
            CounterSeriesTask.Color.THREAD_STATE_RUNNING);

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      // The pressure stall indicator for full CPU metric is not defined.
      double pressureStallFullIo = ResourceUsage.readPressureStallIndicator(IO, FULL);
      double pressureStallFullMemory = ResourceUsage.readPressureStallIndicator(MEMORY, FULL);
      double pressureStallSomeIo = ResourceUsage.readPressureStallIndicator(IO, SOME);
      double pressureStallSomeMemory = ResourceUsage.readPressureStallIndicator(MEMORY, SOME);
      double pressureStallSomeCpu = ResourceUsage.readPressureStallIndicator(CPU, SOME);

      consumer.accept(PRESSURE_STALL_FULL_IO, pressureStallFullIo);
      consumer.accept(PRESSURE_STALL_FULL_MEMORY, pressureStallFullMemory);
      consumer.accept(PRESSURE_STALL_SOME_IO, pressureStallSomeIo);
      consumer.accept(PRESSURE_STALL_SOME_MEMORY, pressureStallSomeMemory);
      consumer.accept(PRESSURE_STALL_SOME_CPU, pressureStallSomeCpu);
    }
  }

  static class SkyframeCountsCollector implements CounterSeriesCollector {
    private record SkyFunctionProfilerTasks(
        CounterSeriesTask totalCounter, CounterSeriesTask doneCounter) {}

    private static final ImmutableMap<SkyFunctionName, SkyFunctionProfilerTasks>
        SKYFUNCTION_PROFILER_TASKS =
            ImmutableMap.of(
                SkyFunctions.PACKAGE,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (PACKAGE)", "package (total)", /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (PACKAGE)", "package (done)", /* color= */ null)),
                SkyFunctions.BZL_LOAD,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (BZL_LOAD)", "bzl_load (total)", /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (BZL_LOAD)", "bzl_load (done)", /* color= */ null)),
                SkyFunctions.GLOB,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask("SkyFunction (GLOB)", "glob (total)", /* color= */ null),
                    new CounterSeriesTask("SkyFunction (GLOB)", "glob (done)", /* color= */ null)),
                SkyFunctions.GLOBS,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (GLOBS)", "globs (total)", /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (GLOBS)", "globs (done)", /* color= */ null)),
                SkyFunctions.CONFIGURED_TARGET,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (CONFIGURED_TARGET)",
                        "configured target (total)",
                        /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (CONFIGURED_TARGET)",
                        "configured target (done)",
                        /* color= */ null)),
                SkyFunctions.ASPECT,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (ASPECT)", "aspect (total)", /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (ASPECT)", "aspect (done)", /* color= */ null)),
                SkyFunctions.ACTION_EXECUTION,
                new SkyFunctionProfilerTasks(
                    new CounterSeriesTask(
                        "SkyFunction (ACTION_EXECUTION)",
                        "action execution (total)",
                        /* color= */ null),
                    new CounterSeriesTask(
                        "SkyFunction (ACTION_EXECUTION)",
                        "action execution (done)",
                        /* color= */ null)));

    private final InMemoryGraph graph;

    private SkyframeCountsCollector(InMemoryGraph graph) {
      this.graph = graph;
    }

    @Override
    public void collect(double deltaNanos, BiConsumer<CounterSeriesTask, Double> consumer) {
      Multiset<SkyFunctionName> skykeyDoneCounter = HashMultiset.create();
      Multiset<SkyFunctionName> skykeyCounter = HashMultiset.create();
      for (InMemoryNodeEntry entry : graph.getAllNodeEntries()) {
        SkyFunctionName name = entry.getKey().functionName();
        if (SKYFUNCTION_PROFILER_TASKS.containsKey(name)) {
          skykeyCounter.add(name);
          if (entry.isDone()) {
            skykeyDoneCounter.add(name);
          }
        }
      }
      for (var entry : SKYFUNCTION_PROFILER_TASKS.entrySet()) {
        SkyFunctionName functionName = entry.getKey();
        consumer.accept(
            entry.getValue().totalCounter(), (double) skykeyCounter.count(functionName));
        consumer.accept(
            entry.getValue().doneCounter(), (double) skykeyDoneCounter.count(functionName));
      }
    }
  }
}
