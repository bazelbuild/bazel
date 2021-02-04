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

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/** Thread to collect local resource usage data and log into JSON profile. */
public class CollectLocalResourceUsage extends Thread {
  // TODO(twerth): Make these configurable.
  private static final Duration BUCKET_DURATION = Duration.ofSeconds(1);
  private static final long LOCAL_CPU_SLEEP_MILLIS = 200;

  private volatile boolean stopLocalUsageCollection;
  private volatile boolean profilingStarted;

  @GuardedBy("this")
  private TimeSeries localCpuUsage;

  @GuardedBy("this")
  private TimeSeries localMemoryUsage;

  private Stopwatch stopwatch;

  @Override
  public void run() {
    stopwatch = Stopwatch.createStarted();
    synchronized (this) {
      localCpuUsage =
          new TimeSeries(
              /* startTimeMillis= */ stopwatch.elapsed().toMillis(), BUCKET_DURATION.toMillis());
      localMemoryUsage =
          new TimeSeries(
              /* startTimeMillis= */ stopwatch.elapsed().toMillis(), BUCKET_DURATION.toMillis());
    }
    OperatingSystemMXBean osBean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    Duration previousElapsed = stopwatch.elapsed();
    long previousCpuTimeNanos = osBean.getProcessCpuTime();
    profilingStarted = true;
    while (!stopLocalUsageCollection) {
      try {
        Thread.sleep(LOCAL_CPU_SLEEP_MILLIS);
      } catch (InterruptedException e) {
        return;
      }
      Duration nextElapsed = stopwatch.elapsed();
      long nextCpuTimeNanos = osBean.getProcessCpuTime();
      long memoryUsage =
          memoryBean.getHeapMemoryUsage().getUsed() + memoryBean.getNonHeapMemoryUsage().getUsed();
      double deltaNanos = nextElapsed.minus(previousElapsed).toNanos();
      double cpuLevel = (nextCpuTimeNanos - previousCpuTimeNanos) / deltaNanos;
      synchronized (this) {
        if (localCpuUsage != null) {
          localCpuUsage.addRange(previousElapsed.toMillis(), nextElapsed.toMillis(), cpuLevel);
        }
        if (localMemoryUsage != null) {
          long memoryUsageMb = memoryUsage / (1024 * 1024);
          localMemoryUsage.addRange(
              previousElapsed.toMillis(), nextElapsed.toMillis(), memoryUsageMb);
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

  public synchronized void logCollectedData() {
    if (!profilingStarted) {
      return;
    }
    Preconditions.checkArgument(stopLocalUsageCollection);
    long endTimeNanos = System.nanoTime();
    long elapsedNanos = stopwatch.elapsed(TimeUnit.NANOSECONDS);
    long startTimeNanos = endTimeNanos - elapsedNanos;
    int len = (int) (elapsedNanos / BUCKET_DURATION.toNanos()) + 1;
    Profiler profiler = Profiler.instance();

    logCollectedData(profiler, localCpuUsage, ProfilerTask.LOCAL_CPU_USAGE, startTimeNanos, len);
    localCpuUsage = null;

    logCollectedData(
        profiler, localMemoryUsage, ProfilerTask.LOCAL_MEMORY_USAGE, startTimeNanos, len);
    localMemoryUsage = null;
  }

  private static void logCollectedData(
      Profiler profiler, TimeSeries timeSeries, ProfilerTask type, long startTimeNanos, int len) {
    double[] localCpuUsageValues = timeSeries.toDoubleArray(len);
    for (int i = 0; i < len; i++) {
      long eventTimeNanos = startTimeNanos + i * BUCKET_DURATION.toNanos();
      profiler.logEventAtTime(eventTimeNanos, type, String.valueOf(localCpuUsageValues[i]));
    }
  }
}
