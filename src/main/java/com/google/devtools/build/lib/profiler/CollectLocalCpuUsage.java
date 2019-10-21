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
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/** Thread to collect local cpu usage data and log into JSON profile. */
public class CollectLocalCpuUsage extends Thread {
  // TODO(twerth): Make these configurable.
  private static final Duration BUCKET_DURATION = Duration.ofSeconds(1);
  private static final long LOCAL_CPU_SLEEP_MILLIS = 200;

  private volatile boolean stopCpuUsage;
  private volatile boolean profilingStarted;
  private TimeSeries localCpuUsage;
  private Stopwatch stopwatch;

  @Override
  public void run() {
    stopwatch = Stopwatch.createStarted();
    localCpuUsage =
        new TimeSeries(
            /* startTimeMillis= */ stopwatch.elapsed().toMillis(), BUCKET_DURATION.toMillis());
    OperatingSystemMXBean bean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    Duration previousElapsed = stopwatch.elapsed();
    long previousCpuTimeNanos = bean.getProcessCpuTime();
    profilingStarted = true;
    while (!stopCpuUsage) {
      try {
        Thread.sleep(LOCAL_CPU_SLEEP_MILLIS);
      } catch (InterruptedException e) {
        return;
      }
      Duration nextElapsed = stopwatch.elapsed();
      long nextCpuTimeNanos = bean.getProcessCpuTime();
      double deltaNanos = nextElapsed.minus(previousElapsed).toNanos();
      double cpuLevel = (nextCpuTimeNanos - previousCpuTimeNanos) / deltaNanos;
      localCpuUsage.addRange(previousElapsed.toMillis(), nextElapsed.toMillis(), cpuLevel);
      previousElapsed = nextElapsed;
      previousCpuTimeNanos = nextCpuTimeNanos;
    }
  }

  public void stopCollecting() {
    Preconditions.checkArgument(!stopCpuUsage);
    stopCpuUsage = true;
    interrupt();
  }

  public void logCollectedData() {
    if (!profilingStarted) {
      return;
    }
    Preconditions.checkArgument(stopCpuUsage);
    long endTimeNanos = System.nanoTime();
    long elapsedNanos = stopwatch.elapsed(TimeUnit.NANOSECONDS);
    long startTimeNanos = endTimeNanos - elapsedNanos;
    int len = (int) (elapsedNanos / BUCKET_DURATION.toNanos()) + 1;
    double[] localCpuUsageValues = localCpuUsage.toDoubleArray(len);
    Profiler profiler = Profiler.instance();
    for (int i = 0; i < len; i++) {
      long eventTimeNanos = startTimeNanos + i * BUCKET_DURATION.toNanos();
      profiler.logEventAtTime(
          eventTimeNanos, ProfilerTask.LOCAL_CPU_USAGE, String.valueOf(localCpuUsageValues[i]));
    }
    localCpuUsage = null;
  }
}
