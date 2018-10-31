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
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;
import java.util.concurrent.TimeUnit;

/** Thread to collect local cpu usage data and log into JSON profile. */
public class CollectLocalCpuUsage extends Thread {
  // TODO(twerth): Make these configurable.
  private static final long BUCKET_SIZE_MILLIS = 1000;
  private static final long LOCAL_CPU_SLEEP_MILLIS = 200;

  private volatile boolean stopCpuUsage;
  private long cpuProfileStartMillis;
  private CpuUsageTimeSeries localCpuUsage;

  @Override
  public void run() {
    stopCpuUsage = false;
    cpuProfileStartMillis = System.currentTimeMillis();
    localCpuUsage = new CpuUsageTimeSeries(cpuProfileStartMillis, BUCKET_SIZE_MILLIS);
    OperatingSystemMXBean bean =
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    long previousTimeMillis = System.currentTimeMillis();
    long previousCpuTimeMillis = TimeUnit.NANOSECONDS.toMillis(bean.getProcessCpuTime());
    while (!stopCpuUsage) {
      try {
        Thread.sleep(LOCAL_CPU_SLEEP_MILLIS);
      } catch (InterruptedException e) {
        return;
      }
      long nextTimeMillis = System.currentTimeMillis();
      long nextCpuTimeMillis = TimeUnit.NANOSECONDS.toMillis(bean.getProcessCpuTime());
      double deltaMillis = nextTimeMillis - previousTimeMillis;
      double cpuLevel = (nextCpuTimeMillis - previousCpuTimeMillis) / deltaMillis;
      localCpuUsage.addRange(previousTimeMillis, nextTimeMillis, cpuLevel);
      previousTimeMillis = nextTimeMillis;
      previousCpuTimeMillis = nextCpuTimeMillis;
    }
  }

  public void stopCollecting() {
    Preconditions.checkArgument(!stopCpuUsage);
    stopCpuUsage = true;
    interrupt();
  }

  public void logCollectedData() {
    Preconditions.checkArgument(stopCpuUsage);
    long currentTimeNanos = System.nanoTime();
    long currentTimeMillis = System.currentTimeMillis();
    int len = (int) ((currentTimeMillis - cpuProfileStartMillis) / BUCKET_SIZE_MILLIS) + 1;
    double[] localCpuUsageValues = localCpuUsage.toDoubleArray(len);
    Profiler profiler = Profiler.instance();
    for (int i = 0; i < len; i++) {
      long timeMillis = cpuProfileStartMillis + i * BUCKET_SIZE_MILLIS;
      long timeNanos =
          TimeUnit.MILLISECONDS.toNanos(timeMillis - currentTimeMillis) + currentTimeNanos;
      profiler.logEventAtTime(
          timeNanos, ProfilerTask.LOCAL_CPU_USAGE, String.valueOf(localCpuUsageValues[i]));
    }
    localCpuUsage = null;
  }
}
