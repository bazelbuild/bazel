// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.io.Files;
import com.sun.management.OperatingSystemMXBean;
import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.Iterator;

/**
 * Provides methods to measure the current resource usage of the current
 * process. Also provides some convenience methods to obtain several system
 * characteristics, like number of processors , total memory, etc.
 */
public final class ResourceUsage {

  /*
   * Use com.sun.management.OperatingSystemMXBean instead of
   * java.lang.management.OperatingSystemMXBean because the latter does not
   * support getTotalPhysicalMemorySize() and getFreePhysicalMemorySize().
   */
  private static final OperatingSystemMXBean OS_BEAN =
      (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

  private static final MemoryMXBean MEM_BEAN = ManagementFactory.getMemoryMXBean();
  private static final Splitter WHITESPACE_SPLITTER = Splitter.on(CharMatcher.whitespace());

  /**
   * Calculates an estimate of the current total CPU usage and the CPU usage of
   * the process in percent measured from the two given measurements. The
   * returned CPU usages rea average values for the time between the two
   * measurements. The returned array contains the total CPU usage at index 0
   * and the CPU usage of the measured process at index 1.
   */
  public static float[] calculateCurrentCpuUsage(Measurement oldMeasurement,
      Measurement newMeasurement) {
    if (oldMeasurement == null) {
      return new float[2];
    }
    long idleJiffies =
        newMeasurement.getTotalCpuIdleTimeInJiffies()
            - oldMeasurement.getTotalCpuIdleTimeInJiffies();
    long oldProcessJiffies =
        oldMeasurement.getCpuUtilizationInJiffies()[0]
            + oldMeasurement.getCpuUtilizationInJiffies()[1];
    long newProcessJiffies =
        newMeasurement.getCpuUtilizationInJiffies()[0]
            + newMeasurement.getCpuUtilizationInJiffies()[1];
    long processJiffies = newProcessJiffies - oldProcessJiffies;
    long elapsedTimeJiffies =
        newMeasurement.getTimeInJiffies() - oldMeasurement.getTimeInJiffies();
    int processors = getAvailableProcessors();
    // TODO(bazel-team): Sometimes smaller then zero. Not sure why.
    double totalUsage = Math.max(0, 1.0D - (double) idleJiffies / elapsedTimeJiffies / processors);
    double usage = Math.max(0, (double) processJiffies / elapsedTimeJiffies / processors);
    return new float[] {(float) totalUsage * 100, (float) usage * 100};
  }

  private ResourceUsage() {
  }

  /**
   * Returns the number of processors available to the Java virtual machine.
   */
  public static int getAvailableProcessors() {
    return OS_BEAN.getAvailableProcessors();
  }

  /**
   * Returns the total physical memory in bytes.
   */
  public static long getTotalPhysicalMemorySize() {
    return OS_BEAN.getTotalPhysicalMemorySize();
  }

  /**
   * Returns the operating system architecture.
   */
  public static String getOsArchitecture() {
    return OS_BEAN.getArch();
  }

  /**
   * Returns the operating system name.
   */
  public static String getOsName() {
    return OS_BEAN.getName();
  }

  /**
   * Returns the operating system version.
   */
  public static String getOsVersion() {
    return OS_BEAN.getVersion();
  }

  /**
   * Returns the initial size of heap memory in bytes.
   *
   * @see MemoryMXBean#getHeapMemoryUsage()
   */
  public static long getHeapMemoryInit() {
    return MEM_BEAN.getHeapMemoryUsage().getInit();
  }

  /**
   * Returns the initial size of non heap memory in bytes.
   *
   * @see MemoryMXBean#getNonHeapMemoryUsage()
   */
  public static long getNonHeapMemoryInit() {
    return MEM_BEAN.getNonHeapMemoryUsage().getInit();
  }

  /**
   * Returns the maximum size of heap memory in bytes.
   *
   * @see MemoryMXBean#getHeapMemoryUsage()
   */
  public static long getHeapMemoryMax() {
    return MEM_BEAN.getHeapMemoryUsage().getMax();
  }

  /**
   * Returns the maximum size of non heap memory in bytes.
   *
   * @see MemoryMXBean#getNonHeapMemoryUsage()
   */
  public static long getNonHeapMemoryMax() {
    return MEM_BEAN.getNonHeapMemoryUsage().getMax();
  }

  /**
   * Returns a measurement of the current resource usage of the current process.
   */
  public static Measurement measureCurrentResourceUsage() {
    return measureCurrentResourceUsage("self");
  }

  /**
   * Returns a measurement of the current resource usage of the process with the
   * given process id.
   *
   * @param processId the process id or <code>self</code> for the current
   *        process.
   */
  public static Measurement measureCurrentResourceUsage(String processId) {
    return new Measurement(MEM_BEAN.getHeapMemoryUsage().getUsed(), MEM_BEAN.getHeapMemoryUsage()
        .getCommitted(), MEM_BEAN.getNonHeapMemoryUsage().getUsed(), MEM_BEAN
        .getNonHeapMemoryUsage().getCommitted(), (float) OS_BEAN.getSystemLoadAverage(), OS_BEAN
        .getFreePhysicalMemorySize(), getCurrentTotalIdleTimeInJiffies(),
        getCurrentCpuUtilizationInJiffies(processId));
  }

  /**
   * Returns the current total idle time of the processors since system boot.
   * Reads /proc/stat to obtain this information.
   */
  private static long getCurrentTotalIdleTimeInJiffies() {
    try {
      File file = new File("/proc/stat");
      String content = Files.asCharSource(file, US_ASCII).read();
      String value = Iterables.get(WHITESPACE_SPLITTER.split(content), 5);
      return Long.parseLong(value);
    } catch (NumberFormatException | IOException e) {
      return 0L;
    }
  }

  /**
   * Returns the current cpu utilization of the current process with the given
   * id in jiffies. The returned array contains the following information: The
   * 1st entry is the number of jiffies that the process has executed in user
   * mode, and the 2nd entry is the number of jiffies that the process has
   * executed in kernel mode. Reads /proc/self/stat to obtain this information.
   *
   * @param processId the process id or <code>self</code> for the current
   *        process.
   */
  private static long[] getCurrentCpuUtilizationInJiffies(String processId) {
    try {
      File file = new File("/proc/" + processId + "/stat");
      if (file.isDirectory()) {
        return new long[2];
      }
      Iterator<String> stat =
          WHITESPACE_SPLITTER.split(Files.asCharSource(file, US_ASCII).read()).iterator();
      for (int i = 0; i < 13; ++i) {
        stat.next();
      }
      long token13 = Long.parseLong(stat.next());
      long token14 = Long.parseLong(stat.next());
      return new long[] { token13, token14 };
    } catch (NumberFormatException | IOException e) {
      return new long[2];
    }
  }

  /**
   * A snapshot of the resource usage of the current process at a point in time.
   */
  public static class Measurement {

    private final long timeInNanos;
    private final long heapMemoryUsed;
    private final long heapMemoryCommitted;
    private final long nonHeapMemoryUsed;
    private final long nonHeapMemoryCommitted;
    private final float loadAverageLastMinute;
    private final long freePhysicalMemory;
    private final long totalCpuIdleTimeInJiffies;
    private final long[] cpuUtilizationInJiffies;

    public Measurement(long heapMemoryUsed, long heapMemoryCommitted, long nonHeapMemoryUsed,
        long nonHeapMemoryCommitted, float loadAverageLastMinute, long freePhysicalMemory,
        long totalCpuIdleTimeInJiffies, long[] cpuUtilizationInJiffies) {
      super();
      timeInNanos = System.nanoTime();
      this.heapMemoryUsed = heapMemoryUsed;
      this.heapMemoryCommitted = heapMemoryCommitted;
      this.nonHeapMemoryUsed = nonHeapMemoryUsed;
      this.nonHeapMemoryCommitted = nonHeapMemoryCommitted;
      this.loadAverageLastMinute = loadAverageLastMinute;
      this.freePhysicalMemory = freePhysicalMemory;
      this.totalCpuIdleTimeInJiffies = totalCpuIdleTimeInJiffies;
      this.cpuUtilizationInJiffies = cpuUtilizationInJiffies;
    }

    /**
     * Returns the time of the measurement in jiffies.
     */
    public long getTimeInJiffies() {
      return timeInNanos / 10000000;
    }

    /**
     * Returns the time of the measurement in ms.
     */
    public long getTimeInMs() {
      return timeInNanos / 1000000;
    }

    /**
     * Returns the amount of used heap memory in bytes at the time of
     * measurement.
     *
     * @see MemoryMXBean#getHeapMemoryUsage()
     */
    public long getHeapMemoryUsed() {
      return heapMemoryUsed;
    }

    /**
     * Returns the amount of used non heap memory in bytes at the time of
     * measurement.
     *
     * @see MemoryMXBean#getNonHeapMemoryUsage()
     */
    public long getHeapMemoryCommitted() {
      return heapMemoryCommitted;
    }

    /**
     * Returns the amount of memory in bytes that is committed for the Java
     * virtual machine to use for the heap at the time of measurement.
     *
     * @see MemoryMXBean#getHeapMemoryUsage()
     */
    public long getNonHeapMemoryUsed() {
      return nonHeapMemoryUsed;
    }

    /**
     * Returns the amount of memory in bytes that is committed for the Java
     * virtual machine to use for non heap memory at the time of measurement.
     *
     * @see MemoryMXBean#getNonHeapMemoryUsage()
     */
    public long getNonHeapMemoryCommitted() {
      return nonHeapMemoryCommitted;
    }

    /**
     * Returns the system load average for the last minute at the time of
     * measurement.
     *
     * @see OperatingSystemMXBean#getSystemLoadAverage()
     */
    public float getLoadAverageLastMinute() {
      return loadAverageLastMinute;
    }

    /**
     * Returns the free physical memmory in bytes at the time of measurement.
     */
    public long getFreePhysicalMemory() {
      return freePhysicalMemory;
    }

    /**
     * Returns the current total cpu idle since system boot in jiffies.
     */
    public long getTotalCpuIdleTimeInJiffies() {
      return totalCpuIdleTimeInJiffies;
    }

    /**
     * Returns the current cpu utilization of the current process in jiffies.
     * The returned array contains the following information: The 1st entry is
     * the number of jiffies that the process has executed in user mode, and the
     * 2nd entry is the number of jiffies that the process has executed in
     * kernel mode. Reads /proc/self/stat to obtain this information.
     */
    public long[] getCpuUtilizationInJiffies() {
      return cpuUtilizationInJiffies;
    }

    /**
     * Returns the current cpu utilization of the current process in ms. The
     * returned array contains the following information: The 1st entry is the
     * number of ms that the process has executed in user mode, and the 2nd
     * entry is the number of ms that the process has executed in kernel mode.
     * Reads /proc/self/stat to obtain this information.
     */
    public long[] getCpuUtilizationInMs() {
      return new long[] {cpuUtilizationInJiffies[0] * 10, cpuUtilizationInJiffies[1] * 10};
    }
  }
}
