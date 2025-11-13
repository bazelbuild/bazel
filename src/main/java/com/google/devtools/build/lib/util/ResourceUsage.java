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
import com.google.common.io.Files;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.sun.management.OperatingSystemMXBean;
import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.nio.charset.Charset;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Provides methods to measure the current resource usage of the current process. Also provides some
 * convenience methods to obtain several system characteristics, like number of processors , total
 * memory, etc.
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
  private static final Pattern PSI_AVG10_VALUE_PATTERN_FULL =
      Pattern.compile("^full avg10=([\\d.]+).*");
  private static final Pattern PSI_AVG10_VALUE_PATTERN_SOME =
      Pattern.compile("^some avg10=([\\d.]+).*");
  private static final String PSI_AVG10_START_FULL = "full avg10";
  private static final String PSI_AVG10_START_SOME = "some avg10";

  private ResourceUsage() {}

  /** Returns the number of processors available to the Java virtual machine. */
  public static int getAvailableProcessors() {
    return OS_BEAN.getAvailableProcessors();
  }

  /** Returns the total physical memory in bytes. */
  public static long getTotalPhysicalMemorySize() {
    return OS_BEAN.getTotalPhysicalMemorySize();
  }

  /** Returns the operating system architecture. */
  public static String getOsArchitecture() {
    return OS_BEAN.getArch();
  }

  /** Returns the operating system name. */
  public static String getOsName() {
    return OS_BEAN.getName();
  }

  /** Returns the operating system version. */
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

  /** Returns a measurement of the current resource usage of the current process. */
  public static Measurement measureCurrentResourceUsage() {
    return new Measurement(
        System.nanoTime(),
        MEM_BEAN.getHeapMemoryUsage().getUsed(),
        MEM_BEAN.getHeapMemoryUsage().getCommitted(),
        MEM_BEAN.getNonHeapMemoryUsage().getUsed(),
        MEM_BEAN.getNonHeapMemoryUsage().getCommitted(),
        (float) OS_BEAN.getSystemLoadAverage(),
        readPressureStallIndicator(
            PressureStallIndicatorResource.MEMORY, PressureStallIndicatorMetric.FULL),
        readPressureStallIndicator(
            PressureStallIndicatorResource.IO, PressureStallIndicatorMetric.FULL),
        getAvailableMemory(),
        getCurrentCpuUtilizationInMs());
  }

  /**
   * Returns the current cpu utilization of the current process with the given id in ms. The
   * returned array contains the following information: The 1st entry is the number of ms that the
   * process has executed in user mode, and the 2nd entry is the number of ms that the process has
   * executed in kernel mode. Reads /proc/self/stat to obtain this information. The values may not
   * have millisecond accuracy.
   */
  private static long[] getCurrentCpuUtilizationInMs() {
    try {
      File file = new File("/proc/self/stat");
      if (file.isDirectory() || !file.canRead()) {
        return new long[2];
      }
      List<String> stat =
          WHITESPACE_SPLITTER.splitToList(Files.asCharSource(file, US_ASCII).read());
      if (stat.size() < 15) {
        return new long[2]; // Tolerate malformed input.
      }
      // /proc/self/stat contains values in jiffies, which are 10 ms.
      return new long[] {Long.parseLong(stat.get(13)) * 10, Long.parseLong(stat.get(14)) * 10};
    } catch (NumberFormatException | IOException e) {
      return new long[2];
    }
  }

  /**
   * Reads the Pressure Staller Indicator file for a given type and returns the double value for
   * `avg10`, or -1 if we couldn't read that value.
   */
  public static float readPressureStallIndicator(
      PressureStallIndicatorResource resource, PressureStallIndicatorMetric metric) {
    String fileName = "/proc/pressure/" + resource.getResource();
    File procFile = new File(fileName);
    if (!procFile.canRead()) {
      return -1.0F;
    }
    try {
      List<String> lines = Files.readLines(procFile, Charset.defaultCharset());
      for (String line : lines) {
        switch (metric) {
          case FULL:
            // Tries to find a line in file with the `full` metrics
            if (!line.startsWith(PSI_AVG10_START_FULL)) {
              break;
            }
            Matcher fullMatcher = PSI_AVG10_VALUE_PATTERN_FULL.matcher(line);
            if (!fullMatcher.matches()) {
              return -1.0F;
            }
            return Float.parseFloat(fullMatcher.group(1));
          case SOME:
            // Tries to find a line in file with the `some` metrics
            if (!line.startsWith(PSI_AVG10_START_SOME)) {
              break;
            }
            Matcher someMatcher = PSI_AVG10_VALUE_PATTERN_SOME.matcher(line);
            if (!someMatcher.matches()) {
              return -1.0F;
            }
            return Float.parseFloat(someMatcher.group(1));
        }
      }
      return -1.0F;
    } catch (IOException e) {
      return -1.0F;
    }
  }

  /**
   * Represents a type of resource which pressure stall indicator could be collected.
   *
   * <p>Indicators for only this 3 types of resources are available in Linux machines.
   */
  public enum PressureStallIndicatorResource {
    MEMORY("memory"),
    IO("io"),
    CPU("cpu");

    private final String resource;

    PressureStallIndicatorResource(String resource) {
      this.resource = resource;
    }

    public String getResource() {
      return resource;
    }
  }

  /**
   * Represents a type of metric for pressure stall indicators. The "some" metric indicates the
   * share of time in which at least some tasks are stalled on a given resource. The "full" metric
   * indicates the share of time in which all non-idle tasks are stalled on a given resource
   * simultaneously. (CPU full is undefined at the system level, by default always zero)
   */
  public enum PressureStallIndicatorMetric {
    FULL("full"),
    SOME("some");

    private final String metric;

    PressureStallIndicatorMetric(String metric) {
      this.metric = metric;
    }

    public String getMetric() {
      return metric;
    }
  }

  public static long getAvailableMemory() {
    long availableMemory;
    try {
      // TODO(larsrc): Use control flow instead of execptions
      ProcMeminfoParser meminfo = new ProcMeminfoParser();
      // Convert to bytes so that the fallback units are consistent.
      availableMemory = meminfo.getFreeRamKb() << 10;
    } catch (IOException e) {
      // /proc/meminfo isn't available outside Linux. On OS X, the OperatingSystem bean returns the
      // number of free pages multiplied by the page size, which is still incorrect. What we really
      // want here is (vm_stats.inactive_count + vm_stats.free_count) * page_size, but Java gives us
      // only free.
      // Seems like some virtual Ganeti machines also have issues getting this.
      availableMemory = OS_BEAN.getFreePhysicalMemorySize();
    }
    return availableMemory;
  }

  /**
   * A snapshot of the resource usage of the current process at a point in time.
   *
   * @attr timeInNanos The time of the measurement in nanoseconds.
   * @attr heapMemoryUsed The amount of heap memory used in bytes.
   * @attr heapMemoryCommitted The amount of heap memory committed in bytes.
   * @attr nonHeapMemoryUsed The amount of non-heap memory used in bytes.
   * @attr nonHeapMemoryCommitted The amount of non-heap memory committed in bytes.
   * @attr loadAverageLastMinute The load average of the system in the last minute.
   * @attr memoryPressureLast10Sec The memory pressure from the Linux Pressure Stall Indicator
   *     system, or -1 if PSI cannot be read.
   * @attr ioPressureLast10Sec The IO pressure from the Linux Pressure Stall Indicator system, or -1
   *     if PSI cannot be read.
   * @attr freePhysicalMemory The amount of free physical memory in bytes.
   * @attr cpuUtilizationInMs The current cpu utilization of the current process in ms. The returned
   *     array contains the following information: The 1st entry is the number of ms that the
   *     process has executed in user mode, and the 2nd entry is the number of ms that the process
   *     has executed in kernel mode. Reads /proc/self/stat to obtain this information.
   */
  @SuppressWarnings("ArrayRecordComponent")
  public record Measurement(
      long timeInNanos,
      long heapMemoryUsed,
      long heapMemoryCommitted,
      long nonHeapMemoryUsed,
      long nonHeapMemoryCommitted,
      float loadAverageLastMinute,
      float memoryPressureLast10Sec,
      float ioPressureLast10Sec,
      long freePhysicalMemory,
      long[] cpuUtilizationInMs) {

    /** Returns the time of the measurement in ms. */
    public long timeInMs() {
      return timeInNanos / 1000000;
    }

    @Override
    public long[] cpuUtilizationInMs() {
      return new long[] {cpuUtilizationInMs[0], cpuUtilizationInMs[1]};
    }
  }
}
