// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.io.Files;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.ProcMeminfoParser;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class estimates the local host's resource capacity.
 */
@ThreadCompatible
public final class LocalHostCapacity {

  private static final Logger LOG = Logger.getLogger(LocalHostCapacity.class.getName());

  /**
   * Stores parsed /proc/stat CPU time counters.
   * See {@link LocalHostCapacity#getCpuTimes(String)} for details.
   */
  @Immutable
  private final static class CpuTimes {
    private final long idleJiffies;
    private final long totalJiffies;

    CpuTimes(long idleJiffies, long totalJiffies) {
      this.idleJiffies = idleJiffies;
      this.totalJiffies = totalJiffies;
    }

    /**
     * Return idle CPU ratio using current and previous CPU readings or 0 if
     * ratio is undefined.
     */
    double getIdleRatio(CpuTimes prevTimes) {
      if (prevTimes.totalJiffies == 0 || totalJiffies == prevTimes.totalJiffies) {
        return 0;
      }
      return ((double)(idleJiffies - prevTimes.idleJiffies) /
          (double)(totalJiffies - prevTimes.totalJiffies));
    }
  }

  /**
   * Used to store available local CPU and RAM resources information.
   * See {@link LocalHostCapacity#getFreeResources(FreeResources)} for details.
   */
  public static final class FreeResources {

    private final Clock clock;
    private final CpuTimes cpuTimes;
    private final long lastTimestamp;
    private final double freeCpu;
    private final double freeMb;
    private final long interval;

    private FreeResources(Clock localClock, ProcMeminfoParser memInfo, String statContent,
                          FreeResources prevStats) {
      clock = localClock;
      lastTimestamp = localClock.nanoTime();
      freeMb = ProcMeminfoParser.kbToMb(memInfo.getFreeRamKb());
      cpuTimes = getCpuTimes(statContent);
      if (prevStats == null) {
        interval = 0;
        freeCpu = 0.0;
      } else {
        interval = lastTimestamp - prevStats.lastTimestamp;
        freeCpu = getLocalHostCapacity().getCpuUsage() * cpuTimes.getIdleRatio(prevStats.cpuTimes);
      }
    }

    /**
     * Returns amount of available RAM in MB.
     */
    public double getFreeMb() { return freeMb; }

    /**
     * Returns average available CPU resources (as a fraction of the CPU core,
     * so one fully CPU-bound thread should consume exactly 1.0 CPU resource).
     */
    public double getAvgFreeCpu() { return freeCpu; }

    /**
     * Returns interval in ms between CPU load measurements used to calculate
     * average available CPU resources.
     */
    public long getInterval() { return interval / 1000000; }

    /**
     * Returns age of available resource data in ms.
     */
    public long getReadingAge() {
      return (clock.nanoTime() - lastTimestamp) / 1000000;
    }
  }

  // Disables getFreeResources() if error occured during reading or parsing
  // /proc/* information.
  @VisibleForTesting
  static boolean isDisabled;

  // If /proc/* information is not available, assume 3000 MB and 2 CPUs.
  private static ResourceSet DEFAULT_RESOURCES = ResourceSet.create(3000.0, 2.0, 1.0,
      Integer.MAX_VALUE);

  private LocalHostCapacity() {}

  /**
   * Estimates of the local host's resource capacity,
   * obtained by reading /proc/cpuinfo and /proc/meminfo.
   */
  private static ResourceSet localHostCapacity;

  /**
   * Estimates of the local host's resource capacity,
   * obtained by reading /proc/cpuinfo and /proc/meminfo.
   */
  public static ResourceSet getLocalHostCapacity() {
    if (localHostCapacity == null) {
      localHostCapacity = getLocalHostCapacity("/proc/cpuinfo", "/proc/meminfo");
    }
    return localHostCapacity;
  }

  /**
   * Returns new FreeResources object populated with free RAM information from
   * /proc/meminfo and CPU load information from the /proc/stat. First call
   * should be made with null parameter to instantiate new FreeResources object.
   * Subsequent calls will use information inside it to calculate average CPU
   * load over the time between calls and to calculate amount of free CPU
   * resources and generate new FreeResources() instance.
   *
   * If information is not available due to error, functionality will be disabled
   * and method will always return null.
   */
  public static FreeResources getFreeResources(FreeResources stats) {
    return getFreeResources(BlazeClock.instance(), "/proc/meminfo", "/proc/stat", stats);
  }

  private static final Splitter NEWLINE_SPLITTER = Splitter.on('\n').omitEmptyStrings();

  @VisibleForTesting
  static int getLogicalCpuCount(String cpuinfoContent) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoContent);
    int count = 0;
    for (String line : lines) {
      if(line.startsWith("processor")) {
        count++;
      }
    }
    if (count == 0) {
      throw new IllegalArgumentException("Can't locate processor in the /proc/cpuinfo");
    }
    return count;
  }

  @VisibleForTesting
  static int getPhysicalCpuCount(String cpuinfoContent, int logicalCpuCount) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoContent);
    Set<String> uniq = new HashSet<>();
    for (String line : lines) {
      if(line.startsWith("physical id")) {
        uniq.add(line);
      }
    }
    int physicalCpuCount = uniq.size();
    if (physicalCpuCount == 0) {
      physicalCpuCount = logicalCpuCount;
    }
    return physicalCpuCount;
  }

  @VisibleForTesting
  static int getCoresPerCpu(String cpuinfoFileContent) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoFileContent);
    Set<String> uniq = new HashSet<>();
    for (String line : lines) {
      if(line.startsWith("core id")) {
        uniq.add(line);
      }
    }
    int coresPerCpu = uniq.size();
    if (coresPerCpu == 0) {
      coresPerCpu = 1;
    }
    return coresPerCpu;
  }

  /**
   * Parses cpu line of the /proc/stats, calculates number of idle and total
   * CPU jiffies and returns CpuTimes instance with that information.
   *
   * Total CPU time includes <b>all</b> time reported to be spent by the CPUs,
   * including so-called "stolen" time - time spent by other VMs on the same
   * workstation.
   */
  private static CpuTimes getCpuTimes(String statContent) {
    String[] cpuStats = statContent.substring(0, statContent.indexOf('\n')).trim().split(" +");
    // Supported versions of /proc/stat (Linux kernel 2.6.x) must contain either
    // 9 or 10 fields:
    //   "cpu" utime ultime stime idle iowait irq softirq steal(since 2.6.11) 0
    // We are interested in total time (sum of all columns) and idle time.
    if (cpuStats.length < 9 | cpuStats.length > 10) {
      throw new IllegalArgumentException("Unrecognized /proc/stat format");
    }
    if (!cpuStats[0].equals("cpu")) {
      throw new IllegalArgumentException("/proc/stat does not start with cpu keyword");
    }
    long idleCpuJiffies = Long.parseLong(cpuStats[4]); // "idle" column.
    long totalJiffies = 0;
    for (int i = 1; i < cpuStats.length; i++) {
      totalJiffies += Long.parseLong(cpuStats[i]);
    }
    long totalCpuJiffies = totalJiffies;
    return new CpuTimes(idleCpuJiffies, totalCpuJiffies);
  }

  @VisibleForTesting
  static ResourceSet getLocalHostCapacity(String cpuinfoFile, String meminfoFile) {
    try {
      String cpuinfoContent = readContent(cpuinfoFile);
      ProcMeminfoParser memInfo = new ProcMeminfoParser(meminfoFile);
      int logicalCpuCount = getLogicalCpuCount(cpuinfoContent);
      int physicalCpuCount = getPhysicalCpuCount(cpuinfoContent, logicalCpuCount);
      int coresPerCpu = getCoresPerCpu(cpuinfoContent);
      int totalCores = coresPerCpu * physicalCpuCount;
      boolean hyperthreading = (logicalCpuCount != totalCores);
      double ramMb = ProcMeminfoParser.kbToMb(memInfo.getTotalKb());
      final double EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU = 0.6;
      return ResourceSet.create(
          ramMb,
          logicalCpuCount * (hyperthreading ? EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU : 1.0),
          1.0,
          Integer.MAX_VALUE);
    } catch (IOException | IllegalArgumentException e) {
      disableProcFsUse(e);
      return DEFAULT_RESOURCES;
    }
  }

  @VisibleForTesting
  static FreeResources getFreeResources(Clock localClock, String meminfoFile, String statFile,
                                        FreeResources prevStats) {
    if (isDisabled) { return null; }
    try {
      String statContent = readContent(statFile);
      return new FreeResources(localClock, new ProcMeminfoParser(meminfoFile),
                               statContent, prevStats);
    } catch (IOException | IllegalArgumentException e) {
      disableProcFsUse(e);
      return null;
    }
  }

  /**
   * For testing purposes only. Do not use it.
   */
  @VisibleForTesting
  static void setLocalHostCapacity(ResourceSet resources) {
    localHostCapacity = resources;
    isDisabled = false;
  }

  private static String readContent(String filename) throws IOException {
    return Files.toString(new File(filename), Charset.defaultCharset());
  }

  /**
   * Disables use of /proc filesystem. Called internally when unexpected
   * exception is caught.
   */
  private static void disableProcFsUse(Throwable cause) {
    LoggingUtil.logToRemote(Level.WARNING, "Unable to read system load or capacity", cause);
    LOG.log(Level.WARNING, "Unable to read system load or capacity", cause);
    isDisabled = true;
  }
}
